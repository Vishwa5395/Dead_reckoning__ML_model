import pickle
import numpy as np
import torch
import json
from pathlib import Path
import sys

ROOT = Path("v6_smartphone_idr")
sys.path.insert(0, str(ROOT.parent))

from v6_smartphone_idr.src.models_v6 import PINODeadReckoningNetV6
from v6_smartphone_idr.src.physics_engine_v6 import VehiclePhysicsEngineV6

device = torch.device("cpu")
ckpt = torch.load("v6_smartphone_idr/checkpoints/best_model_v6.pth", map_location=device, weights_only=False)
model = PINODeadReckoningNetV6(in_channels=6, conv_channels=32, gru_hidden=32, dropout=0.15)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with open("v6_smartphone_idr/data/scalers_v6.pkl", "rb") as f:
    scalers = pickle.load(f)
with open("v6_smartphone_idr/data/test_scenarios_v6.pkl", "rb") as f:
    test_scenarios = pickle.load(f)

sharp = test_scenarios["sharp_turns"]
print(f"Loaded {len(sharp)} journeys in sharp_turns")

s_X = scalers["X"]
s_ydv = scalers["y_dv"]
s_yw = scalers["y_w"]

WINDOW_10HZ = 20
OUTAGE_10HZ = 100
DT = 0.1

for j_idx, j in enumerate(sharp[:3]):
    a_fwd = j["a_fwd"]
    w_yaw = j["w_yaw"]
    a_lat = j["a_lat"]
    w_accel = j["w_yaw_accel"]
    v_true = j["v_true"]
    w_true = j["w_true"]
    
    n = len(v_true)
    for start in range(WINDOW_10HZ + 10, min(n - OUTAGE_10HZ, 200), 100):
        history_v = list(v_true[start - WINDOW_10HZ: start])
        w_preds = []
        w_gyros = []
        w_gts = []
        for k in range(OUTAGE_10HZ):
            cur = start + k
            ch_a_fwd = np.clip(a_fwd[cur - WINDOW_10HZ + 1: cur + 1], -8.0, 8.0)
            ch_w_yaw = np.clip(w_yaw[cur - WINDOW_10HZ + 1: cur + 1], -1.2, 1.2)
            ch_a_lat = np.clip(a_lat[cur - WINDOW_10HZ + 1: cur + 1], -8.0, 8.0)
            ch_v_prev = np.clip(np.array(history_v[-WINDOW_10HZ:]), 0.0, 45.0)
            ch_w_accel = np.clip(w_accel[cur - WINDOW_10HZ + 1: cur + 1], -4.0, 4.0)
            ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)

            window = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
            win_scaled = s_X.transform(window.reshape(1, -1)).reshape(1, WINDOW_10HZ, 6).astype(np.float32)

            with torch.no_grad():
                dv_s, w_s, z_log, ba_s, bw_s, log_v = model(torch.tensor(win_scaled))

            w_pred = float(s_yw.inverse_transform([[w_s.item()]])[0, 0])
            dv_pred = float(s_ydv.inverse_transform([[dv_s.item()]])[0, 0])
            w_preds.append(w_pred)
            w_gyros.append(w_yaw[cur])
            w_gts.append(w_true[cur])
            history_v.append(max(0.0, history_v[-1] + dv_pred))

        max_turn = max(abs(x) for x in w_gts)
        print(f"Journey {j_idx} @ start {start}: Max true w = {max_turn:.3f} rad/s")
        print(f"  Mean GT |w|: {np.mean(np.abs(w_gts)):.3f}, Mean Pred |w|: {np.mean(np.abs(w_preds)):.3f}, Mean Gyro |w|: {np.mean(np.abs(w_gyros)):.3f}")
        print(f"  Cumul Angle GT: {sum(w_gts)*DT*180/np.pi:.1f} deg, Pred: {sum(w_preds)*DT*180/np.pi:.1f} deg, Gyro: {sum(w_gyros)*DT*180/np.pi:.1f} deg")
