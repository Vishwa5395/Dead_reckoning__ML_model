import pickle
import torch
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = ROOT.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from v7_sept_model.src.models_v7 import StraightSpecialistS1, TurningSpecialistS2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(ROOT / "data" / "test_scenarios_v7.pkl", "rb") as f:
    test_scenarios = pickle.load(f)

with open(WS_ROOT / "v4_turn_focused" / "data" / "scalers_v4.pkl", "rb") as f:
    scalers = pickle.load(f)

s_X, s_yd, s_yo = scalers["X"], scalers["y_disp"], scalers["y_ori"]

exp0 = StraightSpecialistS1(in_channels=4).to(device)
c0 = torch.load(ROOT / "checkpoints" / "best_supreme_motorway.pth", map_location=device, weights_only=False)
exp0.load_state_dict(c0["model_state_dict"] if "model_state_dict" in c0 else c0)

exp1 = TurningSpecialistS2(in_channels=6).to(device)
c1 = torch.load(ROOT / "checkpoints" / "best_supreme_roundabout.pth", map_location=device, weights_only=False)
exp1.load_state_dict(c1["model_state_dict"] if "model_state_dict" in c1 else c1)

exp2 = TurningSpecialistS2(in_channels=6).to(device)
c2 = torch.load(ROOT / "checkpoints" / "best_supreme_quick_accel.pth", map_location=device, weights_only=False)
exp2.load_state_dict(c2["model_state_dict"] if "model_state_dict" in c2 else c2)

exp3 = StraightSpecialistS1(in_channels=4).to(device)
c3 = torch.load(ROOT / "checkpoints" / "best_supreme_hard_brake.pth", map_location=device, weights_only=False)
exp3.load_state_dict(c3["model_state_dict"] if "model_state_dict" in c3 else c3)

exp4 = TurningSpecialistS2(in_channels=6).to(device)
c4 = torch.load(ROOT / "checkpoints" / "best_supreme_sharp_turns.pth", map_location=device, weights_only=False)
exp4.load_state_dict(c4["model_state_dict"] if "model_state_dict" in c4 else c4)

models = [exp0, exp1, exp2, exp3, exp4]
for m in models:
    m.eval()

WINDOW = 10
OUTAGE = 10

def eval_one(m, scen, in_ch=6):
    journeys = test_scenarios[scen]
    drifts = []
    for j in journeys:
        x_gps, w_gps, headings = j["x_gps"], j["w_gps"], j["headings"]
        a_fwd, w_yaw, a_lat, w_accel = j["a_fwd"], j["w_yaw"], j["a_lat"], j["w_yaw_accel"]
        for s in range(WINDOW + 1, len(x_gps) - OUTAGE, 10):
            history_x = list(x_gps[s - WINDOW: s])
            pos_gt, pos_pred = [(0.0, 0.0)], [(0.0, 0.0)]
            psi_gt = np.radians(headings[s])
            psi_pred = psi_gt
            for k in range(OUTAGE):
                cur = s + k
                ch_a_fwd = np.clip(a_fwd[cur-9:cur+1], -8.0, 8.0)
                ch_w_yaw = np.clip(w_yaw[cur-9:cur+1], -1.0, 1.0)
                ch_a_lat = np.clip(a_lat[cur-9:cur+1], -8.0, 8.0)
                ch_v_prev = np.clip(np.array(history_x[-10:]), 0.0, 45.0)
                ch_w_accel = np.clip(w_accel[cur-9:cur+1], -2.0, 2.0)
                ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)
                win_6 = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
                win_s6 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)
                t_x = torch.tensor(win_s6, dtype=torch.float32, device=device)
                if in_ch == 4:
                    t_x = t_x[:, :, :4]
                with torch.no_grad():
                    dp, op, zp = m(t_x)
                xr = float(s_yd.inverse_transform([[dp.item()]])[0, 0])
                wr = float(s_yo.inverse_transform([[op.item()]])[0, 0])
                history_x.append(xr)
                psi_gt += w_gps[cur]
                psi_pred += wr
                pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
                pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))
            drifts.append(np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1]))
    return np.mean(drifts)

print("=== Dedicated Specialist Direct Results ===")
print(f"Motorway (exp0):    {eval_one(exp0, 'motorway', 4):.2f} m")
print(f"Roundabout (exp1):  {eval_one(exp1, 'roundabout', 6):.2f} m")
print(f"Quick Accel (exp2): {eval_one(exp2, 'quick_accel', 6):.2f} m")
print(f"Hard Brake (exp3):  {eval_one(exp3, 'hard_brake', 4):.2f} m")
print(f"Sharp Turns (exp4): {eval_one(exp4, 'sharp_turns', 6):.2f} m")
