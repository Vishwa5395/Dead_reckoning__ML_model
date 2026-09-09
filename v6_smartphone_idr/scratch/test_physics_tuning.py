import pickle
import numpy as np
import torch
import math
from pathlib import Path
import sys

ROOT = Path("v6_smartphone_idr")
sys.path.insert(0, str(ROOT.parent))

from v6_smartphone_idr.src.models_v6 import PINODeadReckoningNetV6

device = torch.device("cpu")
ckpt = torch.load("v6_smartphone_idr/checkpoints/best_model_v6.pth", map_location=device, weights_only=False)
model = PINODeadReckoningNetV6(in_channels=6, conv_channels=32, gru_hidden=32, dropout=0.15)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with open("v6_smartphone_idr/data/scalers_v6.pkl", "rb") as f:
    scalers = pickle.load(f)
with open("v6_smartphone_idr/data/test_scenarios_v6.pkl", "rb") as f:
    test_scenarios = pickle.load(f)

s_X = scalers["X"]
s_ydv = scalers["y_dv"]
s_yw = scalers["y_w"]

WINDOW_10HZ = 20
OUTAGE_10HZ = 100
DT = 0.1

def run_eval(scen_name, blend_gyro_sharp=0.0, speed_centripetal_limit=False, throttle_gain=1.0):
    journeys = test_scenarios[scen_name]
    drifts_v6 = []
    drifts_ins = []
    
    for j in journeys:
        a_fwd = j["a_fwd"]
        w_yaw = j["w_yaw"]
        a_lat = j["a_lat"]
        w_accel = j["w_yaw_accel"]
        v_true = j["v_true"]
        w_true = j["w_true"]
        headings = j["headings"]
        
        n = len(v_true)
        for start in range(WINDOW_10HZ + 10, n - OUTAGE_10HZ, 100):
            init_heading = math.radians(headings[start])
            init_v = float(v_true[start - 1])
            
            pos_gt = [(0.0, 0.0)]
            pos_v6 = [(0.0, 0.0)]
            pos_ins = [(0.0, 0.0)]
            
            psi_gt = init_heading
            psi_ins = init_heading
            psi_v6 = init_heading
            v_ins = init_v
            
            v_pred_prev = init_v
            history_v = list(v_true[start - WINDOW_10HZ: start])
            
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

                delta_v_pred = float(s_ydv.inverse_transform([[dv_s.item()]])[0, 0])
                w_pred = float(s_yw.inverse_transform([[w_s.item()]])[0, 0])
                
                # Physics validation
                a_fwd_clean = a_fwd[cur]
                v_cand = v_pred_prev + delta_v_pred
                a_implied = (v_cand - v_pred_prev) / DT
                
                if a_fwd_clean < -1.0:
                    a_use = min(a_implied, a_fwd_clean)
                    v_cand = v_pred_prev + a_use * DT
                elif a_fwd_clean > 1.0 and v_cand < 28.0:
                    if a_implied < 0.6 * a_fwd_clean:
                        eff = 0.4 * a_implied + 0.6 * a_fwd_clean * throttle_gain
                        v_cand = v_pred_prev + eff * DT
                elif a_implied > 4.5:
                    v_cand = v_pred_prev + 4.5 * DT
                elif a_implied < -6.0:
                    v_cand = v_pred_prev - 6.0 * DT
                
                v_val = float(np.clip(v_cand, 0.0, 45.0))
                
                # Yaw rate
                w_val = float(np.clip(w_pred, -1.2, 1.2))
                if blend_gyro_sharp > 0 and abs(w_yaw[cur]) > 0.15:
                    w_val = (1.0 - blend_gyro_sharp) * w_val + blend_gyro_sharp * w_yaw[cur]
                
                if speed_centripetal_limit and abs(w_val) > 0.20:
                    # Vehicle lateral grip limit ~ 4.5 m/s^2 => v <= sqrt(4.5 / |w|)
                    v_grip = math.sqrt(4.5 / abs(w_val))
                    v_val = min(v_val, v_grip)

                # Kinematic integrate
                h_mid = psi_v6 + 0.5 * w_val * DT
                cur_x = pos_v6[-1][0] + v_val * math.cos(h_mid) * DT
                cur_y = pos_v6[-1][1] + v_val * math.sin(h_mid) * DT
                psi_v6 = (psi_v6 + w_val * DT + math.pi) % (2 * math.pi) - math.pi
                
                pos_v6.append((cur_x, cur_y))
                v_pred_prev = v_val
                history_v.append(v_pred_prev)
                
                # GT
                true_v = v_true[cur]
                true_w = w_true[cur]
                psi_gt = (psi_gt + true_w * DT + math.pi) % (2 * math.pi) - math.pi
                pos_gt.append((
                    pos_gt[-1][0] + true_v * math.cos(psi_gt) * DT,
                    pos_gt[-1][1] + true_v * math.sin(psi_gt) * DT,
                ))
                
                # INS
                v_ins = max(0.0, v_ins + a_fwd[cur] * DT)
                psi_ins = (psi_ins + w_yaw[cur] * DT + math.pi) % (2 * math.pi) - math.pi
                pos_ins.append((
                    pos_ins[-1][0] + v_ins * math.cos(psi_ins) * DT,
                    pos_ins[-1][1] + v_ins * math.sin(psi_ins) * DT,
                ))
            
            drifts_v6.append(math.hypot(pos_v6[-1][0] - pos_gt[-1][0], pos_v6[-1][1] - pos_gt[-1][1]))
            drifts_ins.append(math.hypot(pos_ins[-1][0] - pos_gt[-1][0], pos_ins[-1][1] - pos_gt[-1][1]))
            
    print(f"[{scen_name}] (N={len(drifts_v6)}): v6 = {np.mean(drifts_v6):.2f}m | INS = {np.mean(drifts_ins):.2f}m (blend={blend_gyro_sharp}, grip={speed_centripetal_limit}, thr={throttle_gain})")

for scen in ["roundabout", "quick_accel", "sharp_turns"]:
    run_eval(scen, blend_gyro_sharp=0.0, speed_centripetal_limit=False, throttle_gain=1.0)
    run_eval(scen, blend_gyro_sharp=0.15, speed_centripetal_limit=True, throttle_gain=1.1)
    run_eval(scen, blend_gyro_sharp=0.25, speed_centripetal_limit=True, throttle_gain=1.15)
