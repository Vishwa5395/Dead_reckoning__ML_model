import pickle
import numpy as np
import torch
from v7_sept_model.src.models_v7 import TurningSpecialistS2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('v7_sept_model/data/scalers_v7.pkl', 'rb') as f: scalers = pickle.load(f)
with open('v7_sept_model/data/test_scenarios_v7.pkl', 'rb') as f: scens = pickle.load(f)

s_X = scalers['X']; s_yd = scalers['y_disp']; s_yo = scalers['y_ori']

s2_st = TurningSpecialistS2(in_channels=6).to(device)
ckpt_v4 = torch.load('v4_turn_focused/checkpoints/best_model_ablation_D_attn_coupling.pth', map_location=device)
s2_st.load_state_dict(ckpt_v4['model_state_dict'])
s2_st.eval()

journeys = scens['sharp_turns']

def test_sharp_turns(cent_mode="alat_only", cent_thresh=2.1, cent_gain=0.9, speed_decay=1.0):
    drifts = []
    for j_idx, j in enumerate(journeys):
        x_gps = j['x_gps']
        a_fwd = j['a_fwd']
        w_yaw = j['w_yaw']
        a_lat = j['a_lat']
        w_accel = j['w_yaw_accel']
        w_gps = j['w_gps']
        headings = j['headings']

        for s in range(11, len(x_gps) - 10, 10):
            history_x = list(x_gps[s - 10: s])
            pos_gt = [(0.0, 0.0)]
            pos_pred = [(0.0, 0.0)]
            psi_gt = np.radians(headings[s])
            psi_pred = psi_gt

            entry_decel = (history_x[-1] - history_x[-5]) / 4.0
            stopping_mode = (entry_decel < -0.5 and 0.8 < history_x[-1] < 7.5)

            for k in range(10):
                cur = s + k
                win_6 = np.stack([
                    np.clip(a_fwd[cur-9:cur+1], -8, 8),
                    np.clip(w_yaw[cur-9:cur+1], -1, 1),
                    np.clip(a_lat[cur-9:cur+1], -8, 8),
                    np.clip(np.array(history_x[-10:]), 0, 45),
                    np.clip(w_accel[cur-9:cur+1], -2, 2),
                    np.clip(a_lat[cur-9:cur+1] - np.array(history_x[-10:]) * w_yaw[cur-9:cur+1], -8, 8)
                ], axis=-1)
                win_s6 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

                with torch.no_grad():
                    d_p, o_p, _ = s2_st(torch.tensor(win_s6, dtype=torch.float32, device=device))
                    xr = float(s_yd.inverse_transform([[d_p.item()]])[0, 0])
                    wr = float(s_yo.inverse_transform([[o_p.item()]])[0, 0]) * 0.08
                    v_est = max(history_x[-1], 2.5)

                    if abs(a_lat[cur]) > cent_thresh:
                        if cent_mode == "alat_only":
                            w_cent = -np.sign(a_lat[cur]) * (abs(a_lat[cur]) / v_est) * cent_gain
                            wr += w_cent
                        elif cent_mode == "wyaw_sign":
                            # Use sign from gyro
                            w_cent = np.sign(w_yaw[cur]) * (abs(a_lat[cur]) / v_est) * cent_gain
                            wr += w_cent
                        elif cent_mode == "agreement":
                            # Only apply if -sign(alat) == sign(wyaw)
                            if -np.sign(a_lat[cur]) == np.sign(w_yaw[cur]):
                                w_cent = -np.sign(a_lat[cur]) * (abs(a_lat[cur]) / v_est) * cent_gain
                                wr += w_cent
                        xr = min(xr, max(1.5, history_x[-1] - speed_decay))

                    if stopping_mode:
                        xr = max(0.0, min(xr, history_x[-1] + entry_decel * 0.80))

                history_x.append(xr)
                psi_gt += w_gps[cur]
                psi_pred += wr

                pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
                pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

            drift = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
            drifts.append(drift)

    return float(np.mean(drifts))

print(f"Mode 'alat_only':  Mean = {test_sharp_turns('alat_only'):.2f}m")
print(f"Mode 'wyaw_sign':  Mean = {test_sharp_turns('wyaw_sign'):.2f}m")
print(f"Mode 'agreement':  Mean = {test_sharp_turns('agreement'):.2f}m")
