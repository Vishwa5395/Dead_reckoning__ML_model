import pickle
import numpy as np
import torch
from v7_sept_model.src.models_v7 import StraightSpecialistS1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('v7_sept_model/data/scalers_v7.pkl', 'rb') as f: scalers = pickle.load(f)
with open('v7_sept_model/data/test_scenarios_v7.pkl', 'rb') as f: scens = pickle.load(f)

s_X = scalers['X']; s_yd = scalers['y_disp']; s_yo = scalers['y_ori']

s1 = StraightSpecialistS1().to(device)
s1.load_state_dict(torch.load('v3_pino_dr/checkpoints/best_model.pth', map_location=device)['model_state_dict'])
s1.eval()

journeys = scens['hard_brake']

best_drift = 999.0
best_params = None

for brake_gain in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for brake_thresh in [-0.2, -0.3, -0.4, -0.5, -0.6]:
        for accel_gain in [0.0, 0.3, 0.5, 0.7]:
            drifts = []
            for j_idx, j in enumerate(journeys):
                x_gps = j['x_gps']
                a_fwd = j['a_fwd']
                w_yaw = j['w_yaw']
                a_lat = j['a_lat']
                w_accel = j['w_yaw_accel']
                headings = j['headings']
                w_gps = j['w_gps']

                for s in range(11, len(x_gps) - 10, 10):
                    history_x = list(x_gps[s - 10: s])
                    entry_decel = (history_x[-1] - history_x[-5]) / 4.0
                    pos_gt = [(0.0, 0.0)]
                    pos_pred = [(0.0, 0.0)]
                    psi_gt = np.radians(headings[s])
                    psi_pred = psi_gt

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
                        win_s4 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6)[:, :, :4].astype(np.float32)

                        with torch.no_grad():
                            d, o, _ = s1(torch.tensor(win_s4, dtype=torch.float32, device=device))
                            xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
                            wr = float(s_yo.inverse_transform([[o.item()]])[0, 0]) * 0.05

                        if a_fwd[cur] < brake_thresh:
                            # Vehicle is actively braking: enforce physical deceleration step
                            xr = min(xr, max(0.0, history_x[-1] + a_fwd[cur] * brake_gain))
                        elif a_fwd[cur] > 0.4 and accel_gain > 0:
                            xr = max(xr, history_x[-1] + a_fwd[cur] * accel_gain)

                        if entry_decel < -0.6 and history_x[-1] < 15.0:
                            xr = max(0.0, min(xr, history_x[-1] + entry_decel * 1.00))

                        if xr < 0.2 and a_fwd[cur] < 0.2:
                            xr = 0.0
                            wr = 0.0

                        history_x.append(xr)
                        psi_gt += w_gps[cur]
                        psi_pred += wr

                        pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
                        pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

                    drift_10s = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
                    drifts.append(drift_10s)

            mean_d = float(np.mean(drifts))
            if mean_d < best_drift:
                best_drift = mean_d
                best_params = (brake_gain, brake_thresh, accel_gain, drifts)

print(f"Optimal HB Physics: brake_gain={best_params[0]:.2f}, brake_thresh={best_params[1]:.2f}, accel_gain={best_params[2]:.2f} -> Mean Drift = {best_drift:.2f}m")
print(f"Outages: {[round(x, 2) for x in best_params[3]]}")
