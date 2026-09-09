import pickle
import numpy as np
import torch
from v7_sept_model.src.models_v7 import TurningSpecialistS2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("v7_sept_model/data/scalers_v7.pkl", "rb") as f:
    scalers = pickle.load(f)
with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

s_X = scalers["X"]
s_yd = scalers["y_disp"]
s_yo = scalers["y_ori"]
j = scens["roundabout"][0]

model = TurningSpecialistS2(in_channels=6).to(device)
ckpt = torch.load("v7_sept_model/checkpoints/best_supreme_roundabout.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

x_gps = j["x_gps"]
w_gps = j["w_gps"]
a_fwd = j["a_fwd"]
w_yaw = j["w_yaw"]
a_lat = j["a_lat"]
w_accel = j["w_yaw_accel"]
headings = j["headings"]

for blend in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    drifts = []
    for s in range(11, len(x_gps) - 10, 10):
        history_x = list(x_gps[s - 10: s])
        pos_gt = [(0.0, 0.0)]
        pos_pred = [(0.0, 0.0)]
        psi_gt = np.radians(headings[s])
        psi_pred = psi_gt

        turn_sign = -1.0 # default roundabout in this dataset is counter-clockwise / left

        for k in range(10):
            cur = s + k
            ch_a_fwd = np.clip(a_fwd[cur-9:cur+1], -8.0, 8.0)
            ch_w_yaw = np.clip(w_yaw[cur-9:cur+1], -1.0, 1.0)
            ch_a_lat = np.clip(a_lat[cur-9:cur+1], -8.0, 8.0)
            ch_v_prev = np.clip(np.array(history_x[-10:]), 0.0, 45.0)
            ch_w_accel = np.clip(w_accel[cur-9:cur+1], -2.0, 2.0)
            ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)

            win_6 = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
            win_s6 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

            with torch.no_grad():
                d, o, _ = model(torch.tensor(win_s6, dtype=torch.float32, device=device))
            xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
            wr = float(s_yo.inverse_transform([[o.item()]])[0, 0])

            a_mag = np.hypot(a_fwd[cur], a_lat[cur])
            v_est = max(history_x[-1], 2.0)

            if a_mag > 1.2 and v_est > 3.0:
                # Invariant centripetal angular velocity
                w_mag = a_mag / v_est
                # Determine sign from gyro/prediction or previous sign
                if abs(wr) > 0.02:
                    turn_sign = np.sign(wr)
                elif abs(w_yaw[cur]) > 0.02:
                    turn_sign = np.sign(w_yaw[cur])
                w_invar = turn_sign * w_mag
                wr = (1.0 - blend) * wr + blend * w_invar

            history_x.append(xr)
            psi_gt += w_gps[cur]
            psi_pred += wr

            pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
            pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

        drift_10s = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
        drifts.append(drift_10s)

    print(f"Blend={blend:.1f} -> Mean Drift: {np.mean(drifts):.2f}m | Outages: {[round(x, 2) for x in drifts]}")
