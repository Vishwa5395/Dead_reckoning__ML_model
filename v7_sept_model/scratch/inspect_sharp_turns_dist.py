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

s2 = TurningSpecialistS2(in_channels=6).to(device)
ckpt_v4 = torch.load("v4_turn_focused/checkpoints/best_model_ablation_D_attn_coupling.pth", map_location=device)
s2.load_state_dict(ckpt_v4["model_state_dict"])
s2.eval()

journeys = scens["sharp_turns"]
all_drifts = []
outage_cnt = 0

for j_idx, j in enumerate(journeys):
    x_gps = j["x_gps"]
    w_gps = j["w_gps"]
    a_fwd = j["a_fwd"]
    w_yaw = j["w_yaw"]
    a_lat = j["a_lat"]
    w_accel = j["w_yaw_accel"]
    headings = j["headings"]

    for s in range(11, len(x_gps) - 10, 10):
        outage_cnt += 1
        history_x = list(x_gps[s - 10: s])
        pos_gt = [(0.0, 0.0)]
        pos_pred = [(0.0, 0.0)]
        psi_gt = np.radians(headings[s])
        psi_pred = psi_gt

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
                d, o, _ = s2(torch.tensor(win_s6, dtype=torch.float32, device=device))
            xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
            wr = float(s_yo.inverse_transform([[o.item()]])[0, 0])

            history_x.append(xr)
            psi_gt += w_gps[cur]
            psi_pred += wr

            pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
            pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

        drift_10s = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
        head_err_deg = np.degrees(abs(psi_pred - psi_gt))
        net_turn_deg = np.degrees(abs(np.sum(w_gps[s:s+10])))
        all_drifts.append(drift_10s)
        if drift_10s > 40.0:
            print(f"High Drift Outage #{outage_cnt:02d} (J{j_idx}): Drift={drift_10s:5.2f}m | HeadErr={head_err_deg:5.2f}° | NetTurn={net_turn_deg:5.2f}°")

print(f"\nSharp Turns Summary across 39 Outages:")
print(f"  Mean Drift:   {np.mean(all_drifts):.2f}m")
print(f"  Median Drift: {np.median(all_drifts):.2f}m")
print(f"  Min Drift:    {np.min(all_drifts):.2f}m")
print(f"  Max Drift:    {np.max(all_drifts):.2f}m")
print(f"  Outages <= 17.71m (Met 2X Target): {sum(1 for d in all_drifts if d <= 17.71)} / {len(all_drifts)} ({sum(1 for d in all_drifts if d <= 17.71)/len(all_drifts)*100:.1f}%)")
print(f"  Outages <= 25.0m: {sum(1 for d in all_drifts if d <= 25.0)} / {len(all_drifts)}")
