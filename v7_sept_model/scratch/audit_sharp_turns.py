import pickle
import numpy as np
import torch
from v7_sept_model.src.models_v7 import TurningSpecialistS2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('v7_sept_model/data/scalers_v7.pkl', 'rb') as f:
    scalers = pickle.load(f)
with open('v7_sept_model/data/test_scenarios_v7.pkl', 'rb') as f:
    scens = pickle.load(f)

s_X = scalers['X']
s_yd = scalers['y_disp']
s_yo = scalers['y_ori']
s2_st = TurningSpecialistS2(in_channels=6).to(device)
ckpt_v4 = torch.load('v4_turn_focused/checkpoints/best_model_ablation_D_attn_coupling.pth', map_location=device, weights_only=False)
s2_st.load_state_dict(ckpt_v4['model_state_dict'])
s2_st.eval()

results = []
out_cnt = 0

for j_idx, j in enumerate(scens['sharp_turns']):
    x_gps = j['x_gps']
    w_gps = j['w_gps']
    a_fwd = j['a_fwd']
    w_yaw = j['w_yaw']
    a_lat = j['a_lat']
    w_accel = j['w_yaw_accel']
    headings = j['headings']

    for s in range(11, len(x_gps) - 10, 10):
        out_cnt += 1
        history_x = list(x_gps[s - 10: s])
        pos_gt = [(0.0, 0.0)]
        pos_pred = [(0.0, 0.0)]
        psi_gt = np.radians(headings[s])
        psi_pred = psi_gt

        turn_pred_steps = []
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
                d, o, _ = s2_st(torch.tensor(win_s6, dtype=torch.float32, device=device))
            xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
            wr = float(s_yo.inverse_transform([[o.item()]])[0, 0]) * 0.05
            v_est = max(history_x[-1], 2.5)
            if abs(a_lat[cur]) > 2.4:
                w_cent = -np.sign(a_lat[cur]) * (abs(a_lat[cur]) / v_est) * 1.00
                wr += w_cent
                xr = min(xr, max(1.5, history_x[-1] - 0.40))

            turn_pred_steps.append(wr)
            history_x.append(xr)
            psi_gt += w_gps[cur]
            psi_pred += wr

            pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
            pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

        drift = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
        net_turn_gt = np.degrees(sum(w_gps[s:s+10]))
        net_turn_pred = np.degrees(sum(turn_pred_steps))
        head_err = np.degrees(abs(psi_pred - psi_gt))
        max_alat = float(np.max(np.abs(a_lat[s:s+10])))
        max_wyaw = float(np.max(np.abs(w_yaw[s:s+10])))
        dist_gt = float(sum(x_gps[s:s+10]))
        dist_pred = float(sum(history_x[10:]))

        results.append({
            'out_id': out_cnt,
            'j_idx': j_idx,
            's': s,
            'drift': drift,
            'net_turn_gt': net_turn_gt,
            'net_turn_pred': net_turn_pred,
            'head_err': head_err,
            'dist_diff': dist_pred - dist_gt,
            'max_alat': max_alat,
            'max_wyaw': max_wyaw,
        })

results.sort(key=lambda r: r['drift'], reverse=True)
print("=== TOP 15 WORST OUTAGES IN SHARP TURNS (Current Mean: 32.95m) ===")
for r in results[:15]:
    print(f"#{r['out_id']:02d} (J{r['j_idx']}, s={r['s']:3d}) | Drift={r['drift']:5.2f}m | TurnGT={r['net_turn_gt']:6.1f}°, TurnPred={r['net_turn_pred']:6.1f}°, HeadErr={r['head_err']:5.1f}° | DistDiff={r['dist_diff']:5.1f}m | max|alat|={r['max_alat']:4.1f}, max|w|={r['max_wyaw']:4.2f}")

print("\n=== TOP 10 BEST OUTAGES IN SHARP TURNS ===")
for r in results[-10:]:
    print(f"#{r['out_id']:02d} (J{r['j_idx']}, s={r['s']:3d}) | Drift={r['drift']:5.2f}m | TurnGT={r['net_turn_gt']:6.1f}°, TurnPred={r['net_turn_pred']:6.1f}°, HeadErr={r['head_err']:5.1f}° | DistDiff={r['dist_diff']:5.1f}m")
