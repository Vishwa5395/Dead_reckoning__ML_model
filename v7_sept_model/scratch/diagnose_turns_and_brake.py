import pickle
import numpy as np
import torch
from v7_sept_model.src.models_v7 import StraightSpecialistS1, TurningSpecialistS2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('v7_sept_model/data/scalers_v7.pkl', 'rb') as f:
    scalers = pickle.load(f)
with open('v7_sept_model/data/test_scenarios_v7.pkl', 'rb') as f:
    test_scenarios = pickle.load(f)

s_X = scalers['X']
s_yd = scalers['y_disp']
s_yo = scalers['y_ori']

s1_mot = StraightSpecialistS1().to(device)
s1_mot.load_state_dict(torch.load('v3_pino_dr/checkpoints/best_model.pth', map_location=device, weights_only=False)['model_state_dict'])
s1_mot.eval()

s2_st = TurningSpecialistS2(in_channels=6).to(device)
ckpt_v4 = torch.load('v4_turn_focused/checkpoints/best_model_ablation_D_attn_coupling.pth', map_location=device, weights_only=False)
s2_st.load_state_dict(ckpt_v4['model_state_dict'])
s2_st.eval()

print("=" * 90)
print("AUDIT: SHARP TURNS OUTAGE-BY-OUTAGE (39 Sequences)")
print("=" * 90)

journeys = test_scenarios['sharp_turns']
outage_idx = 0
st_results = []

for j_idx, j in enumerate(journeys):
    x_gps = j['x_gps']
    a_fwd = j['a_fwd']
    w_yaw = j['w_yaw']
    a_lat = j['a_lat']
    w_accel = j['w_yaw_accel']
    w_gps = j['w_gps']
    headings = j['headings']

    for s in range(11, len(x_gps) - 10, 10):
        outage_idx += 1
        history_x = list(x_gps[s - 10: s])
        pos_gt = [(0.0, 0.0)]
        pos_pred = [(0.0, 0.0)]
        psi_gt = np.radians(headings[s])
        psi_pred = psi_gt

        entry_decel = (history_x[-1] - history_x[-5]) / 4.0
        stopping_mode = (entry_decel < -0.5 and history_x[-1] < 7.5)

        gt_speeds = []
        pred_speeds = []
        gt_omegas = []
        pred_omegas = []
        alat_seq = []
        afwd_seq = []
        wyaw_seq = []

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
                if stopping_mode:
                    xr = max(0.0, min(xr, history_x[-1] + entry_decel * 0.80))

            history_x.append(xr)
            psi_gt += w_gps[cur]
            psi_pred += wr

            pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
            pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

            gt_speeds.append(x_gps[cur])
            pred_speeds.append(xr)
            gt_omegas.append(w_gps[cur])
            pred_omegas.append(wr)
            alat_seq.append(a_lat[cur])
            afwd_seq.append(a_fwd[cur])
            wyaw_seq.append(w_yaw[cur])

        drift_10s = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
        st_results.append({
            'outage_idx': outage_idx,
            'j_idx': j_idx,
            's': s,
            'drift_10s': drift_10s,
            'gt_speeds': gt_speeds,
            'pred_speeds': pred_speeds,
            'gt_omegas': gt_omegas,
            'pred_omegas': pred_omegas,
            'alat_seq': alat_seq,
            'afwd_seq': afwd_seq,
            'wyaw_seq': wyaw_seq,
            'psi_gt_net': psi_gt - np.radians(headings[s]),
            'psi_pred_net': psi_pred - np.radians(headings[s]),
            'gt_dist': np.sum(gt_speeds),
            'pred_dist': np.sum(pred_speeds),
        })

st_sorted = sorted(st_results, key=lambda x: x['drift_10s'], reverse=True)
print(f"Top 10 Worst Outages in Sharp Turns:")
for r in st_sorted[:10]:
    yaw_err_deg = np.degrees(abs(r['psi_pred_net'] - r['psi_gt_net']))
    speed_err = r['pred_dist'] - r['gt_dist']
    print(f"Outage #{r['outage_idx']:2d} (J{r['j_idx']} s={r['s']:3d}): Drift = {r['drift_10s']:6.2f}m | "
          f"GT dist={r['gt_dist']:5.1f}m Pred dist={r['pred_dist']:5.1f}m (err={speed_err:+5.1f}m) | "
          f"GT turn={np.degrees(r['psi_gt_net']):+6.1f}° Pred turn={np.degrees(r['psi_pred_net']):+6.1f}° (err={yaw_err_deg:5.1f}°)")

print("\n" + "=" * 90)
print("AUDIT: HARD BRAKE OUTAGE-BY-OUTAGE (12 Sequences)")
print("=" * 90)

journeys_hb = test_scenarios['hard_brake']
hb_results = []
outage_idx = 0
for j_idx, j in enumerate(journeys_hb):
    x_gps = j['x_gps']
    a_fwd = j['a_fwd']
    w_yaw = j['w_yaw']
    a_lat = j['a_lat']
    w_accel = j['w_yaw_accel']
    w_gps = j['w_gps']
    headings = j['headings']

    for s in range(11, len(x_gps) - 10, 10):
        outage_idx += 1
        history_x = list(x_gps[s - 10: s])
        pos_gt = [(0.0, 0.0)]
        pos_pred = [(0.0, 0.0)]
        psi_gt = np.radians(headings[s])
        psi_pred = psi_gt

        entry_decel = (history_x[-1] - history_x[-5]) / 4.0

        gt_speeds = []
        pred_speeds = []
        gt_omegas = []
        pred_omegas = []

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
                d, o, _ = s1_mot(torch.tensor(win_s6[:, :, :4], dtype=torch.float32, device=device))
                xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
                wr = float(s_yo.inverse_transform([[o.item()]])[0, 0]) * 0.05
                if a_fwd[cur] < -0.8:
                    xr = min(xr, max(0.0, history_x[-1] + a_fwd[cur] * 0.20))
                if entry_decel < -0.6 and history_x[-1] < 15.0:
                    xr = max(0.0, min(xr, history_x[-1] + entry_decel * 1.00))
                if xr < 0.2 and a_fwd[cur] < 0.2:
                    xr, wr = 0.0, 0.0

            history_x.append(xr)
            psi_gt += w_gps[cur]
            psi_pred += wr

            pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
            pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

            gt_speeds.append(x_gps[cur])
            pred_speeds.append(xr)
            gt_omegas.append(w_gps[cur])
            pred_omegas.append(wr)

        drift_10s = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
        hb_results.append({
            'outage_idx': outage_idx,
            'j_idx': j_idx,
            's': s,
            'drift_10s': drift_10s,
            'gt_speeds': gt_speeds,
            'pred_speeds': pred_speeds,
            'gt_omegas': gt_omegas,
            'pred_omegas': pred_omegas,
            'psi_gt_net': psi_gt - np.radians(headings[s]),
            'psi_pred_net': psi_pred - np.radians(headings[s]),
            'gt_dist': np.sum(gt_speeds),
            'pred_dist': np.sum(pred_speeds),
        })

hb_sorted = sorted(hb_results, key=lambda x: x['drift_10s'], reverse=True)
print(f"Top Worst Outages in Hard Brake (Total 12):")
for r in hb_sorted:
    yaw_err_deg = np.degrees(abs(r['psi_pred_net'] - r['psi_gt_net']))
    speed_err = r['pred_dist'] - r['gt_dist']
    print(f"Outage #{r['outage_idx']:2d} (J{r['j_idx']} s={r['s']:3d}): Drift = {r['drift_10s']:6.2f}m | "
          f"GT dist={r['gt_dist']:5.1f}m Pred dist={r['pred_dist']:5.1f}m (err={speed_err:+5.1f}m) | "
          f"GT turn={np.degrees(r['psi_gt_net']):+6.1f}° Pred turn={np.degrees(r['psi_pred_net']):+6.1f}° (err={yaw_err_deg:5.1f}°)")
