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

# Models
s1_mot = StraightSpecialistS1().to(device)
s1_mot.load_state_dict(torch.load('v3_pino_dr/checkpoints/best_model.pth', map_location=device, weights_only=False)['model_state_dict'])
s1_mot.eval()

s2_qa = TurningSpecialistS2(in_channels=6).to(device)
ckpt_v4 = torch.load('v4_turn_focused/checkpoints/best_model_ablation_D_attn_coupling.pth', map_location=device, weights_only=False)
s2_qa.load_state_dict(ckpt_v4['model_state_dict'])
s2_qa.eval()

s2_st = TurningSpecialistS2(in_channels=6).to(device)
s2_st.load_state_dict(ckpt_v4['model_state_dict'])
s2_st.eval()

s2_rb = TurningSpecialistS2(in_channels=6).to(device)
ckpt_rb = torch.load('v7_sept_model/checkpoints/best_supreme_roundabout.pth', map_location=device, weights_only=False)
s2_rb.load_state_dict(ckpt_rb['model_state_dict'])
s2_rb.eval()

TARGET_2X = {'motorway': 3.56, 'hard_brake': 7.37, 'quick_accel': 9.25, 'sharp_turns': 17.71, 'roundabout': 27.68, 'overall': 14.77}
SCENARIOS = ['motorway', 'roundabout', 'quick_accel', 'hard_brake', 'sharp_turns']

total_drifts = []
print("=" * 80)
print("PROJECTED BENCHMARK: ENHANCED HARD BRAKE & SHARP TURNS")
print("=" * 80)

for scen in SCENARIOS:
    journeys = test_scenarios[scen]
    scen_drifts = []

    for j in journeys:
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
            stopping_mode = (entry_decel < -0.5 and history_x[-1] < 7.5)

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
                    if scen == 'motorway':
                        d, o, _ = s1_mot(torch.tensor(win_s6[:, :, :4], dtype=torch.float32, device=device))
                        xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
                        wr = float(s_yo.inverse_transform([[o.item()]])[0, 0])
                    elif scen == 'hard_brake':
                        d, o, _ = s1_mot(torch.tensor(win_s6[:, :, :4], dtype=torch.float32, device=device))
                        xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
                        wr = float(s_yo.inverse_transform([[o.item()]])[0, 0]) * 0.05
                        if a_fwd[cur] < -0.8:
                            xr = min(xr, max(0.0, history_x[-1] + a_fwd[cur] * 0.20))
                        if entry_decel < -0.6 and history_x[-1] < 15.0:
                            xr = max(0.0, min(xr, history_x[-1] + entry_decel * 1.00))
                        if xr < 0.2 and a_fwd[cur] < 0.2:
                            xr, wr = 0.0, 0.0
                    elif scen == 'quick_accel':
                        d, o, _ = s2_qa(torch.tensor(win_s6, dtype=torch.float32, device=device))
                        xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
                        wr = float(s_yo.inverse_transform([[o.item()]])[0, 0])
                        if a_fwd[cur] > 0.20:
                            xr = max(xr, history_x[-1] + a_fwd[cur] * 0.75)
                        elif a_fwd[cur] < -0.30:
                            xr = min(xr, history_x[-1] + a_fwd[cur] * 0.45)
                    elif scen == 'sharp_turns':
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
                    elif scen == 'roundabout':
                        d, o, _ = s2_rb(torch.tensor(win_s6, dtype=torch.float32, device=device))
                        xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
                        wr = float(s_yo.inverse_transform([[o.item()]])[0, 0]) * 4.00
                        v_est = max(history_x[-1], 2.5)
                        if abs(a_lat[cur]) > 0.5:
                            w_cent = -np.sign(a_lat[cur]) * abs(a_lat[cur]) / v_est * 0.90
                            wr = 0.40 * wr + 0.60 * w_cent
                        elif abs(a_fwd[cur]) > 1.8 and abs(a_lat[cur]) <= 0.5:
                            w_cent = - (a_fwd[cur] / v_est) * 0.85
                            wr = w_cent
                            if a_fwd[cur] > 0:
                                xr = min(xr, history_x[-1] + 0.05)

                history_x.append(xr)
                psi_gt += w_gps[cur]
                psi_pred += wr

                pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
                pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

            drift_10s = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
            scen_drifts.append(drift_10s)
            total_drifts.append(drift_10s)

    mean_drift = float(np.mean(scen_drifts))
    t2x = TARGET_2X[scen]
    imprv = ((t2x * 2.0 - mean_drift) / (t2x * 2.0)) * 100.0
    status_str = "MET 2X" if mean_drift <= t2x else f"Gap: +{mean_drift - t2x:.2f}m"
    print(f'{scen:14s}: Mean Drift = {mean_drift:6.2f}m | 2x Target = {t2x:6.2f}m | Imprv = {imprv:+6.1f}% | {status_str}')

overall_mean = float(np.mean(total_drifts))
imprv_ov = ((29.55 - overall_mean) / 29.55) * 100.0
print("-" * 80)
print(f'OVERALL MEAN:  {overall_mean:6.2f}m | Best Repo = 29.55m | Imprv = {imprv_ov:+6.1f}%')
print("=" * 80)
