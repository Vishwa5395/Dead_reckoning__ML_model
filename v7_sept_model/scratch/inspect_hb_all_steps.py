import pickle
import numpy as np
import torch
from v7_sept_model.src.models_v7 import StraightSpecialistS1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('v7_sept_model/data/scalers_v7.pkl', 'rb') as f: scalers = pickle.load(f)
with open('v7_sept_model/data/test_scenarios_v7.pkl', 'rb') as f: scens = pickle.load(f)

s_X = scalers['X']; s_yd = scalers['y_disp']

s1 = StraightSpecialistS1().to(device)
s1.load_state_dict(torch.load('v3_pino_dr/checkpoints/best_model.pth', map_location=device)['model_state_dict'])
s1.eval()

journeys = scens['hard_brake']
outage_idx = 0

for j_idx, j in enumerate(journeys):
    x_gps = j['x_gps']
    a_fwd = j['a_fwd']
    w_yaw = j['w_yaw']
    a_lat = j['a_lat']
    w_accel = j['w_yaw_accel']

    for s in range(11, len(x_gps) - 10, 10):
        outage_idx += 1
        history_x = list(x_gps[s - 10: s])
        entry_speed = history_x[-1]
        entry_decel = (history_x[-1] - history_x[-5]) / 4.0

        raw_preds = []
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
                d, _, _ = s1(torch.tensor(win_s4, dtype=torch.float32, device=device))
                xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
            raw_preds.append(xr)
            history_x.append(xr)

        gt = x_gps[s:s+10]
        afwd_sub = a_fwd[s:s+10]
        print(f"\nHB Outage #{outage_idx} (J{j_idx} s={s}): Entry v={entry_speed:.2f}, decel={entry_decel:.2f}")
        print(f"  GT v:      {np.round(gt, 2)}")
        print(f"  Raw Pred:  {np.round(raw_preds, 2)}")
        print(f"  Diff (P-G):{np.round(np.array(raw_preds) - gt, 2)}")
        print(f"  a_fwd:     {np.round(afwd_sub, 2)}")
