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

for j_idx, s, name in [(1, 41, "Outage 15"), (2, 51, "Outage 30"), (2, 111, "Outage 36")]:
    j = journeys[j_idx]
    x_gps = j['x_gps']
    a_fwd = j['a_fwd']
    w_yaw = j['w_yaw']
    a_lat = j['a_lat']
    w_accel = j['w_yaw_accel']

    history_x = list(x_gps[s - 10: s])
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
        win_s6 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

        with torch.no_grad():
            d_p, _, _ = s2_st(torch.tensor(win_s6, dtype=torch.float32, device=device))
        xr = float(s_yd.inverse_transform([[d_p.item()]])[0, 0])
        raw_preds.append(xr)
        history_x.append(xr)

    print(f"\n{name} (J{j_idx} s={s}):")
    print(f"  GT:       {np.round(x_gps[s:s+10], 2)}")
    print(f"  Raw Pred: {np.round(raw_preds, 2)}")
