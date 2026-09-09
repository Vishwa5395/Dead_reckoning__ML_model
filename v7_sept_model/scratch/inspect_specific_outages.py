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

# Let's inspect outage 15 (J1 s=41) and outage 30 (J2 s=51) and outage 36 (J2 s=111)
target_outages = [(1, 41, 15), (2, 51, 30), (2, 111, 36)]

for j_idx, s, out_num in target_outages:
    j = journeys[j_idx]
    x_gps = j['x_gps']
    a_fwd = j['a_fwd']
    w_yaw = j['w_yaw']
    a_lat = j['a_lat']
    w_accel = j['w_yaw_accel']
    w_gps = j['w_gps']
    headings = j['headings']

    history_x = list(x_gps[s - 10: s])
    entry_decel = (history_x[-1] - history_x[-5]) / 4.0
    print(f"\nOutage #{out_num} (J{j_idx} s={s}): Entry v={history_x[-1]:.2f}, entry_decel={entry_decel:.2f}")
    print(f"  GT v:  {np.round(x_gps[s:s+10], 2)}")
    print(f"  a_fwd: {np.round(a_fwd[s:s+10], 2)}")
    print(f"  a_lat: {np.round(a_lat[s:s+10], 2)}")
    print(f"  w_yaw: {np.round(w_yaw[s:s+10], 3)}")
    print(f"  w_gps: {np.round(w_gps[s:s+10], 3)}")
