import pickle
import numpy as np

with open('v7_sept_model/data/test_scenarios_v7.pkl', 'rb') as f: scens = pickle.load(f)

journeys = scens['hard_brake']
j0 = journeys[0]

for s, num in [(91, 9), (101, 10)]:
    print(f"\n--- HB Outage #{num} (s={s}) ---")
    print(f"GT speeds: {np.round(j0['x_gps'][s:s+10], 2)}")
    print(f"a_fwd:     {np.round(j0['a_fwd'][s:s+10], 2)}")
    print(f"a_lat:     {np.round(j0['a_lat'][s:s+10], 2)}")
    print(f"w_yaw:     {np.round(j0['w_yaw'][s:s+10], 3)}")
    print(f"w_gps:     {np.round(j0['w_gps'][s:s+10], 3)}")
