import pickle
import numpy as np

with open('v7_sept_model/data/test_scenarios_v7.pkl', 'rb') as f: scens = pickle.load(f)

journeys = scens['sharp_turns']

for j_idx, j in enumerate(journeys):
    x_gps = j['x_gps']
    w_gps = j['w_gps']
    a_lat = j['a_lat']
    w_yaw = j['w_yaw']

    # For the whole journey:
    cent_ideal = - w_gps * x_gps
    corr_lat = np.corrcoef(a_lat, cent_ideal)[0, 1]
    corr_yaw = np.corrcoef(w_yaw, w_gps)[0, 1]
    print(f"Journey {j_idx}: len={len(x_gps)} | Corr(a_lat, -w_gps*v) = {corr_lat:+.3f} | Corr(w_yaw, w_gps) = {corr_yaw:+.3f}")
