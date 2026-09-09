import pickle
import numpy as np

with open('v7_sept_model/data/test_scenarios_v7.pkl', 'rb') as f: scens = pickle.load(f)

journeys = scens['sharp_turns']
idx = 0

print(f"{'#':2s} | {'J':2s} | {'s':3s} | {'GT sum':8s} | {'w_yaw sum':10s} | {'alat sum':10s} | {'Sign(GT)==Sign(w_yaw)':22s} | {'Sign(GT)==Sign(-alat)'}")
print("-" * 85)

for j_idx, j in enumerate(journeys):
    x_gps = j['x_gps']
    w_gps = j['w_gps']
    w_yaw = j['w_yaw']
    a_lat = j['a_lat']

    for s in range(11, len(x_gps) - 10, 10):
        idx += 1
        gt_turn = np.sum(w_gps[s:s+10])
        wyaw_turn = np.sum(w_yaw[s:s+10])
        alat_turn = np.sum(-a_lat[s:s+10])

        if abs(np.degrees(gt_turn)) > 30.0:
            match_yaw = np.sign(gt_turn) == np.sign(wyaw_turn)
            match_lat = np.sign(gt_turn) == np.sign(alat_turn)
            print(f"#{idx:2d} | J{j_idx} | {s:3d} | {np.degrees(gt_turn):+7.1f}° | {np.degrees(wyaw_turn):+9.1f}° | {alat_turn:+9.2f} | {str(match_yaw):22s} | {str(match_lat)}")
