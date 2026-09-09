import pickle
import numpy as np

with open('v7_sept_model/data/test_scenarios_v7.pkl', 'rb') as f:
    scens = pickle.load(f)

journeys = scens['sharp_turns']
outage_idx = 0

print(f"{'Outage':7s} | {'J':2s} | {'s':3s} | {'GT turn':8s} | {'w_yaw turn':11s} | {'alat turn':10s} | {'Mean Spd':8s} | {'Max |alat|':10s} | {'Max |afwd|':10s}")
print("-" * 85)

for j_idx, j in enumerate(journeys):
    x_gps = j['x_gps']
    w_gps = j['w_gps']
    a_fwd = j['a_fwd']
    w_yaw = j['w_yaw']
    a_lat = j['a_lat']
    headings = j['headings']

    for s in range(11, len(x_gps) - 10, 10):
        outage_idx += 1
        gt_turn = np.degrees(np.sum(w_gps[s:s+10]))
        wyaw_turn = np.degrees(np.sum(w_yaw[s:s+10]))
        v_mean = np.mean(x_gps[s:s+10])
        max_alat = np.max(np.abs(a_lat[s:s+10]))
        max_afwd = np.max(np.abs(a_fwd[s:s+10]))
        
        # Centripetal yaw estimate: w ~ -a_lat / v
        v_safe = np.maximum(x_gps[s:s+10], 2.0)
        alat_turn = np.degrees(np.sum(-a_lat[s:s+10] / v_safe))
        
        flag = ""
        if abs(gt_turn) > 30.0:
            flag = " <-- BIG TURN"
            
        print(f"#{outage_idx:2d}     | J{j_idx} | {s:3d} | {gt_turn:+7.1f}° | {wyaw_turn:+10.1f}° | {alat_turn:+9.1f}° | {v_mean:6.2f}m/s | {max_alat:8.2f} | {max_afwd:8.2f} {flag}")
