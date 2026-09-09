import pickle
import numpy as np

with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

journeys = scens["sharp_turns"]
outage_cnt = 0
for j_idx, j in enumerate(journeys):
    x_gps = j["x_gps"]
    w_gps = j["w_gps"]
    w_yaw = j["w_yaw"]
    a_lat = j["a_lat"]
    for s in range(11, len(x_gps) - 10, 10):
        outage_cnt += 1
        net_w_gps = np.sum(w_gps[s:s+10])
        net_w_imu = np.sum(w_yaw[s:s+10])
        net_alat = np.sum(a_lat[s:s+10])
        if abs(net_w_gps) > 0.8: # strong turn (> 45 deg)
            ratio = net_w_imu / net_w_gps
            print(f"Outage #{outage_cnt:02d}: net_w_gps={np.degrees(net_w_gps):+6.1f}°, net_w_imu={np.degrees(net_w_imu):+6.1f}°, ratio={ratio:5.2f}, net_alat={net_alat:+5.2f}")
