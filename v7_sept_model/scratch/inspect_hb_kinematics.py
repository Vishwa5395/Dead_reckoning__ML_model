import pickle
import numpy as np

with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

journeys = scens["hard_brake"]
outage_cnt = 0
for j_idx, j in enumerate(journeys):
    x_gps = j["x_gps"]
    w_gps = j["w_gps"]
    a_fwd = j["a_fwd"]
    w_yaw = j["w_yaw"]
    for s in range(11, len(x_gps) - 10, 10):
        outage_cnt += 1
        v_start = x_gps[s]
        v_end = x_gps[s+9]
        min_v = np.min(x_gps[s:s+10])
        max_v = np.max(x_gps[s:s+10])
        min_afwd = np.min(a_fwd[s:s+10])
        max_yaw = np.max(np.abs(w_gps[s:s+10]))
        print(f"Outage {outage_cnt:02d} (J{j_idx}): v_start={v_start:5.2f}, v_end={v_end:5.2f}, min_v={min_v:5.2f}, max_v={max_v:5.2f}, min_afwd={min_afwd:5.2f}, max_yaw_deg={np.degrees(max_yaw):5.2f}°")
