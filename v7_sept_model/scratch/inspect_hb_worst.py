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
    a_lat = j["a_lat"]
    headings = j["headings"]
    for s in range(11, len(x_gps) - 10, 10):
        outage_cnt += 1
        if outage_cnt in [4, 9, 10]:
            print(f"\n=================== OUTAGE {outage_cnt} (J{j_idx}, s={s}) ===================")
            for k in range(10):
                cur = s + k
                print(f"k={k} | v_gps={x_gps[cur]:5.2f}, afwd={a_fwd[cur]:+5.2f}, alat={a_lat[cur]:+5.2f}, w_gps={w_gps[cur]:+6.3f}, w_imu={w_yaw[cur]:+6.3f}")
