import pickle
import numpy as np

with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

journeys = scens["sharp_turns"]
w_g_list = []
w_y_list = []
a_l_list = []

for j in journeys:
    x_gps = j["x_gps"]
    w_gps = j["w_gps"]
    w_yaw = j["w_yaw"]
    a_lat = j["a_lat"]
    for s in range(11, len(x_gps) - 10, 10):
        for k in range(10):
            cur = s + k
            w_g_list.append(w_gps[cur])
            w_y_list.append(w_yaw[cur])
            a_l_list.append(a_lat[cur])

w_g = np.array(w_g_list)
w_y = np.array(w_y_list)
a_l = np.array(a_l_list)

corr_gyro = np.corrcoef(w_y, w_g)[0, 1]
corr_alat = np.corrcoef(a_l, w_g)[0, 1]

print(f"Sharp Turns Correlations with Ground Truth Heading Change (w_gps):")
print(f"  Corr(w_yaw, w_gps) = {corr_gyro:.4f}")
print(f"  Corr(a_lat, w_gps) = {corr_alat:.4f}")
