import pickle
import numpy as np

with open("v6_smartphone_idr/data/test_scenarios_v6.pkl", "rb") as f:
    test_scenarios = pickle.load(f)

sharp = test_scenarios["sharp_turns"]
j2 = sharp[2]
w_yaw = j2["w_yaw"][130:230]
w_true = j2["w_true"][130:230]

print("w_yaw[:10]:", w_yaw[:10])
print("w_true[:10]:", w_true[:10])
print("Correlation in slice:", np.corrcoef(w_yaw, w_true)[0, 1])
print("Correlation in full journey:", np.corrcoef(j2["w_yaw"], j2["w_true"])[0, 1])
