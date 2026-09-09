import pickle
import numpy as np

with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

j = scens["quick_accel"][0]
s = 21
for k in range(10):
    cur = s + k
    print(f"k={k} | w_yaw_imu={j['w_yaw'][cur]:+7.4f} | w_gps={j['w_gps'][cur]:+7.4f} | ratio={j['w_gps'][cur] / max(abs(j['w_yaw'][cur]), 1e-4):5.2f}")
