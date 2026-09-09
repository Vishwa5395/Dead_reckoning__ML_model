import pickle

with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

j = scens["roundabout"][0]
s = 21
for k in range(10):
    cur = s + k
    print(f"k={k} | w_gt={j['w_gps'][cur]:+7.4f} | w_imu={j['w_yaw'][cur]:+7.4f} | a_lat={j['a_lat'][cur]:+7.4f} | a_fwd={j['a_fwd'][cur]:+7.4f}")
