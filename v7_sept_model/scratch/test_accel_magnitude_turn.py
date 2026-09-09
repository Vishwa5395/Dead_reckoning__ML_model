import pickle
import numpy as np

with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

j = scens["roundabout"][0]
s = 21

for k in range(10):
    cur = s + k
    af = j["a_fwd"][cur]
    al = j["a_lat"][cur]
    a_mag = np.hypot(af, al)
    v_gt = j["x_gps"][cur]
    w_gt = j["w_gps"][cur]
    w_imu = j["w_yaw"][cur]
    # If a_mag is centripetal: a_c = v * |w| => |w| = a_mag / v
    w_inferred = a_mag / max(v_gt, 1.0)
    print(f"k={k} | v={v_gt:5.2f} | w_gt={w_gt:+6.3f} | a_fwd={af:+5.2f}, a_lat={al:+5.2f} | a_mag={a_mag:5.2f} | w_inferred={w_inferred:5.3f} rad/s ({np.degrees(w_inferred):5.1f}°/s)")
