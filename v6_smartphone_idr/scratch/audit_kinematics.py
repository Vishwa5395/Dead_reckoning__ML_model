import pickle
import numpy as np

with open("v6_smartphone_idr/data/val_scenarios_specialists.pkl", "rb") as f:
    val_scenarios = pickle.load(f)

for j in val_scenarios["all"]:
    v = j["v_true"]
    w = j["w_yaw"]
    w_true = j["w_true"]
    a_lat = j["a_lat"]
    mask = (v > 3.0) & (np.abs(w_true) > 0.05)
    if mask.sum() > 20:
        c1 = np.corrcoef(a_lat[mask], (v * w)[mask])[0, 1]
        c2 = np.corrcoef(w[mask], w_true[mask])[0, 1]
        print(f"{j['name']}: corr(a_lat, v*w) = {c1:.3f}, corr(w_yaw, w_true) = {c2:.3f}, n={mask.sum()}")
