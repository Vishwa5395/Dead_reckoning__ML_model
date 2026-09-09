import pickle
import numpy as np

with open("v6_smartphone_idr/data/test_scenarios_v6.pkl", "rb") as f:
    test_scenarios = pickle.load(f)

for scen, journeys in test_scenarios.items():
    if not journeys:
        continue
    j = journeys[0]
    h = j["headings"]
    dh = np.diff(h)
    dh = (dh + 180) % 360 - 180
    w_true = j["w_true"][:-1]
    
    # Check correlation between dh/0.1 and w_true
    corr = np.corrcoef(np.radians(dh) / 0.1, w_true)[0, 1]
    print(f"Scenario {scen}: corr(dh_rate, w_true) = {corr:.4f}")
