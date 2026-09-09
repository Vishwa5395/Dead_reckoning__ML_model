import pickle
import numpy as np

with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

for name, journeys in scens.items():
    del_v = []
    a_f = []
    for j in journeys:
        x_gps = j["x_gps"]
        a_fwd = j["a_fwd"]
        for t in range(1, len(x_gps)):
            dv = x_gps[t] - x_gps[t-1]
            af = a_fwd[t]
            del_v.append(dv)
            a_f.append(af)
    del_v = np.array(del_v)
    a_f = np.array(a_f)
    # Fit dv = gain * af + bias
    A = np.vstack([a_f, np.ones(len(a_f))]).T
    gain, bias = np.linalg.lstsq(A, del_v, rcond=None)[0]
    corr = np.corrcoef(a_f, del_v)[0, 1]
    print(f"Scenario {name:15s}: gain={gain:6.3f}, bias={bias:6.3f}, corr={corr:6.3f}, std(dv)={np.std(del_v):5.2f}")
