import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
    sc = pickle.load(f)

for scen in ["roundabout", "sharp_turns"]:
    print(f"\n=== Scenario {scen} ===")
    for j in sc[scen]:
        w = j["w_gps"]
        alat = j["a_lat"]
        wyaw = j["w_yaw"]
        v = j["x_gps"]
        
        # Look at steps where |w| > 0.1 rad (~ 5.7 deg)
        idx = np.where(np.abs(w) > 0.1)[0]
        if len(idx) > 0:
            print(f"Journey with {len(idx)} high-turn steps:")
            for i in idx[:10]:
                print(f"  step {i:3d}: w_gps={w[i]:+6.3f} rad ({w[i]*180/np.pi:+5.1f} deg), "
                      f"w_yaw={wyaw[i]:+6.3f} rad/s, a_lat={alat[i]:+6.2f} m/s^2, v={v[i]:4.1f} m/s, "
                      f"-alat/v={-alat[i]/max(v[i],1.0):+6.3f}")
