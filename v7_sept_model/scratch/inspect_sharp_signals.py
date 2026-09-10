import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
    test_scenarios = pickle.load(f)

sharp_journeys = test_scenarios["sharp_turns"]

# Let's inspect outage 21 and outage 39
# Find which journey and index they correspond to
WINDOW = 10
OUTAGE = 10

outage_id = 0
for j_idx, j in enumerate(sharp_journeys):
    x_gps = j["x_gps"]; w_gps = j["w_gps"]; headings = j["headings"]
    a_fwd = j["a_fwd"]; w_yaw = j["w_yaw"]; a_lat = j["a_lat"]; w_accel = j["w_yaw_accel"]

    for s in range(WINDOW + 1, len(x_gps) - OUTAGE, 10):
        outage_id += 1
        if outage_id in [2, 13, 15, 21, 32, 39]:
            print(f"\n--- OUTAGE {outage_id} (Journey {j_idx}, start {s}) ---")
            gt_turn = sum(w_gps[s:s+10]) * 180 / np.pi
            gyro_sum = sum(w_yaw[s:s+10]) * 180 / np.pi
            print(f"Total GT Turn: {gt_turn:.2f} deg | Raw Gyro Turn: {gyro_sum:.2f} deg")
            print(f"w_yaw (rad/s): {w_yaw[s:s+10]}")
            print(f"w_gps (rad/step): {w_gps[s:s+10]}")
            print(f"a_lat (m/s^2): {a_lat[s:s+10]}")
            print(f"a_fwd (m/s^2): {a_fwd[s:s+10]}")
            print(f"speed (m/s): {x_gps[s:s+10]}")
