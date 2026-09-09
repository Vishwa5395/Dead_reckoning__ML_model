import pickle
import numpy as np

with open('v7_sept_model/data/test_scenarios_v7.pkl', 'rb') as f:
    test_scenarios = pickle.load(f)

journeys_st = test_scenarios['sharp_turns']

# Let's inspect Outage 21 (J1 s=101), Outage 22 (J1 s=111), Outage 32 (J2 s=71), Outage 2 (J0 s=21), Outage 15 (J1 s=41)
inspections = [
    ('Outage 2 (J0 s=21)', 0, 21),
    ('Outage 15 (J1 s=41)', 1, 41),
    ('Outage 21 (J1 s=101)', 1, 101),
    ('Outage 22 (J1 s=111)', 1, 111),
    ('Outage 31 (J2 s=61)', 2, 61),
    ('Outage 32 (J2 s=71)', 2, 71),
    ('Outage 36 (J2 s=111)', 2, 111),
    ('Outage 39 (J2 s=141)', 2, 141),
]

print("=" * 80)
print("INSPECTION OF WORST SHARP TURN OUTAGES")
print("=" * 80)

for name, j_idx, s in inspections:
    j = journeys_st[j_idx]
    x_gps = j['x_gps']
    a_fwd = j['a_fwd']
    w_yaw = j['w_yaw']
    a_lat = j['a_lat']
    w_gps = j['w_gps']
    
    gt_turn = np.degrees(np.sum(w_gps[s:s+10]))
    imu_turn = np.degrees(np.sum(w_yaw[s:s+10]))
    entry_speed = x_gps[s-1]
    entry_decel = (x_gps[s-1] - x_gps[s-5]) / 4.0
    gt_speeds = x_gps[s:s+10]
    
    print(f"\n--- {name} ---")
    print(f"Entry Speed = {entry_speed:.2f} m/s | Entry Decel = {entry_decel:.2f} m/s²")
    print(f"GT Turn (sum w_gps) = {gt_turn:+.1f}° | IMU Turn (sum w_yaw) = {imu_turn:+.1f}°")
    print(f"GT Speeds: {np.round(gt_speeds, 2)}")
    print(f"w_yaw:     {np.round(w_yaw[s:s+10], 3)}")
    print(f"w_gps:     {np.round(w_gps[s:s+10], 3)}")
    print(f"a_lat:     {np.round(a_lat[s:s+10], 2)}")
    print(f"a_fwd:     {np.round(a_fwd[s:s+10], 2)}")
