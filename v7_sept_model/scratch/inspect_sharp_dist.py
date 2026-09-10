import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "data" / "data_sharp_turns.npz")
yo = d["yo_tr"] * 2.0263271 - 1.0224137 # rad/step
yd = d["yd_tr"] * 45.0 # m/step

print(f"Total training windows: {len(yo)}")
print(f"Mean yo: {np.mean(yo):.4f} rad, std: {np.std(yo):.4f} rad")
print(f"Min yo: {np.min(yo):.4f}, Max yo: {np.max(yo):.4f}")

# Convert rad/step to deg/step
deg_step = yo * 180 / np.pi
print(f"Percent with |deg_step| < 1.0 deg: {np.mean(np.abs(deg_step) < 1.0)*100:.2f}%")
print(f"Percent with |deg_step| < 3.0 deg: {np.mean(np.abs(deg_step) < 3.0)*100:.2f}%")
print(f"Percent with |deg_step| >= 5.0 deg: {np.mean(np.abs(deg_step) >= 5.0)*100:.2f}%")
print(f"Percent with |deg_step| >= 10.0 deg: {np.mean(np.abs(deg_step) >= 10.0)*100:.2f}%")

# Now check the input windows X_tr
X = d["X_tr"] # (28140, 10, 6)
# Channel 1: w_yaw, Channel 2: a_lat
import pickle
with open(ROOT / "data" / "scalers_v7.pkl", "rb") as f:
    s_X = pickle.load(f)["X"]
X_p = s_X.inverse_transform(X[:1000].reshape(1000, -1)).reshape(1000, 10, 6)
print("X_p sample shape:", X_p.shape)
