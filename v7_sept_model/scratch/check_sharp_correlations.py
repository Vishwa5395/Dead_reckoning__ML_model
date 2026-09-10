import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

d = np.load(DATA_DIR / "data_sharp_turns.npz")
X_tr = d["X_tr"] # (N, 10, 6)
yo_tr = d["yo_tr"] # (N, 1) in normalized space [0, 1]

# Invert yo_tr to rad/step
w_step = yo_tr * 2.0263271 - 1.0224137

# Let's inspect rows where |w_step| > 0.1 rad/step (~ 5.7 deg/step, i.e. 57 deg/s)
sharp_idx = np.where(np.abs(w_step) > 0.1)[0]
print(f"Total training windows with |w_step| > 0.1: {len(sharp_idx)}")

# Check correlation between w_step and channels in X_tr
# Channels in normalized space: 0: a_fwd, 1: w_yaw, 2: a_lat, 3: v_prev, 4: w_accel, 5: centripetal
# Invert X_tr to physical:
with open(DATA_DIR / "scalers_v7.pkl", "rb") as f:
    scalers = pickle.load(f)
s_X = scalers["X"]

X_phys = s_X.inverse_transform(X_tr[sharp_idx].reshape(len(sharp_idx), -1)).reshape(len(sharp_idx), 10, 6)
w_sharp = w_step[sharp_idx]

alat_last = X_phys[:, -1, 2]
wyaw_last = X_phys[:, -1, 1]
afwd_last = X_phys[:, -1, 0]
v_last = X_phys[:, -1, 3]

print("Correlation with w_sharp:")
print("wyaw_last correlation:", np.corrcoef(wyaw_last, w_sharp[:, 0])[0, 1])
print("alat_last correlation:", np.corrcoef(alat_last, w_sharp[:, 0])[0, 1])
print("alat_last / v correlation:", np.corrcoef(alat_last / np.maximum(v_last, 1.0), w_sharp[:, 0])[0, 1])
print("wyaw mean / std on sharp:", np.mean(wyaw_last), np.std(wyaw_last))
print("w_sharp mean / std on sharp:", np.mean(w_sharp), np.std(w_sharp))
