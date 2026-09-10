import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from v7_sept_model.src.moe_five_model import TurningExpertNetwork

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
WS_ROOT = ROOT.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = np.load(DATA_DIR / "data_sharp_turns.npz")
X_va = torch.tensor(d["X_va"], dtype=torch.float32, device=device)
yo_va = torch.tensor(d["yo_va"], dtype=torch.float32, device=device)
yd_va = torch.tensor(d["yd_va"], dtype=torch.float32, device=device)

# Load v4-D
v4d = TurningExpertNetwork(in_channels=6).to(device)
ckpt_v4 = torch.load(WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth", map_location=device, weights_only=False)
v4d.load_state_dict(ckpt_v4["model_state_dict"])
v4d.eval()

with torch.no_grad():
    dp, op, _ = v4d(X_va)

# Convert to degrees
gt_deg = (yo_va.cpu().numpy() * 2.0263271 - 1.0224137) * 180 / np.pi
pred_deg = (op.cpu().numpy() * 2.0263271 - 1.0224137) * 180 / np.pi

# Check on high-turn windows (|gt_deg| > 5 deg)
high_idx = np.where(np.abs(gt_deg) > 5.0)[0]
print(f"Validation windows with |gt_deg| > 5 deg: {len(high_idx)} / {len(gt_deg)}")
gt_high = gt_deg[high_idx]
pred_high = pred_deg[high_idx]

print(f"GT High Turn: Mean Abs = {np.mean(np.abs(gt_high)):.2f} deg, Max Abs = {np.max(np.abs(gt_high)):.2f} deg")
print(f"Pred High Turn: Mean Abs = {np.mean(np.abs(pred_high)):.2f} deg, Max Abs = {np.max(np.abs(pred_high)):.2f} deg")
print(f"Ratio Pred / GT on High Turns: {np.mean(np.abs(pred_high)) / np.mean(np.abs(gt_high)):.3f}")

# Check on extreme-turn windows (|gt_deg| > 15 deg)
ext_idx = np.where(np.abs(gt_deg) > 15.0)[0]
print(f"\nValidation windows with |gt_deg| > 15 deg: {len(ext_idx)} / {len(gt_deg)}")
if len(ext_idx) > 0:
    gt_ext = gt_deg[ext_idx]
    pred_ext = pred_deg[ext_idx]
    print(f"GT Ext Turn: Mean Abs = {np.mean(np.abs(gt_ext)):.2f} deg")
    print(f"Pred Ext Turn: Mean Abs = {np.mean(np.abs(pred_ext)):.2f} deg")
    print(f"Ratio Pred / GT on Extreme Turns: {np.mean(np.abs(pred_ext)) / np.mean(np.abs(gt_ext)):.3f}")
