import sys
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from v7_sept_model.src.moe_five_model import TurningExpertNetwork

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
WS_ROOT = ROOT.parent
V4_D_CKPT = WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth"

def train_sharp_turn_specialist(epochs=40, batch_size=128, lr=2.0e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TrainSharp] Device: {device}")

    d = np.load(DATA_DIR / "data_sharp_turns.npz")
    X_tr = torch.tensor(d["X_tr"], dtype=torch.float32)
    yd_tr = torch.tensor(d["yd_tr"], dtype=torch.float32)
    yo_tr = torch.tensor(d["yo_tr"], dtype=torch.float32)
    yz_tr = torch.tensor(d["yz_tr"], dtype=torch.float32)

    X_va = torch.tensor(d["X_va"], dtype=torch.float32, device=device)
    yd_va = torch.tensor(d["yd_va"], dtype=torch.float32, device=device)
    yo_va = torch.tensor(d["yo_va"], dtype=torch.float32, device=device)
    yz_va = torch.tensor(d["yz_va"], dtype=torch.float32, device=device)

    # Convert yo to degrees to compute sampling weights and loss weights
    yo_deg_tr = (yo_tr * 2.0263271 - 1.0224137) * 180 / np.pi
    yo_deg_va = (yo_va * 2.0263271 - 1.0224137) * 180 / np.pi

    # Sampling weights: boost probability of high-turn samples (|deg| > 3.0)
    sample_weights = 1.0 + 8.0 * torch.clamp(torch.abs(yo_deg_tr) / 3.0, 0.0, 5.0)
    sampler = WeightedRandomSampler(sample_weights.squeeze().tolist(), num_samples=len(X_tr), replacement=True)

    tr_loader = DataLoader(
        TensorDataset(X_tr, yd_tr, yo_tr, yz_tr, sample_weights),
        batch_size=batch_size,
        sampler=sampler,
    )

    model = TurningExpertNetwork(in_channels=6).to(device)
    base_state = torch.load(V4_D_CKPT, map_location=device, weights_only=False)["model_state_dict"]
    model.load_state_dict(base_state)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_huber = nn.HuberLoss(delta=0.02, reduction="none")
    loss_bce = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_ratio = 0.0
    best_state = None

    print("\nStarting specialized training on sharp turns...")
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        tot_l_d = 0.0
        tot_l_o = 0.0

        for bx, byd, byo, byz, bw in tr_loader:
            bx = bx.to(device)
            byd = byd.to(device)
            byo = byo.to(device)
            byz = byz.to(device)
            bw = bw.to(device)

            optimizer.zero_grad()
            dp, op, zp = model(bx)

            # Displacement loss
            l_d = loss_huber(dp, byd).mean()

            # Orientation loss with Asymmetric Under-Turn Penalty:
            # If |pred| < |gt| during an acute turn, apply an extra 3x penalty!
            gt_diff = torch.abs(byo - 0.5045) # deviation from zero turn
            pred_diff = torch.abs(op - 0.5045)
            under_turn_mask = (gt_diff > 0.03) & (pred_diff < gt_diff)
            asym_weight = torch.where(under_turn_mask, 3.5, 1.0)

            l_o = (loss_huber(op, byo) * bw * asym_weight).mean()
            l_z = loss_bce(zp, byz)

            loss = l_d + 12.0 * l_o + 0.25 * l_z
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot_loss += loss.item()
            tot_l_d += l_d.item()
            tot_l_o += l_o.item()

        scheduler.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            dp_va, op_va, zp_va = model(X_va)
            l_d_va = loss_huber(dp_va, yd_va).mean().item()
            l_o_va = loss_huber(op_va, yo_va).mean().item()
            l_z_va = loss_bce(zp_va, yz_va).item()
            val_loss = l_d_va + 12.0 * l_o_va + 0.25 * l_z_va

            # Check turning ratio on validation high-turn windows (|gt_deg| > 5.0)
            pred_deg_va = (op_va * 2.0263271 - 1.0224137) * 180 / np.pi
            high_mask = torch.abs(yo_deg_va) > 5.0
            ratio = (torch.abs(pred_deg_va[high_mask]).mean() / torch.abs(yo_deg_va[high_mask]).mean()).item()

        if ep % 5 == 0 or ep == epochs or ep == 1:
            print(f"Ep {ep:02d}/{epochs:02d} | Train: {tot_loss/len(tr_loader):.4f} (l_o: {tot_l_o/len(tr_loader):.4f}) | Val: {val_loss:.4f} | Turn Ratio on >5°: {ratio:.3f}")

        if val_loss < best_val_loss and ratio > 0.40:
            best_val_loss = val_loss
            best_ratio = ratio
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "val_loss": val_loss, "turn_ratio": ratio}, CKPT_DIR / "best_supreme_sharp_turns.pth")

    if best_state is None:
        # Fallback to last
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"model_state_dict": best_state, "val_loss": val_loss, "turn_ratio": ratio}, CKPT_DIR / "best_supreme_sharp_turns.pth")

    print(f"\n[TrainSharp] Complete in {time.time()-t0:.1f}s | Best Val: {best_val_loss:.4f} | Best Turn Ratio: {best_ratio:.3f}")

if __name__ == "__main__":
    train_sharp_turn_specialist(epochs=35, batch_size=128, lr=1.8e-4)
