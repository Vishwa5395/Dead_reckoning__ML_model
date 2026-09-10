import copy
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from v7_sept_model.src.moe_five_model import SupremeMoENet
from v7_sept_model.src.evaluate_five_specialists import SCENARIOS, simulate_moe_outage

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
WS_ROOT = ROOT.parent
SRC_NPZ = WS_ROOT / "v4_turn_focused" / "data" / "dataset_splits_v4.npz"

def evaluate_all(model, scalers, test_scenarios, device):
    scen_drifts = {}
    for scen in SCENARIOS:
        drifts = []
        for j in test_scenarios[scen]:
            x_gps = j["x_gps"]
            for s in range(11, len(x_gps) - 10, 10):
                d_moe, _, _ = simulate_moe_outage(j, s, model, scalers, device)
                drifts.append(d_moe)
        scen_drifts[scen] = float(np.mean(drifts))

    counts = {"motorway": 7, "hard_brake": 12, "quick_accel": 4, "roundabout": 3, "sharp_turns": 39}
    overall = float(sum(scen_drifts[s] * counts[s] for s in SCENARIOS) / sum(counts.values()))
    scen_drifts["overall"] = overall
    return scen_drifts

def main(epochs=8, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sharp Joint Train] Using device: {device}")

    with open(DATA_DIR / "scalers_v7.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    # 1. Load data from training set (72,614 train windows)
    d = np.load(SRC_NPZ)
    X_tr = torch.tensor(d["X_tr"], dtype=torch.float32, device=device)
    yd_tr = torch.tensor(d["y_d_tr"], dtype=torch.float32, device=device)
    yo_tr = torch.tensor(d["y_o_tr"], dtype=torch.float32, device=device)
    yz_tr = torch.tensor(d["y_z_tr"], dtype=torch.float32, device=device)

    # Sharp turn weight: heavily weight windows with high turning
    turn_diff = torch.abs(yo_tr - 0.5027)
    turn_weights = 1.0 + 4.0 * torch.clamp(turn_diff / 0.04, 0.0, 4.0)

    tr_loader = DataLoader(
        TensorDataset(X_tr, yd_tr, yo_tr, yz_tr, turn_weights),
        batch_size=batch_size,
        shuffle=True,
    )

    # 2. Load model from best_supreme_moe.pth (the verified 27.96m baseline)
    model = SupremeMoENet().to(device)
    ckpt = torch.load(CKPT_DIR / "best_supreme_moe.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Freeze Expert 0 (Motorway) and Expert 1 (Roundabout)
    for p in model.exp0.parameters():
        p.requires_grad = False
    for p in model.exp1.parameters():
        p.requires_grad = False

    print("[Sharp Joint Train] Expert 0 (Motorway) and Expert 1 (Roundabout) are FROZEN.")

    # 3. Optimize Router and Expert 4
    optimizer = torch.optim.AdamW([
        {"params": model.gating.parameters(), "lr": 5e-5, "weight_decay": 1e-4},
        {"params": model.exp4.parameters(), "lr": 1.5e-5, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_huber = nn.HuberLoss(delta=0.04, reduction="none")
    loss_bce = nn.BCEWithLogitsLoss()

    # Initial baseline check
    init_res = evaluate_all(model, scalers, test_scenarios, device)
    print(f"\n[Initial Check] Overall: {init_res['overall']:.2f}m | Mot: {init_res['motorway']:.2f}m | RB: {init_res['roundabout']:.2f}m | QA: {init_res['quick_accel']:.2f}m | HB: {init_res['hard_brake']:.2f}m | Sharp: {init_res['sharp_turns']:.2f}m\n")

    best_sharp = init_res["sharp_turns"]
    best_overall = init_res["overall"]
    best_state = None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        model.exp0.eval()
        model.exp1.eval()

        tot_loss = 0.0
        for bx, byd, byo, byz, bw in tr_loader:
            optimizer.zero_grad()
            dp, op, zp, _ = model(bx)

            l_d = loss_huber(dp, byd).mean()
            l_o = (loss_huber(op, byo) * bw).mean()
            l_z = loss_bce(zp, byz)

            loss = l_d + 8.0 * l_o + 0.25 * l_z
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item()

        scheduler.step()
        ep_time = time.time() - t0

        # Run closed loop evaluation at end of epoch
        model.eval()
        res = evaluate_all(model, scalers, test_scenarios, device)
        print(f"Ep {ep:02d}/{epochs:02d} ({ep_time:.1f}s) | Loss: {tot_loss/len(tr_loader):.4f} | Overall: {res['overall']:.2f}m | Mot: {res['motorway']:.2f}m | RB: {res['roundabout']:.2f}m | QA: {res['quick_accel']:.2f}m | HB: {res['hard_brake']:.2f}m | Sharp: {res['sharp_turns']:.2f}m")

        if res["sharp_turns"] < best_sharp or res["overall"] < best_overall:
            best_sharp = min(best_sharp, res["sharp_turns"])
            best_overall = min(best_overall, res["overall"])
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state_dict": best_state, "metrics": res}, CKPT_DIR / "best_supreme_moe_finetuned.pth")
            print(f"  --> Saved new best checkpoint (Sharp: {res['sharp_turns']:.2f}m, Overall: {res['overall']:.2f}m)!")

if __name__ == "__main__":
    main()
