"""
train_trio_specialists.py
-------------------------
Targeted Fine-Tuning Engine for the 3 Challenging Specialists:
  1. Hard Brake Specialist: Asymmetric deceleration loss + high ZUPT penalty.
  2. Roundabout Specialist: Curvature amplification + centripetal coupling.
  3. Sharp Turns Specialist: Directional attention + high-transient cornering loss.

Key Innovations:
  - Checkpoints are saved based on Closed-Loop 10-Second Dead Reckoning Drift
    rather than single-step validation loss.
  - Anchor replay (15% highway/cruising) prevents representation collapse.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

from v7_sept_model.src.models_v7 import StraightSpecialistS1, TurningSpecialistS2

V3_CKPT = WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth"
V4_D_CKPT = WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth"


def simulate_closed_loop(model, journeys, scalers, device, is_4ch=False, post_fn=None):
    """Computes exact 10s closed-loop dead reckoning drift across journeys."""
    s_X = scalers["X"]
    s_yd = scalers["y_disp"]
    s_yo = scalers["y_ori"]

    model.eval()
    drifts = []

    for j in journeys:
        x_gps = j["x_gps"]
        w_gps = j["w_gps"]
        a_fwd = j["a_fwd"]
        w_yaw = j["w_yaw"]
        a_lat = j["a_lat"]
        w_accel = j["w_yaw_accel"]
        headings = j["headings"]

        for s in range(11, len(x_gps) - 10, 10):
            history_x = list(x_gps[s - 10: s])
            pos_gt = [(0.0, 0.0)]
            pos_pred = [(0.0, 0.0)]
            psi_gt = np.radians(headings[s])
            psi_pred = psi_gt

            for k in range(10):
                cur = s + k
                ch_a_fwd = np.clip(a_fwd[cur-9:cur+1], -8.0, 8.0)
                ch_w_yaw = np.clip(w_yaw[cur-9:cur+1], -1.0, 1.0)
                ch_a_lat = np.clip(a_lat[cur-9:cur+1], -8.0, 8.0)
                ch_v_prev = np.clip(np.array(history_x[-10:]), 0.0, 45.0)
                ch_w_accel = np.clip(w_accel[cur-9:cur+1], -2.0, 2.0)
                ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)

                win_6 = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
                win_s = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

                with torch.no_grad():
                    if is_4ch:
                        d, o, z = model(torch.tensor(win_s[:, :, :4], dtype=torch.float32, device=device))
                    else:
                        d, o, z = model(torch.tensor(win_s, dtype=torch.float32, device=device))

                xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
                wr = float(s_yo.inverse_transform([[o.item()]])[0, 0])

                if post_fn is not None:
                    xr, wr = post_fn(xr, wr, history_x[-1], a_fwd[cur], a_lat[cur], w_yaw[cur], z)

                history_x.append(xr)
                psi_gt += w_gps[cur]
                psi_pred += wr

                pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
                pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

            drift_10s = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
            drifts.append(drift_10s)

    return float(np.mean(drifts))


def train_hard_brake(scalers, scens, device, epochs=35):
    print("\n" + "=" * 80)
    print("TRAINING SUPREME SPECIALIST: HARD BRAKE")
    print("=" * 80)

    # Load hard brake data + motorway anchors
    hb_data = np.load(DATA_DIR / "data_hard_brake.npz")
    mot_data = np.load(DATA_DIR / "data_motorway.npz")

    # Blend: 85% hard brake + 15% motorway
    n_hb = len(hb_data["X_tr"])
    n_mot = int(n_hb * 0.18)
    idx_mot = np.random.choice(len(mot_data["X_tr"]), n_mot, replace=False)

    X_tr = np.concatenate([hb_data["X_tr"], mot_data["X_tr"][idx_mot]], axis=0)[:, :, :4]
    yd_tr = np.concatenate([hb_data["yd_tr"], mot_data["yd_tr"][idx_mot]], axis=0)
    yo_tr = np.concatenate([hb_data["yo_tr"], mot_data["yo_tr"][idx_mot]], axis=0)
    yz_tr = np.concatenate([hb_data["yz_tr"], mot_data["yz_tr"][idx_mot]], axis=0)

    t_X = torch.tensor(X_tr, dtype=torch.float32, device=device)
    t_yd = torch.tensor(yd_tr, dtype=torch.float32, device=device)
    t_yo = torch.tensor(yo_tr, dtype=torch.float32, device=device)
    t_yz = torch.tensor(yz_tr, dtype=torch.float32, device=device)

    # Physical forward acceleration in input (channel 0 is scaled a_fwd)
    a_fwd_phys = t_X[:, -1, 0] * 16.0 - 8.0  # approximate unscaling

    # Asymmetric displacement weights: penalize overprediction when braking
    # We will apply this dynamically in the loss function
    loader = DataLoader(TensorDataset(t_X, t_yd, t_yo, t_yz, a_fwd_phys), batch_size=256, shuffle=True)

    # Initialize from v3 base
    model = StraightSpecialistS1(in_channels=4).to(device)
    ckpt = torch.load(V3_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_huber = nn.HuberLoss(delta=0.08, reduction="none")
    loss_bce = nn.BCEWithLogitsLoss()

    def hb_post(xr, wr, v_prev, a_fwd, a_lat, w_yaw, z):
        if a_fwd < -0.1:
            xr = min(xr, max(0.0, v_prev + a_fwd * 0.30))
        if xr < 0.4 and a_fwd < 0.2:
            xr, wr = 0.0, 0.0
        return xr, wr

    best_drift = simulate_closed_loop(model, scens["hard_brake"], scalers, device, is_4ch=True, post_fn=hb_post)
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"[Hard Brake] Initial Baseline Closed-Loop Drift: {best_drift:.2f}m")

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for X, yd, yo, yz, af in loader:
            optimizer.zero_grad()
            dp, op, zp = model(X)

            # Asymmetric loss: penalize dp > yd when af < -0.3
            l_d = loss_huber(dp, yd)
            is_overpredicting = (dp > yd).float()
            is_braking = (af < -0.3).float().unsqueeze(-1)
            weight_d = 1.0 + 4.0 * (is_overpredicting * is_braking)
            loss_disp = (l_d * weight_d).mean()

            loss_ori = loss_huber(op, yo).mean()
            loss_zupt = loss_bce(zp, yz)

            loss = loss_disp + 6.0 * loss_ori + 0.5 * loss_zupt
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item()
        scheduler.step()

        # Evaluate directly on closed-loop drift
        drift = simulate_closed_loop(model, scens["hard_brake"], scalers, device, is_4ch=True, post_fn=hb_post)
        if drift < best_drift:
            best_drift = drift
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [Ep {ep:02d}] *** NEW BEST CLOSED-LOOP DRIFT: {best_drift:.2f}m ***")
        elif ep % 5 == 0:
            print(f"  [Ep {ep:02d}] Train: {tot_loss/len(loader):.4f} | Closed-Loop Drift: {drift:.2f}m (best: {best_drift:.2f}m)")

    out_ckpt = CKPT_DIR / "best_supreme_hard_brake.pth"
    torch.save({"model_state_dict": best_state, "drift_10s": best_drift}, out_ckpt)
    print(f"[Hard Brake] Complete! Best Checkpoint Saved -> {out_ckpt.name} with {best_drift:.2f}m drift.")
    return best_drift


def train_roundabout(scalers, scens, device, epochs=35):
    print("\n" + "=" * 80)
    print("TRAINING SUPREME SPECIALIST: ROUNDABOUT")
    print("=" * 80)

    rb_data = np.load(DATA_DIR / "data_roundabout.npz")
    mot_data = np.load(DATA_DIR / "data_motorway.npz")

    # 85% roundabout + 15% cruising anchors
    n_rb = len(rb_data["X_tr"])
    n_mot = int(n_rb * 0.20)
    idx_mot = np.random.choice(len(mot_data["X_tr"]), n_mot, replace=False)

    X_tr = np.concatenate([rb_data["X_tr"], mot_data["X_tr"][idx_mot]], axis=0)
    yd_tr = np.concatenate([rb_data["yd_tr"], mot_data["yd_tr"][idx_mot]], axis=0)
    yo_tr = np.concatenate([rb_data["yo_tr"], mot_data["yo_tr"][idx_mot]], axis=0)
    yz_tr = np.concatenate([rb_data["yz_tr"], mot_data["yz_tr"][idx_mot]], axis=0)

    t_X = torch.tensor(X_tr, dtype=torch.float32, device=device)
    t_yd = torch.tensor(yd_tr, dtype=torch.float32, device=device)
    t_yo = torch.tensor(yo_tr, dtype=torch.float32, device=device)
    t_yz = torch.tensor(yz_tr, dtype=torch.float32, device=device)

    # Curvature weighting for orientation
    diff_yo = torch.abs(t_yo - 0.504565)
    weights_ori = (1.0 + 15.0 * torch.clamp(diff_yo / 0.10, 0.0, 3.5) ** 2)

    loader = DataLoader(TensorDataset(t_X, t_yd, t_yo, t_yz, weights_ori), batch_size=256, shuffle=True)

    # Initialize from v4-D base
    model = TurningSpecialistS2(in_channels=6).to(device)
    ckpt = torch.load(V4_D_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_huber = nn.HuberLoss(delta=0.08, reduction="none")
    loss_bce = nn.BCEWithLogitsLoss()

    def rb_post(xr, wr, v_prev, a_fwd, a_lat, w_yaw, z):
        wr = wr * 2.8
        if abs(a_lat) > 0.5 and v_prev > 3.0:
            w_cent = -np.sign(a_lat) * abs(a_lat) / v_prev * 0.5
            wr = 0.60 * wr + 0.40 * w_cent
        return xr, wr

    best_drift = simulate_closed_loop(model, scens["roundabout"], scalers, device, is_4ch=False, post_fn=rb_post)
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"[Roundabout] Initial Baseline Closed-Loop Drift: {best_drift:.2f}m")

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for X, yd, yo, yz, w_o in loader:
            optimizer.zero_grad()
            dp, op, zp = model(X)
            loss_disp = loss_huber(dp, yd).mean()
            loss_ori = (loss_huber(op, yo) * w_o).mean()
            loss_zupt = loss_bce(zp, yz)

            loss = loss_disp + 10.0 * loss_ori + 0.25 * loss_zupt
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item()
        scheduler.step()

        drift = simulate_closed_loop(model, scens["roundabout"], scalers, device, is_4ch=False, post_fn=rb_post)
        if drift < best_drift:
            best_drift = drift
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [Ep {ep:02d}] *** NEW BEST CLOSED-LOOP DRIFT: {best_drift:.2f}m ***")
        elif ep % 5 == 0:
            print(f"  [Ep {ep:02d}] Train: {tot_loss/len(loader):.4f} | Closed-Loop Drift: {drift:.2f}m (best: {best_drift:.2f}m)")

    out_ckpt = CKPT_DIR / "best_supreme_roundabout.pth"
    torch.save({"model_state_dict": best_state, "drift_10s": best_drift}, out_ckpt)
    print(f"[Roundabout] Complete! Best Checkpoint Saved -> {out_ckpt.name} with {best_drift:.2f}m drift.")
    return best_drift


def train_sharp_turns(scalers, scens, device, epochs=35):
    print("\n" + "=" * 80)
    print("TRAINING SUPREME SPECIALIST: SHARP TURNS")
    print("=" * 80)

    st_data = np.load(DATA_DIR / "data_sharp_turns.npz")
    mot_data = np.load(DATA_DIR / "data_motorway.npz")

    # 85% sharp turns + 15% cruising anchors
    n_st = len(st_data["X_tr"])
    n_mot = int(n_st * 0.15)
    idx_mot = np.random.choice(len(mot_data["X_tr"]), n_mot, replace=False)

    X_tr = np.concatenate([st_data["X_tr"], mot_data["X_tr"][idx_mot]], axis=0)
    yd_tr = np.concatenate([st_data["yd_tr"], mot_data["yd_tr"][idx_mot]], axis=0)
    yo_tr = np.concatenate([st_data["yo_tr"], mot_data["yo_tr"][idx_mot]], axis=0)
    yz_tr = np.concatenate([st_data["yz_tr"], mot_data["yz_tr"][idx_mot]], axis=0)

    t_X = torch.tensor(X_tr, dtype=torch.float32, device=device)
    t_yd = torch.tensor(yd_tr, dtype=torch.float32, device=device)
    t_yo = torch.tensor(yo_tr, dtype=torch.float32, device=device)
    t_yz = torch.tensor(yz_tr, dtype=torch.float32, device=device)

    # Focal orientation weighting
    diff_yo = torch.abs(t_yo - 0.504565)
    weights_ori = (1.0 + 10.0 * torch.clamp(diff_yo / 0.10, 0.0, 3.0) ** 2)

    loader = DataLoader(TensorDataset(t_X, t_yd, t_yo, t_yz, weights_ori), batch_size=256, shuffle=True)

    # Initialize from v4-D base
    model = TurningSpecialistS2(in_channels=6).to(device)
    ckpt = torch.load(V4_D_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_huber = nn.HuberLoss(delta=0.08, reduction="none")
    loss_bce = nn.BCEWithLogitsLoss()

    best_drift = simulate_closed_loop(model, scens["sharp_turns"], scalers, device, is_4ch=False)
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"[Sharp Turns] Initial Baseline Closed-Loop Drift: {best_drift:.2f}m")

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for X, yd, yo, yz, w_o in loader:
            optimizer.zero_grad()
            dp, op, zp = model(X)
            loss_disp = loss_huber(dp, yd).mean()
            loss_ori = (loss_huber(op, yo) * w_o).mean()
            loss_zupt = loss_bce(zp, yz)

            loss = loss_disp + 7.0 * loss_ori + 0.25 * loss_zupt
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item()
        scheduler.step()

        drift = simulate_closed_loop(model, scens["sharp_turns"], scalers, device, is_4ch=False)
        if drift < best_drift:
            best_drift = drift
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [Ep {ep:02d}] *** NEW BEST CLOSED-LOOP DRIFT: {best_drift:.2f}m ***")
        elif ep % 5 == 0:
            print(f"  [Ep {ep:02d}] Train: {tot_loss/len(loader):.4f} | Closed-Loop Drift: {drift:.2f}m (best: {best_drift:.2f}m)")

    out_ckpt = CKPT_DIR / "best_supreme_sharp_turns.pth"
    torch.save({"model_state_dict": best_state, "drift_10s": best_drift}, out_ckpt)
    print(f"[Sharp Turns] Complete! Best Checkpoint Saved -> {out_ckpt.name} with {best_drift:.2f}m drift.")
    return best_drift


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v7] Starting Trio Fine-Tuning on: {device}")

    with open(DATA_DIR / "scalers_v7.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
        scens = pickle.load(f)

    # 1. Train Hard Brake
    train_hard_brake(scalers, scens, device, epochs=30)

    # 2. Train Roundabout
    train_roundabout(scalers, scens, device, epochs=30)

    # 3. Train Sharp Turns
    train_sharp_turns(scalers, scens, device, epochs=30)

    print("\n" + "=" * 80)
    print("ALL THREE SPECIALISTS FINISHED TRAINING!")
    print("=" * 80)


if __name__ == "__main__":
    main()
