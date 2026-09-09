"""
plot_five_supreme.py
--------------------
Generates high-resolution publication trajectory comparisons for all 5 scenarios
under the Supreme 5-Expert Mixture-of-Experts (MoE) system.

100% Pure Neural Network Inference:
- ZERO scenario oracle labels
- ZERO manual post-processing constants or multipliers
- Pure forward pass through SupremeMoENet
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
ARTIFACT_DIR = Path(r"C:\Users\vrish\.gemini\antigravity-ide\brain\a39e438b-116e-439c-be2c-62bfabc31de4")

import sys
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from v7_sept_model.src.moe_five_model import SupremeMoENet
from v7_sept_model.src.models_v7 import TurningSpecialistS2

SCENARIOS = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]
OUTAGE = 10
WINDOW = 10


def generate_plots():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA_DIR / "scalers_v7.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
        scens = pickle.load(f)

    s_X = scalers["X"]
    s_yd = scalers["y_disp"]
    s_yo = scalers["y_ori"]

    # Load Supreme MoE
    moe = SupremeMoENet().to(device)
    ckpt_moe = torch.load(CKPT_DIR / "best_supreme_moe.pth", map_location=device, weights_only=False)
    moe.load_state_dict(ckpt_moe["model_state_dict"])
    moe.eval()

    # Load v4-D Baseline
    v4d = TurningSpecialistS2(in_channels=6).to(device)
    ckpt_v4 = torch.load(WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth", map_location=device, weights_only=False)
    v4d.load_state_dict(ckpt_v4["model_state_dict"])
    v4d.eval()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for scen in SCENARIOS:
        journeys = scens[scen]
        j = journeys[0]
        x_gps = j["x_gps"]
        w_gps = j["w_gps"]
        a_fwd = j["a_fwd"]
        w_yaw = j["w_yaw"]
        a_lat = j["a_lat"]
        w_accel = j["w_yaw_accel"]
        headings = j["headings"]

        s = 11

        history_moe = list(x_gps[s - WINDOW: s])
        history_v4d = list(x_gps[s - WINDOW: s])

        pos_gt = [(0.0, 0.0)]
        pos_moe = [(0.0, 0.0)]
        pos_v4d = [(0.0, 0.0)]
        pos_ins = [(0.0, 0.0)]

        psi_gt = np.radians(headings[s])
        psi_moe = psi_gt
        psi_v4d = psi_gt
        psi_ins = psi_gt
        v_ins = x_gps[s - 1]

        for k in range(OUTAGE):
            cur = s + k
            ch_a_fwd = np.clip(a_fwd[cur-9:cur+1], -8.0, 8.0)
            ch_w_yaw = np.clip(w_yaw[cur-9:cur+1], -1.0, 1.0)
            ch_a_lat = np.clip(a_lat[cur-9:cur+1], -8.0, 8.0)
            ch_v_prev_moe = np.clip(np.array(history_moe[-10:]), 0.0, 45.0)
            ch_v_prev_v4d = np.clip(np.array(history_v4d[-10:]), 0.0, 45.0)
            ch_w_accel = np.clip(w_accel[cur-9:cur+1], -2.0, 2.0)
            ch_cent_moe = np.clip(ch_a_lat - ch_v_prev_moe * ch_w_yaw, -8.0, 8.0)
            ch_cent_v4d = np.clip(ch_a_lat - ch_v_prev_v4d * ch_w_yaw, -8.0, 8.0)

            # MoE forward pass
            win_6_moe = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev_moe, ch_w_accel, ch_cent_moe], axis=-1)
            win_s6_moe = s_X.transform(win_6_moe.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

            with torch.no_grad():
                t_x_moe = torch.tensor(win_s6_moe, dtype=torch.float32, device=device)
                d_m, o_m, _, _ = moe(t_x_moe)

            xr_moe = float(s_yd.inverse_transform([[d_m.item()]])[0, 0])
            wr_moe = float(s_yo.inverse_transform([[o_m.item()]])[0, 0])

            # v4-D forward pass
            win_6_v4d = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev_v4d, ch_w_accel, ch_cent_v4d], axis=-1)
            win_s6_v4d = s_X.transform(win_6_v4d.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

            with torch.no_grad():
                t_x_v4d = torch.tensor(win_s6_v4d, dtype=torch.float32, device=device)
                d_v, o_v, _ = v4d(t_x_v4d)

            xr_v4d = float(s_yd.inverse_transform([[d_v.item()]])[0, 0])
            wr_v4d = float(s_yo.inverse_transform([[o_v.item()]])[0, 0])

            history_moe.append(xr_moe)
            history_v4d.append(xr_v4d)

            psi_gt += w_gps[cur]
            psi_moe += wr_moe
            psi_v4d += wr_v4d
            psi_ins += w_yaw[cur]

            v_ins = max(0.0, v_ins + a_fwd[cur])

            pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
            pos_moe.append((pos_moe[-1][0] + xr_moe * np.cos(psi_moe), pos_moe[-1][1] + xr_moe * np.sin(psi_moe)))
            pos_v4d.append((pos_v4d[-1][0] + xr_v4d * np.cos(psi_v4d), pos_v4d[-1][1] + xr_v4d * np.sin(psi_v4d)))
            pos_ins.append((pos_ins[-1][0] + v_ins * np.cos(psi_ins), pos_ins[-1][1] + v_ins * np.sin(psi_ins)))

        p_gt = np.array(pos_gt)
        p_moe = np.array(pos_moe)
        p_v4d = np.array(pos_v4d)
        p_ins = np.array(pos_ins)

        d_final_moe = np.hypot(p_moe[-1, 0] - p_gt[-1, 0], p_moe[-1, 1] - p_gt[-1, 1])
        d_final_v4d = np.hypot(p_v4d[-1, 0] - p_gt[-1, 0], p_v4d[-1, 1] - p_gt[-1, 1])
        d_final_ins = np.hypot(p_ins[-1, 0] - p_gt[-1, 0], p_ins[-1, 1] - p_gt[-1, 1])

        fig, ax = plt.subplots(figsize=(9, 7.5), dpi=120)
        ax.plot(p_gt[:, 1], p_gt[:, 0], "g-", linewidth=2.8, label="Ground Truth (GPS)", zorder=4)
        ax.plot(p_moe[:, 1], p_moe[:, 0], "b-", linewidth=2.4, label=f"Supreme MoE ({d_final_moe:.2f}m drift)", zorder=5)
        ax.plot(p_v4d[:, 1], p_v4d[:, 0], color="darkorange", linestyle="--", linewidth=2.0, label=f"v4-D Baseline ({d_final_v4d:.2f}m drift)", zorder=3)
        ax.plot(p_ins[:, 1], p_ins[:, 0], "r:", linewidth=1.8, label=f"Pure INS ({d_final_ins:.2f}m drift)", zorder=2)

        ax.scatter([0], [0], c="black", s=70, zorder=6, label="Outage Start")
        ax.scatter([p_gt[-1, 1]], [p_gt[-1, 0]], c="green", s=80, marker="x", zorder=6)
        ax.scatter([p_moe[-1, 1]], [p_moe[-1, 0]], c="blue", s=80, marker="o", zorder=6)

        ax.set_title(f"10s Autonomous GNSS Outage: {scen.replace('_', ' ').title()}", fontsize=14, weight="bold")
        ax.set_xlabel("East Position (m)", fontsize=11)
        ax.set_ylabel("North Position (m)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", fontsize=10)
        plt.tight_layout()

        out_plot = RESULTS_DIR / f"trajectory_supreme_{scen}.png"
        fig.savefig(out_plot)
        plt.close(fig)
        print(f"[Plot] Saved -> {out_plot.name}")

        # Also copy to artifact directory for markdown embedding
        if ARTIFACT_DIR.exists():
            art_dest = ARTIFACT_DIR / f"trajectory_supreme_{scen}.png"
            shutil.copy(out_plot, art_dest)
            print(f"[Plot] Copied to artifact -> {art_dest.name}")


if __name__ == "__main__":
    generate_plots()
