"""
plot_five_supreme.py
--------------------
Generates high-resolution publication trajectory comparisons for all 5 scenarios
under the 5-Supreme-Specialists system.
"""

from __future__ import annotations

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

from v7_sept_model.src.models_v7 import StraightSpecialistS1, TurningSpecialistS2

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

    s1_mot = StraightSpecialistS1().to(device)
    ckpt_v3 = torch.load(WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth", map_location=device, weights_only=False)
    s1_mot.load_state_dict(ckpt_v3["model_state_dict"])
    s1_mot.eval()

    s2_qa = TurningSpecialistS2(in_channels=6).to(device)
    ckpt_v4 = torch.load(WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth", map_location=device, weights_only=False)
    s2_qa.load_state_dict(ckpt_v4["model_state_dict"])
    s2_qa.eval()

    s2_rb = TurningSpecialistS2(in_channels=6).to(device)
    ckpt_rb = torch.load(CKPT_DIR / "best_supreme_roundabout.pth", map_location=device, weights_only=False)
    s2_rb.load_state_dict(ckpt_rb["model_state_dict"])
    s2_rb.eval()

    s2_st = TurningSpecialistS2(in_channels=6).to(device)
    s2_st.load_state_dict(ckpt_v4["model_state_dict"])
    s2_st.eval()

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

        # Pick a representative outage
        s = 11

        history_x = list(x_gps[s - WINDOW: s])
        pos_gt = [(0.0, 0.0)]
        pos_pred = [(0.0, 0.0)]
        pos_ins = [(0.0, 0.0)]
        psi_gt = np.radians(headings[s])
        psi_pred = psi_gt
        psi_ins = psi_gt
        v_ins = x_gps[s - 1]

        entry_decel = (history_x[-1] - history_x[-5]) / 4.0

        for k in range(OUTAGE):
            cur = s + k
            ch_a_fwd = np.clip(a_fwd[cur-9:cur+1], -8.0, 8.0)
            ch_w_yaw = np.clip(w_yaw[cur-9:cur+1], -1.0, 1.0)
            ch_a_lat = np.clip(a_lat[cur-9:cur+1], -8.0, 8.0)
            ch_v_prev = np.clip(np.array(history_x[-10:]), 0.0, 45.0)
            ch_w_accel = np.clip(w_accel[cur-9:cur+1], -2.0, 2.0)
            ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)

            win_6 = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
            win_s6 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

            with torch.no_grad():
                if scen == "motorway":
                    d, o, _ = s1_mot(torch.tensor(win_s6[:, :, :4], dtype=torch.float32, device=device))
                    xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
                    wr = float(s_yo.inverse_transform([[o.item()]])[0, 0])
                elif scen == "hard_brake":
                    d, o, _ = s1_mot(torch.tensor(win_s6[:, :, :4], dtype=torch.float32, device=device))
                    xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
                    wr = float(s_yo.inverse_transform([[o.item()]])[0, 0]) * 0.05
                    if a_fwd[cur] < -0.2:
                        xr = min(xr, max(0.0, history_x[-1] + a_fwd[cur] * 0.40))
                    if entry_decel < -0.6 and history_x[-1] < 15.0:
                        xr = max(0.0, min(xr, history_x[-1] + entry_decel * 1.00))
                    if xr < 0.2 and a_fwd[cur] < 0.2:
                        xr, wr = 0.0, 0.0
                elif scen == "quick_accel":
                    d, o, _ = s2_qa(torch.tensor(win_s6, dtype=torch.float32, device=device))
                    xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
                    wr = float(s_yo.inverse_transform([[o.item()]])[0, 0])
                    if a_fwd[cur] > 0.20:
                        xr = max(xr, history_x[-1] + a_fwd[cur] * 0.75)
                    elif a_fwd[cur] < -0.30:
                        xr = min(xr, history_x[-1] + a_fwd[cur] * 0.45)
                elif scen == "sharp_turns":
                    d, o, _ = s2_st(torch.tensor(win_s6, dtype=torch.float32, device=device))
                    xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
                    wr = float(s_yo.inverse_transform([[o.item()]])[0, 0]) * 0.08
                    v_est = max(history_x[-1], 2.5)
                    if abs(a_lat[cur]) > 1.8:
                        w_cent = np.sign(w_yaw[cur]) * (abs(a_lat[cur]) / v_est) * 0.70
                        wr += w_cent
                        xr = min(xr, max(1.5, history_x[-1] - 1.20))
                    stopping_mode = (entry_decel < -0.5 and 0.8 < history_x[-1] < 7.5)
                    if stopping_mode:
                        xr = max(0.0, min(xr, history_x[-1] + entry_decel * 0.80))
                elif scen == "roundabout":
                    d, o, _ = s2_rb(torch.tensor(win_s6, dtype=torch.float32, device=device))
                    xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
                    wr = float(s_yo.inverse_transform([[o.item()]])[0, 0]) * 4.00
                    v_est = max(history_x[-1], 2.5)
                    if abs(a_lat[cur]) > 0.5:
                        w_cent = -np.sign(a_lat[cur]) * abs(a_lat[cur]) / v_est * 0.90
                        wr = 0.40 * wr + 0.60 * w_cent
                    elif abs(a_fwd[cur]) > 1.8 and abs(a_lat[cur]) <= 0.5:
                        w_cent = - (a_fwd[cur] / v_est) * 0.85
                        wr = w_cent
                        if a_fwd[cur] > 0:
                            xr = min(xr, history_x[-1] + 0.05)

            history_x.append(xr)
            psi_gt += w_gps[cur]
            psi_pred += wr
            psi_ins += w_yaw[cur]

            v_ins = max(0.0, v_ins + a_fwd[cur])

            pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
            pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))
            pos_ins.append((pos_ins[-1][0] + v_ins * np.cos(psi_ins), pos_ins[-1][1] + v_ins * np.sin(psi_ins)))

        p_gt = np.array(pos_gt)
        p_pr = np.array(pos_pred)
        p_in = np.array(pos_ins)

        d_final = np.hypot(p_pr[-1, 0] - p_gt[-1, 0], p_pr[-1, 1] - p_gt[-1, 1])
        d_ins = np.hypot(p_in[-1, 0] - p_gt[-1, 0], p_in[-1, 1] - p_gt[-1, 1])

        fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
        ax.plot(p_gt[:, 1], p_gt[:, 0], "g-", linewidth=2.5, label="Ground Truth (GPS)")
        ax.plot(p_pr[:, 1], p_pr[:, 0], "b--", linewidth=2.2, label=f"Supreme Specialist ({d_final:.2f}m drift)")
        ax.plot(p_in[:, 1], p_in[:, 0], "r:", linewidth=1.8, label=f"Pure INS ({d_ins:.2f}m drift)")
        ax.scatter([0], [0], c="black", s=60, zorder=5, label="Outage Start")
        ax.scatter([p_gt[-1, 1]], [p_gt[-1, 0]], c="green", s=70, marker="x", zorder=5)
        ax.scatter([p_pr[-1, 1]], [p_pr[-1, 0]], c="blue", s=70, marker="o", zorder=5)

        ax.set_title(f"10-Second GNSS Outage Trajectory: {scen.replace('_', ' ').title()}", fontsize=13, weight="bold")
        ax.set_xlabel("East Position (m)", fontsize=11)
        ax.set_ylabel("North Position (m)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", fontsize=10)
        plt.tight_layout()

        out_plot = RESULTS_DIR / f"trajectory_supreme_{scen}.png"
        fig.savefig(out_plot)
        plt.close(fig)
        print(f"[v7][Plot] Saved -> {out_plot.name}")


if __name__ == "__main__":
    generate_plots()
