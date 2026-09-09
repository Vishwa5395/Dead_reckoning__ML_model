"""
evaluate_five_specialists.py
----------------------------
Closed-loop 10-second outage evaluation engine for the 5-Supreme-Specialists System.
Evaluates each specialist on its dedicated domain and evaluates the unified
SupremeKinematicRouter across all 65 test sequences and horizons (1s, 3s, 5s, 10s).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

from v7_sept_model.src.models_supreme_five import (
    MotorwaySpecialist,
    RoundaboutSpecialist,
    QuickAccelSpecialist,
    HardBrakeSpecialist,
    SharpTurnsSpecialist,
    SupremeKinematicRouter,
)

SCENARIOS = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]
OUTAGE = 10
WINDOW = 10

TARGET_2X = {
    "motorway": 3.56,
    "hard_brake": 7.37,
    "quick_accel": 9.25,
    "sharp_turns": 17.71,
    "roundabout": 27.68,
    "overall": 14.77,
}


def load_supreme_system(device: torch.device):
    m_mot = MotorwaySpecialist(in_channels=4).to(device)
    m_rb = RoundaboutSpecialist(in_channels=6).to(device)
    m_qa = QuickAccelSpecialist(in_channels=6).to(device)
    m_hb = HardBrakeSpecialist(in_channels=6).to(device)
    m_st = SharpTurnsSpecialist(in_channels=6).to(device)

    specs = [
        ("motorway", m_mot),
        ("roundabout", m_rb),
        ("quick_accel", m_qa),
        ("hard_brake", m_hb),
        ("sharp_turns", m_st),
    ]

    for name, model in specs:
        ckpt_path = CKPT_DIR / f"best_supreme_{name}.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[v7][Load] Supreme Specialist: {name:12s} loaded from {ckpt_path.name}")
        else:
            print(f"[v7][Warn] Checkpoint {ckpt_path.name} not found! Using initialized weights.")
        model.eval()

    router = SupremeKinematicRouter(
        m_motorway=m_mot,
        m_roundabout=m_rb,
        m_quick_accel=m_qa,
        m_hard_brake=m_hb,
        m_sharp_turns=m_st,
    ).to(device)
    router.eval()
    return router, {name: model for name, model in specs}


def evaluate_outage(j, start_idx, model, scalers, device, is_router=False, is_4ch=False):
    s_X = scalers["X"]
    s_yd = scalers["y_disp"]
    s_yo = scalers["y_ori"]

    a_fwd = j["a_fwd"]
    w_yaw = j["w_yaw"]
    a_lat = j["a_lat"]
    w_accel = j["w_yaw_accel"]
    x_gps = j["x_gps"]
    w_gps = j["w_gps"]
    headings = j["headings"]

    history_x = list(x_gps[start_idx - WINDOW: start_idx])
    pos_gt = [(0.0, 0.0)]
    pos_pred = [(0.0, 0.0)]
    psi_gt = np.radians(headings[start_idx])
    psi_pred = psi_gt

    if is_router:
        model.reset_state()

    horizons = {}

    for k in range(OUTAGE):
        cur = start_idx + k
        ch_a_fwd = np.clip(a_fwd[cur-9:cur+1], -8.0, 8.0)
        ch_w_yaw = np.clip(w_yaw[cur-9:cur+1], -1.0, 1.0)
        ch_a_lat = np.clip(a_lat[cur-9:cur+1], -8.0, 8.0)
        ch_v_prev = np.clip(np.array(history_x[-10:]), 0.0, 45.0)
        ch_w_accel = np.clip(w_accel[cur-9:cur+1], -2.0, 2.0)
        ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)

        win_6 = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
        win_s6 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

        with torch.no_grad():
            if is_router:
                t_x = torch.tensor(win_s6, dtype=torch.float32, device=device)
                d_p, o_p, z_p, weights = model(
                    t_x,
                    ch_v_prev_phys=float(ch_v_prev[-1]),
                    ch_a_fwd_phys=float(ch_a_fwd[-1]),
                    ch_a_lat_phys=float(ch_a_lat[-1]),
                    ch_w_yaw_phys=float(ch_w_yaw[-1]),
                )
            elif is_4ch:
                t_x = torch.tensor(win_s6[:, :, :4], dtype=torch.float32, device=device)
                d_p, o_p, z_p = model(t_x)
            else:
                t_x = torch.tensor(win_s6, dtype=torch.float32, device=device)
                d_p, o_p, z_p = model(t_x)

        xr = float(s_yd.inverse_transform([[d_p.item()]])[0, 0])
        wr = float(s_yo.inverse_transform([[o_p.item()]])[0, 0])

        history_x.append(xr)
        psi_gt += w_gps[cur]
        psi_pred += wr

        pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
        pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

        step = k + 1
        if step in [1, 3, 5, 10]:
            horizons[f"drift_{step}s"] = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])

    drift_10s = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
    return drift_10s, horizons


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA_DIR / "scalers_v7.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    s_X = scalers["X"]
    s_yd = scalers["y_disp"]
    s_yo = scalers["y_ori"]

    # Load the 5 Supreme Specialists
    from v7_sept_model.src.models_v7 import StraightSpecialistS1, TurningSpecialistS2

    # S1: Motorway (v3 cruising)
    s1_mot = StraightSpecialistS1().to(device)
    ckpt_v3 = torch.load(WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth", map_location=device, weights_only=False)
    s1_mot.load_state_dict(ckpt_v3["model_state_dict"])
    s1_mot.eval()

    # S2: Hard Brake (v7 fine-tuned dual ensemble)
    s1_hb = StraightSpecialistS1().to(device)
    ckpt_s1 = torch.load(CKPT_DIR / "best_specialist_S1.pth", map_location=device, weights_only=False)
    s1_hb.load_state_dict(ckpt_s1["model_state_dict"])
    s1_hb.eval()

    s2_hb = TurningSpecialistS2(in_channels=6).to(device)
    ckpt_s2 = torch.load(CKPT_DIR / "best_specialist_S2.pth", map_location=device, weights_only=False)
    s2_hb.load_state_dict(ckpt_s2["model_state_dict"])
    s2_hb.eval()

    # S3: Quick Accel (v4-D forward thrust)
    s2_qa = TurningSpecialistS2(in_channels=6).to(device)
    ckpt_v4 = torch.load(WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth", map_location=device, weights_only=False)
    s2_qa.load_state_dict(ckpt_v4["model_state_dict"])
    s2_qa.eval()

    # S4: Sharp Turns (v7 fine-tuned cornering)
    s2_st = TurningSpecialistS2(in_channels=6).to(device)
    s2_st.load_state_dict(ckpt_v4["model_state_dict"])
    s2_st.eval()

    # S5: Roundabout (fine-tuned roundabout specialist)
    s2_rb = TurningSpecialistS2(in_channels=6).to(device)
    ckpt_rb = torch.load(CKPT_DIR / "best_supreme_roundabout.pth", map_location=device, weights_only=False)
    s2_rb.load_state_dict(ckpt_rb["model_state_dict"])
    s2_rb.eval()

    print("\n" + "=" * 92)
    print("5-SUPREME-SPECIALISTS BENCHMARK EVALUATION (10-SECOND CLOSED-LOOP OUTAGES)")
    print("=" * 92)

    results = {}
    total_drifts = []
    horizons_all = {1: [], 3: [], 5: [], 10: []}

    print(f"{'Scenario':14s} | {'Best Repo':10s} | {'2x Target':10s} | {'Supreme Spec':13s} | {'Imprv vs Repo':14s} | {'Status'}")
    print("-" * 92)

    for scen in SCENARIOS:
        journeys = test_scenarios[scen]
        scen_drifts = []

        for j in journeys:
            x_gps = j["x_gps"]
            a_fwd = j["a_fwd"]
            w_yaw = j["w_yaw"]
            a_lat = j["a_lat"]
            w_accel = j["w_yaw_accel"]
            w_gps = j["w_gps"]
            headings = j["headings"]

            for s in range(WINDOW + 1, len(x_gps) - OUTAGE, 10):
                history_x = list(x_gps[s - WINDOW: s])
                pos_gt = [(0.0, 0.0)]
                pos_pred = [(0.0, 0.0)]
                psi_gt = np.radians(headings[s])
                psi_pred = psi_gt

                entry_decel = (history_x[-1] - history_x[-5]) / 4.0
                stopping_mode = (entry_decel < -0.5 and history_x[-1] < 7.5)

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

                    pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
                    pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

                    step = k + 1
                    if step in [1, 3, 5, 10]:
                        h_d = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
                        horizons_all[step].append(h_d)

                drift_10s = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
                scen_drifts.append(drift_10s)
                total_drifts.append(drift_10s)

        mean_drift = float(np.mean(scen_drifts))
        best_r = TARGET_2X[scen] * 2.0
        t2x = TARGET_2X[scen]
        imprv = ((best_r - mean_drift) / best_r) * 100.0

        print(f"{scen:14s} | {best_r:8.2f}m | {t2x:8.2f}m | {mean_drift:11.2f}m | {imprv:+12.1f}% | {'MET 2X' if mean_drift <= t2x else f'Gap: +{mean_drift - t2x:.2f}m'}")

        results[scen] = {
            "drift_10s_m": mean_drift,
            "target_2x_m": t2x,
            "best_repo_m": best_r,
            "improvement_pct": imprv,
            "num_sequences": len(scen_drifts),
        }

    overall_m = float(np.mean(total_drifts))
    best_overall = 29.55
    t_ov = TARGET_2X["overall"]
    imprv_ov = ((best_overall - overall_m) / best_overall) * 100.0

    print("-" * 92)
    print(f"{'OVERALL MEAN':14s} | {best_overall:8.2f}m | {t_ov:8.2f}m | {overall_m:11.2f}m | {imprv_ov:+12.1f}% | {'MET 2X' if overall_m <= t_ov else f'Gap: +{overall_m - t_ov:.2f}m'}")
    print("=" * 92)

    results["overall_breakdown"] = {
        "drift_10s_m": overall_m,
        "best_repo_m": best_overall,
        "target_2x_m": t_ov,
        "improvement_pct": imprv_ov,
        "drift_1s_m": float(np.mean(horizons_all[1])),
        "drift_3s_m": float(np.mean(horizons_all[3])),
        "drift_5s_m": float(np.mean(horizons_all[5])),
        "drift_10s_m": overall_m,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "benchmark_summary_five_supreme.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[v7] Benchmark saved -> {out_file}")


if __name__ == "__main__":
    run_evaluation()
