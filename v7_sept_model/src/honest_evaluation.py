"""
honest_evaluation.py
--------------------
Honest benchmark evaluation for the v7 5-Supreme-Specialists system.

Modes evaluated:
1. RAW NN (Old 5-Way Router): Pure neural network outputs with old overlapping router.
2. ROUTER + POST-PROCESSING (Old): Old router with test-set tuned postproc.
3. ORACLE + POST-PROCESSING: Known scenario labels (the original reported 21.47m number).
4. 3-REGIME ROUTER (Clean Physics): Sensor-based routing with 3 physically separable regimes
   and minimal first-principles physics (brake deceleration integration).
5. v4-D SINGLE MODEL BASELINE: Turn-focused ablation-D model without specialist routing.
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

import sys
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from v7_sept_model.src.models_v7 import StraightSpecialistS1, TurningSpecialistS2
from v7_sept_model.src.three_regime_router import ThreeRegimeRouter

SCENARIOS = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]
OUTAGE = 10
WINDOW = 10

BEST_REPO = {
    "motorway": 7.12,
    "roundabout": 55.36,
    "quick_accel": 18.50,
    "hard_brake": 14.74,
    "sharp_turns": 35.42,
}


def load_oracle_models(device):
    """Load models as used in the original evaluate_five_specialists.py."""
    s1_mot = StraightSpecialistS1().to(device)
    ckpt_v3 = torch.load(WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth",
                         map_location=device, weights_only=False)
    s1_mot.load_state_dict(ckpt_v3["model_state_dict"])
    s1_mot.eval()

    s2_qa = TurningSpecialistS2(in_channels=6).to(device)
    ckpt_v4 = torch.load(WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth",
                         map_location=device, weights_only=False)
    s2_qa.load_state_dict(ckpt_v4["model_state_dict"])
    s2_qa.eval()

    s2_st = TurningSpecialistS2(in_channels=6).to(device)
    s2_st.load_state_dict(ckpt_v4["model_state_dict"])
    s2_st.eval()

    s2_rb = TurningSpecialistS2(in_channels=6).to(device)
    ckpt_rb = torch.load(CKPT_DIR / "best_supreme_roundabout.pth", map_location=device, weights_only=False)
    s2_rb.load_state_dict(ckpt_rb["model_state_dict"])
    s2_rb.eval()

    return {
        "motorway": s1_mot,
        "hard_brake": s1_mot,
        "quick_accel": s2_qa,
        "sharp_turns": s2_st,
        "roundabout": s2_rb,
    }


def load_v4d_baseline(device):
    """Load the v4-D single model as baseline."""
    model = TurningSpecialistS2(in_channels=6).to(device)
    ckpt = torch.load(WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


class SensorBasedRouter:
    """Old 5-way heuristic router for backward compatibility analysis."""

    def compute_regime(self, v_prev, a_fwd, a_lat, w_yaw):
        scores = np.zeros(5, dtype=np.float32)
        if v_prev >= 18.0 and abs(w_yaw) < 0.025 and abs(a_lat) < 1.0 and a_fwd > -1.0:
            scores[0] = 5.0 + (v_prev - 18.0) * 0.2
        elif v_prev >= 15.0 and abs(w_yaw) < 0.030 and abs(a_lat) < 1.2:
            scores[0] = 2.0

        if a_fwd <= -1.2:
            scores[3] = 4.0 + abs(a_fwd) * 1.5
        elif a_fwd <= -0.7:
            scores[3] = 2.0 + abs(a_fwd) * 1.0

        if a_fwd >= 1.2 and v_prev < 24.0:
            scores[2] = 4.0 + a_fwd * 1.5
        elif a_fwd >= 0.7 and v_prev < 24.0:
            scores[2] = 2.0 + a_fwd * 1.0

        if abs(a_lat) >= 1.8 and 4.0 <= v_prev <= 18.0:
            scores[1] = 4.0 + abs(a_lat) * 1.5
        elif abs(a_lat) >= 1.2 and 4.0 <= v_prev <= 18.0:
            scores[1] = 2.0 + abs(a_lat) * 1.0

        turn_ind = abs(w_yaw)
        if turn_ind >= 0.04 and scores[1] < 3.0:
            scores[4] = 3.5 + turn_ind * 20.0

        if np.max(scores) < 0.5:
            if v_prev >= 16.0:
                scores[0] = 2.0
            else:
                scores[4] = 1.0

        exp_s = np.exp(scores - np.max(scores))
        probs = exp_s / np.sum(exp_s)
        regime_names = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]
        dominant = int(np.argmax(probs))
        return regime_names[dominant], float(probs[dominant])


def build_features(j, cur, history_x):
    """Build the 6-channel input window."""
    a_fwd = j["a_fwd"]; w_yaw = j["w_yaw"]; a_lat = j["a_lat"]
    w_accel = j["w_yaw_accel"]

    ch_a_fwd = np.clip(a_fwd[cur-9:cur+1], -8.0, 8.0)
    ch_w_yaw = np.clip(w_yaw[cur-9:cur+1], -1.0, 1.0)
    ch_a_lat = np.clip(a_lat[cur-9:cur+1], -8.0, 8.0)
    ch_v_prev = np.clip(np.array(history_x[-10:]), 0.0, 45.0)
    ch_w_accel = np.clip(w_accel[cur-9:cur+1], -2.0, 2.0)
    ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)

    win_6 = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
    return win_6, ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev


def apply_postprocessing(scen, xr, wr, j, cur, history_x, entry_decel):
    """Apply the hand-tuned kinematic post-processing for a given scenario."""
    a_fwd = j["a_fwd"]; w_yaw = j["w_yaw"]; a_lat = j["a_lat"]

    if scen == "motorway":
        pass
    elif scen == "hard_brake":
        wr *= 0.05
        if a_fwd[cur] < -0.2:
            xr = min(xr, max(0.0, history_x[-1] + a_fwd[cur] * 0.40))
        if entry_decel < -0.6 and history_x[-1] < 15.0:
            xr = max(0.0, min(xr, history_x[-1] + entry_decel * 1.00))
        if xr < 0.2 and a_fwd[cur] < 0.2:
            xr, wr = 0.0, 0.0
    elif scen == "quick_accel":
        if a_fwd[cur] > 0.20:
            xr = max(xr, history_x[-1] + a_fwd[cur] * 0.75)
        elif a_fwd[cur] < -0.30:
            xr = min(xr, history_x[-1] + a_fwd[cur] * 0.45)
    elif scen == "sharp_turns":
        wr *= 0.08
        v_est = max(history_x[-1], 2.5)
        if abs(a_lat[cur]) > 1.8:
            w_cent = np.sign(w_yaw[cur]) * (abs(a_lat[cur]) / v_est) * 0.70
            wr += w_cent
            xr = min(xr, max(1.5, history_x[-1] - 1.20))
        stopping = (entry_decel < -0.5 and 0.8 < history_x[-1] < 7.5)
        if stopping:
            xr = max(0.0, min(xr, history_x[-1] + entry_decel * 0.80))
    elif scen == "roundabout":
        wr *= 4.00
        v_est = max(history_x[-1], 2.5)
        if abs(a_lat[cur]) > 0.5:
            w_cent = -np.sign(a_lat[cur]) * abs(a_lat[cur]) / v_est * 0.90
            wr = 0.40 * wr + 0.60 * w_cent
        elif abs(j["a_fwd"][cur]) > 1.8 and abs(a_lat[cur]) <= 0.5:
            w_cent = -(j["a_fwd"][cur] / v_est) * 0.85
            wr = w_cent
            if j["a_fwd"][cur] > 0:
                xr = min(xr, history_x[-1] + 0.05)

    return xr, wr


def simulate_outage_oracle(j, start_idx, model, scalers, device, is_4ch, postproc_scen=None):
    """Oracle outage simulation using ground truth scenario name."""
    s_X, s_yd, s_yo = scalers["X"], scalers["y_disp"], scalers["y_ori"]
    x_gps = j["x_gps"]; w_gps = j["w_gps"]; headings = j["headings"]

    history_x = list(x_gps[start_idx - WINDOW: start_idx])
    pos_gt, pos_pred = [(0.0, 0.0)], [(0.0, 0.0)]
    psi_gt = np.radians(headings[start_idx])
    psi_pred = psi_gt
    entry_decel = (history_x[-1] - history_x[-5]) / 4.0

    for k in range(OUTAGE):
        cur = start_idx + k
        win_6, ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev = build_features(j, cur, history_x)
        win_s6 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

        with torch.no_grad():
            if is_4ch:
                t_x = torch.tensor(win_s6[:, :, :4], dtype=torch.float32, device=device)
            else:
                t_x = torch.tensor(win_s6, dtype=torch.float32, device=device)
            d_p, o_p, _ = model(t_x)

        xr = float(s_yd.inverse_transform([[d_p.item()]])[0, 0])
        wr = float(s_yo.inverse_transform([[o_p.item()]])[0, 0])

        if postproc_scen:
            xr, wr = apply_postprocessing(postproc_scen, xr, wr, j, cur, history_x, entry_decel)

        history_x.append(xr)
        psi_gt += w_gps[cur]
        psi_pred += wr
        pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt),
                       pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
        pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred),
                         pos_pred[-1][1] + xr * np.sin(psi_pred)))

    return np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])


def simulate_outage_3regime(j, start_idx, oracle_models, three_router, scalers, device, apply_brake_physics=True):
    """3-Regime Router simulation using live kinematics only."""
    s_X, s_yd, s_yo = scalers["X"], scalers["y_disp"], scalers["y_ori"]
    x_gps = j["x_gps"]; w_gps = j["w_gps"]; headings = j["headings"]

    history_x = list(x_gps[start_idx - WINDOW: start_idx])
    pos_gt, pos_pred = [(0.0, 0.0)], [(0.0, 0.0)]
    psi_gt = np.radians(headings[start_idx])
    psi_pred = psi_gt

    model_map = {
        "motorway": (oracle_models["motorway"], True),
        "hard_brake": (oracle_models["hard_brake"], True),
        "quick_accel": (oracle_models["quick_accel"], False),
        "sharp_turns": (oracle_models["sharp_turns"], False),
        "roundabout": (oracle_models["roundabout"], False),
    }

    for k in range(OUTAGE):
        cur = start_idx + k
        win_6, ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev = build_features(j, cur, history_x)
        win_s6 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

        v_prev = float(ch_v_prev[-1])
        af = float(ch_a_fwd[-1])
        al = float(ch_a_lat[-1])
        wy = float(ch_w_yaw[-1])

        regime = three_router.classify(v_prev, af, al, wy)
        model, is_4ch = model_map[regime]

        with torch.no_grad():
            if is_4ch:
                t_x = torch.tensor(win_s6[:, :, :4], dtype=torch.float32, device=device)
            else:
                t_x = torch.tensor(win_s6, dtype=torch.float32, device=device)
            d_p, o_p, _ = model(t_x)

        xr = float(s_yd.inverse_transform([[d_p.item()]])[0, 0])
        wr = float(s_yo.inverse_transform([[o_p.item()]])[0, 0])

        if apply_brake_physics and af < -0.5:
            phys_speed = max(0.0, history_x[-1] + af * 1.0)
            xr = min(xr, phys_speed)
            if xr < 0.15:
                xr, wr = 0.0, 0.0

        history_x.append(xr)
        psi_gt += w_gps[cur]
        psi_pred += wr
        pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt),
                       pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
        pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred),
                         pos_pred[-1][1] + xr * np.sin(psi_pred)))

    return np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])


def simulate_outage_single_v4d(j, start_idx, model, scalers, device):
    """v4-D Ablation-D single model baseline."""
    s_X, s_yd, s_yo = scalers["X"], scalers["y_disp"], scalers["y_ori"]
    x_gps = j["x_gps"]; w_gps = j["w_gps"]; headings = j["headings"]

    history_x = list(x_gps[start_idx - WINDOW: start_idx])
    pos_gt, pos_pred = [(0.0, 0.0)], [(0.0, 0.0)]
    psi_gt = np.radians(headings[start_idx])
    psi_pred = psi_gt

    for k in range(OUTAGE):
        cur = start_idx + k
        win_6, ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev = build_features(j, cur, history_x)
        win_s6 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

        with torch.no_grad():
            t_x = torch.tensor(win_s6, dtype=torch.float32, device=device)
            d_p, o_p, _ = model(t_x)

        xr = float(s_yd.inverse_transform([[d_p.item()]])[0, 0])
        wr = float(s_yo.inverse_transform([[o_p.item()]])[0, 0])

        history_x.append(xr)
        psi_gt += w_gps[cur]
        psi_pred += wr
        pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt),
                       pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
        pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred),
                         pos_pred[-1][1] + xr * np.sin(psi_pred)))

    return np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])


def run_honest_evaluation(tau_yaw=0.04, tau_accel=1.0, tau_speed=16.0):
    """Run comprehensive honest evaluation across all evaluation paradigms."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(DATA_DIR / "scalers_v7.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    oracle_models = load_oracle_models(device)
    v4d_model = load_v4d_baseline(device)
    three_router = ThreeRegimeRouter(tau_yaw=tau_yaw, tau_accel=tau_accel, tau_speed=tau_speed)

    oracle_is_4ch = {
        "motorway": True,
        "hard_brake": True,
        "quick_accel": False,
        "sharp_turns": False,
        "roundabout": False,
    }

    # Load cross-validation result if available
    cv_file = RESULTS_DIR / "cross_validation_results.json"
    cv_info = None
    if cv_file.exists():
        with open(cv_file, "r") as f:
            cv_info = json.load(f)
        gb = cv_info.get("global_best", {})
        three_router = ThreeRegimeRouter(
            tau_yaw=gb.get("tau_yaw", tau_yaw),
            tau_accel=gb.get("tau_accel", tau_accel),
            tau_speed=gb.get("tau_speed", tau_speed),
        )

    print("\n" + "=" * 125)
    print("COMPREHENSIVE HONEST EVALUATION: v7 vs v4-D vs ORACLE")
    print(f"3-Regime Router Params: tau_yaw={three_router.tau_yaw}, tau_accel={three_router.tau_accel}, tau_speed={three_router.tau_speed}")
    print("=" * 125)
    print(f"{'Scenario':14s} | {'3-Regime Router':>16s} | {'v4-D Single Model':>18s} | {'Oracle Reported':>16s} | {'Previous Best':>14s}")
    print(f"{'':14s} | {'(Honest Deployment)':>16s} | {'(Turn-Focused Base)':>18s} | {'(21.47m Method)':>16s} | {'(Repo Baseline)':>14s}")
    print("-" * 125)

    all_3regime, all_v4d, all_oracle = [], [], []
    results = {"three_regime_router": {}, "v4d_baseline": {}, "oracle_reported": {}}

    for scen in SCENARIOS:
        journeys = test_scenarios[scen]
        scen_3r, scen_v4d, scen_ora = [], [], []

        for j in journeys:
            x_gps = j["x_gps"]
            for s in range(WINDOW + 1, len(x_gps) - OUTAGE, 10):
                d_3r = simulate_outage_3regime(j, s, oracle_models, three_router, scalers, device, apply_brake_physics=True)
                d_v4 = simulate_outage_single_v4d(j, s, v4d_model, scalers, device)
                d_or = simulate_outage_oracle(j, s, oracle_models[scen], scalers, device,
                                              is_4ch=oracle_is_4ch[scen], postproc_scen=scen)

                scen_3r.append(d_3r)
                scen_v4d.append(d_v4)
                scen_ora.append(d_or)

        m_3r = float(np.mean(scen_3r))
        m_v4 = float(np.mean(scen_v4d))
        m_or = float(np.mean(scen_ora))
        best = BEST_REPO[scen]

        all_3regime.extend(scen_3r)
        all_v4d.extend(scen_v4d)
        all_oracle.extend(scen_ora)

        results["three_regime_router"][scen] = {"drift_10s_m": m_3r, "num_outages": len(scen_3r)}
        results["v4d_baseline"][scen] = {"drift_10s_m": m_v4, "num_outages": len(scen_v4d)}
        results["oracle_reported"][scen] = {"drift_10s_m": m_or, "num_outages": len(scen_ora), "best_repo_m": best}

        print(f"{scen:14s} | {m_3r:14.2f}m | {m_v4:16.2f}m | {m_or:14.2f}m | {best:12.2f}m")

    print("-" * 125)
    om_3r = float(np.mean(all_3regime))
    om_v4 = float(np.mean(all_v4d))
    om_or = float(np.mean(all_oracle))
    best_overall = 29.55

    print(f"{'OVERALL':14s} | {om_3r:14.2f}m | {om_v4:16.2f}m | {om_or:14.2f}m | {best_overall:12.2f}m")
    print("=" * 125)

    results["overall"] = {
        "three_regime_drift_10s_m": om_3r,
        "v4d_baseline_drift_10s_m": om_v4,
        "oracle_reported_drift_10s_m": om_or,
        "previous_best_repo_m": best_overall,
        "total_outages": len(all_3regime),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "honest_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Honest Eval] Results saved -> {out_file}")

    return results


if __name__ == "__main__":
    run_honest_evaluation()
