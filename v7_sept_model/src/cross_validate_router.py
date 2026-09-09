"""
cross_validate_router.py
------------------------
Leave-One-Journey-Out Cross-Validation for the 3-Regime Router.

Collapses 5 overlapping scenario categories into 3 physically separable regimes:
  1. CRUISING:              High speed, low dynamics -> Motorway specialist
  2. TURNING:               High |w_yaw| or |a_lat|  -> Sharp Turns / Roundabout specialist
  3. LONGITUDINAL TRANSIENT: |a_fwd| spike           -> Hard Brake (a_fwd < 0) / Quick Accel (a_fwd > 0)

Tunes 3 thresholds (tau_yaw, tau_accel, tau_speed) by scoring DRIFT (not classification accuracy)
across 9 leave-one-journey-out folds. Reports honest cross-validated performance.

Also evaluates v4-D single-model baseline for direct head-to-head comparison.
"""

from __future__ import annotations

import time
import json
import pickle
import itertools
from pathlib import Path
from collections import defaultdict
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

OUTAGE = 10
WINDOW = 10


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_specialist_models(device: torch.device):
    """Load specialist models for each regime."""
    # Motorway: v3 S1
    s1_mot = StraightSpecialistS1().to(device)
    ckpt_v3 = torch.load(WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth",
                         map_location=device, weights_only=False)
    s1_mot.load_state_dict(ckpt_v3["model_state_dict"])
    s1_mot.eval()

    # Quick Accel + Sharp Turns + Hard Brake: v4-D turn-focused backbone
    s2_v4d = TurningSpecialistS2(in_channels=6).to(device)
    ckpt_v4 = torch.load(WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth",
                         map_location=device, weights_only=False)
    s2_v4d.load_state_dict(ckpt_v4["model_state_dict"])
    s2_v4d.eval()

    # Roundabout: fine-tuned specialist
    s2_rb = TurningSpecialistS2(in_channels=6).to(device)
    ckpt_rb = torch.load(CKPT_DIR / "best_supreme_roundabout.pth", map_location=device, weights_only=False)
    s2_rb.load_state_dict(ckpt_rb["model_state_dict"])
    s2_rb.eval()

    # Model lookup: name -> (model, is_4ch)
    # Crucial: hard_brake during dynamic driving must retain turning awareness (6ch v4-D)
    return {
        "motorway":    (s1_mot, True),
        "hard_brake":  (s2_v4d, False),
        "quick_accel": (s2_v4d, False),
        "sharp_turns": (s2_v4d, False),
        "roundabout":  (s2_rb,  False),
    }


def load_v4d_baseline(device: torch.device):
    """Load the v4-D single model as baseline."""
    model = TurningSpecialistS2(in_channels=6).to(device)
    ckpt = torch.load(WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ─── Simulation Engine ────────────────────────────────────────────────────────

def build_features(j, cur, history_x):
    """Build 6-channel input window for a single timestep."""
    a_fwd = j["a_fwd"]
    w_yaw = j["w_yaw"]
    a_lat = j["a_lat"]
    w_accel = j["w_yaw_accel"]

    ch_a_fwd = np.clip(a_fwd[cur-9:cur+1], -8.0, 8.0)
    ch_w_yaw = np.clip(w_yaw[cur-9:cur+1], -1.0, 1.0)
    ch_a_lat = np.clip(a_lat[cur-9:cur+1], -8.0, 8.0)
    ch_v_prev = np.clip(np.array(history_x[-10:]), 0.0, 45.0)
    ch_w_accel = np.clip(w_accel[cur-9:cur+1], -2.0, 2.0)
    ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)

    win_6 = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
    return win_6, float(ch_v_prev[-1]), float(ch_a_fwd[-1]), float(ch_a_lat[-1]), float(ch_w_yaw[-1])


def simulate_outage_routed(j, start_idx, router, specialists, scalers, device, apply_brake_physics=True):
    """Run 10-step outage with 3-regime router selecting specialist per step."""
    s_X, s_yd, s_yo = scalers["X"], scalers["y_disp"], scalers["y_ori"]
    x_gps = j["x_gps"]
    w_gps = j["w_gps"]
    headings = j["headings"]

    history_x = list(x_gps[start_idx - WINDOW: start_idx])
    pos_gt, pos_pred = [(0.0, 0.0)], [(0.0, 0.0)]
    psi_gt = np.radians(headings[start_idx])
    psi_pred = psi_gt

    for k in range(OUTAGE):
        cur = start_idx + k
        win_6, v_prev, af, al, wy = build_features(j, cur, history_x)
        win_s6 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

        regime = router.classify(v_prev, af, al, wy)
        model, is_4ch = specialists[regime]

        with torch.no_grad():
            if is_4ch:
                t_x = torch.tensor(win_s6[:, :, :4], dtype=torch.float32, device=device)
            else:
                t_x = torch.tensor(win_s6, dtype=torch.float32, device=device)
            d_p, o_p, _ = model(t_x)

        xr = float(s_yd.inverse_transform([[d_p.item()]])[0, 0])
        wr = float(s_yo.inverse_transform([[o_p.item()]])[0, 0])

        # Minimal physics: brake deceleration integration
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


def simulate_outage_single_model(j, start_idx, model, scalers, device, is_4ch=False):
    """Run 10-step outage with a single model (v4-D baseline)."""
    s_X, s_yd, s_yo = scalers["X"], scalers["y_disp"], scalers["y_ori"]
    x_gps = j["x_gps"]
    w_gps = j["w_gps"]
    headings = j["headings"]

    history_x = list(x_gps[start_idx - WINDOW: start_idx])
    pos_gt, pos_pred = [(0.0, 0.0)], [(0.0, 0.0)]
    psi_gt = np.radians(headings[start_idx])
    psi_pred = psi_gt

    for k in range(OUTAGE):
        cur = start_idx + k
        win_6, _, _, _, _ = build_features(j, cur, history_x)
        win_s6 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

        with torch.no_grad():
            if is_4ch:
                t_x = torch.tensor(win_s6[:, :, :4], dtype=torch.float32, device=device)
            else:
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


def get_all_journeys(test_scenarios):
    """Flatten all journeys into a list of (scenario, journey_idx, journey_dict)."""
    all_j = []
    for scen in ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]:
        for ji, j in enumerate(test_scenarios[scen]):
            all_j.append((scen, ji, j))
    return all_j


# ─── Cross-Validation Engine ─────────────────────────────────────────────────

def run_cross_validation():
    """Leave-One-Journey-Out Cross-Validation for 3-Regime Router thresholds."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    with open(DATA_DIR / "scalers_v7.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    specialists = load_specialist_models(device)
    v4d_model = load_v4d_baseline(device)
    all_journeys = get_all_journeys(test_scenarios)

    print(f"\n{'=' * 95}")
    print("LEAVE-ONE-JOURNEY-OUT CROSS-VALIDATION FOR 3-REGIME ROUTER")
    print(f"{'=' * 95}")
    print(f"Total journeys: {len(all_journeys)}")
    for scen, ji, j in all_journeys:
        print(f"  {scen:12s} J{ji}: {j.get('name', '?'):10s} ({len(j['x_gps'])} steps)")

    # ─── Compute v4-D Baseline Once for All Journeys ──────────────────────────
    print("\nEvaluating v4-D single-model baseline across all journeys...")
    v4d_journey_drifts = []
    v4d_scen_drifts = defaultdict(list)
    for j_idx, (scen, ji, j) in enumerate(all_journeys):
        x_gps = j["x_gps"]
        drifts = []
        for s in range(WINDOW + 1, len(x_gps) - OUTAGE, 10):
            d = simulate_outage_single_model(j, s, v4d_model, scalers, device, is_4ch=False)
            drifts.append(d)
            v4d_scen_drifts[scen].append(d)
        v4d_journey_drifts.append(drifts)

    v4d_overall = float(np.mean([d for dl in v4d_journey_drifts for d in dl]))
    print(f"v4-D Single Model Overall 10s Drift: {v4d_overall:.2f} m")

    # ─── Candidate threshold grid ────────────────────────────────────────────
    tau_yaw_candidates   = [0.03, 0.05, 0.08, 0.12]
    tau_accel_candidates = [1.2, 1.5, 1.8, 2.2]
    tau_speed_candidates = [14.0, 16.0, 18.0, 20.0]

    all_combos = list(itertools.product(tau_yaw_candidates, tau_accel_candidates, tau_speed_candidates))
    n_combos = len(all_combos)
    print(f"\nEvaluating {n_combos} candidate threshold combinations across all {len(all_journeys)} journeys...")
    t0 = time.time()

    # Pre-evaluate each combination on each journey: combo_journey_drifts[combo_idx][journey_idx] = [outage_drifts]
    combo_journey_drifts = []
    for c_idx, (ty, ta, ts) in enumerate(all_combos):
        router = ThreeRegimeRouter(tau_yaw=ty, tau_accel=ta, tau_speed=ts)
        j_drifts_for_c = []
        for scen, ji, j in all_journeys:
            x_gps = j["x_gps"]
            d_list = []
            for s in range(WINDOW + 1, len(x_gps) - OUTAGE, 10):
                d = simulate_outage_routed(j, s, router, specialists, scalers, device, apply_brake_physics=True)
                d_list.append(d)
            j_drifts_for_c.append(d_list)
        combo_journey_drifts.append(j_drifts_for_c)

        if (c_idx + 1) % 16 == 0 or (c_idx + 1) == n_combos:
            elapsed = time.time() - t0
            print(f"  [Progress] {c_idx + 1}/{n_combos} combos evaluated ({elapsed:.1f}s elapsed)...")

    print(f"Pre-evaluation completed in {time.time() - t0:.2f}s.")

    # ─── Leave-One-Journey-Out Evaluation ─────────────────────────────────────
    print(f"\n{'=' * 95}")
    print("RUNNING 9-FOLD LEAVE-ONE-JOURNEY-OUT CROSS-VALIDATION")
    print(f"{'=' * 95}")

    fold_results = []
    n_journeys = len(all_journeys)

    for fold_idx in range(n_journeys):
        ho_scen, ho_ji, ho_j = all_journeys[fold_idx]
        ho_name = ho_j.get("name", f"{ho_scen}_J{ho_ji}")

        # Train set = all journeys except fold_idx
        # Select best threshold combination based strictly on train drift
        best_train_drift = 999.0
        best_c_idx = 0

        for c_idx in range(n_combos):
            train_drifts = []
            for j_idx in range(n_journeys):
                if j_idx != fold_idx:
                    train_drifts.extend(combo_journey_drifts[c_idx][j_idx])
            mean_train_drift = float(np.mean(train_drifts)) if train_drifts else 999.0
            if mean_train_drift < best_train_drift:
                best_train_drift = mean_train_drift
                best_c_idx = c_idx

        best_params = all_combos[best_c_idx]
        # Evaluate selected parameters on the held-out journey
        ho_drifts = combo_journey_drifts[best_c_idx][fold_idx]
        if len(ho_drifts) == 0:
            ho_drift = float('nan')
            ho_v4d_drift = float('nan')
        else:
            ho_drift = float(np.mean(ho_drifts))
            ho_v4d_drift = float(np.mean(v4d_journey_drifts[fold_idx]))

        fold_results.append({
            "fold": fold_idx + 1,
            "held_out_name": ho_name,
            "held_out_scenario": ho_scen,
            "best_params": {
                "tau_yaw": best_params[0],
                "tau_accel": best_params[1],
                "tau_speed": best_params[2],
            },
            "train_drift_m": best_train_drift,
            "held_out_drift_m": ho_drift,
            "v4d_drift_m": ho_v4d_drift,
            "n_outages": len(ho_drifts),
        })

    # ─── Summary Table ────────────────────────────────────────────────────────
    print(f"\n{'Fold':>4s} | {'Held Out':>10s} | {'Scenario':>12s} | {'Selected Params':>24s} | {'3-Regime':>10s} | {'v4-D':>10s} | {'Winner':>8s}")
    print("-" * 95)
    for r in fold_results:
        if np.isnan(r["held_out_drift_m"]):
            winner = "N/A (0 outages)"
            r_str = "    N/A"
            v_str = "    N/A"
        else:
            winner = "Router" if r["held_out_drift_m"] < r["v4d_drift_m"] else "v4-D"
            r_str = f"{r['held_out_drift_m']:8.2f}m"
            v_str = f"{r['v4d_drift_m']:8.2f}m"
        p_str = f"yaw={r['best_params']['tau_yaw']}, acc={r['best_params']['tau_accel']}, spd={r['best_params']['tau_speed']}"
        print(f"  {r['fold']:2d} | {r['held_out_name']:>10s} | {r['held_out_scenario']:>12s} | {p_str:>24s} | "
              f"{r_str} | {v_str} | {winner:>8s}")
    print("-" * 95)

    valid_folds = [r for r in fold_results if not np.isnan(r["held_out_drift_m"])]
    cv_router_drifts = [r["held_out_drift_m"] for r in valid_folds]
    cv_v4d_drifts = [r["v4d_drift_m"] for r in valid_folds]

    mean_cv_router = float(np.mean(cv_router_drifts))
    std_cv_router = float(np.std(cv_router_drifts))
    mean_cv_v4d = float(np.mean(cv_v4d_drifts))
    std_cv_v4d = float(np.std(cv_v4d_drifts))

    total_outages = sum(r["n_outages"] for r in valid_folds)
    weighted_cv_router = float(sum(r["held_out_drift_m"] * r["n_outages"] for r in valid_folds) / total_outages)
    weighted_cv_v4d = float(sum(r["v4d_drift_m"] * r["n_outages"] for r in valid_folds) / total_outages)

    print(f"  {'MEAN (Fold Avg)':>18s} | {'':>12s} | {'':>24s} | {mean_cv_router:8.2f}m | {mean_cv_v4d:8.2f}m |")
    print(f"  {'STD (Fold Avg)':>18s} | {'':>12s} | {'':>24s} | {std_cv_router:8.2f}m | {std_cv_v4d:8.2f}m |")
    print(f"  {'OUTAGE WEIGHTED':>18s} | {'':>12s} | {'':>24s} | {weighted_cv_router:8.2f}m | {weighted_cv_v4d:8.2f}m |")
    print(f"{'=' * 95}")

    # ─── Global Best Parameter Search (for Deployment) ────────────────────────
    best_global_drift = 999.0
    best_global_c_idx = 0

    for c_idx in range(n_combos):
        all_drifts = [d for j_dl in combo_journey_drifts[c_idx] for d in j_dl]
        drift = float(np.mean(all_drifts))
        if drift < best_global_drift:
            best_global_drift = drift
            best_global_c_idx = c_idx

    best_global_params = all_combos[best_global_c_idx]
    print(f"\n[Deployment] Best Global Thresholds: tau_yaw={best_global_params[0]}, tau_accel={best_global_params[1]}, tau_speed={best_global_params[2]}")
    print(f"             In-Sample Global Mean Drift: {best_global_drift:.2f}m")
    print(f"             Honest CV Mean Drift:        {weighted_cv_router:.2f}m (Outage-weighted)")

    # Per-scenario breakdown under best global parameters
    print(f"\nPer-Scenario Performance Breakdown (Best Global Router vs v4-D):")
    print(f"{'Scenario':14s} | {'3-Regime Router':>16s} | {'v4-D Baseline':>14s} | {'Difference':>12s}")
    print("-" * 62)

    router_scen_drifts = defaultdict(list)
    for j_idx, (scen, ji, j) in enumerate(all_journeys):
        router_scen_drifts[scen].extend(combo_journey_drifts[best_global_c_idx][j_idx])

    for scen in ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]:
        r_m = float(np.mean(router_scen_drifts[scen]))
        v_m = float(np.mean(v4d_scen_drifts[scen]))
        diff = r_m - v_m
        diff_str = f"{diff:+.2f}m"
        print(f"  {scen:12s} | {r_m:14.2f}m | {v_m:12.2f}m | {diff_str:>12s}")

    print("-" * 62)
    print(f"  {'OVERALL':12s} | {best_global_drift:14.2f}m | {v4d_overall:12.2f}m | {best_global_drift - v4d_overall:+.2f}m")
    print("=" * 62)

    # ─── Save Results ─────────────────────────────────────────────────────────
    final_results = {
        "cross_validation": {
            "n_folds": len(fold_results),
            "folds": fold_results,
            "mean_cv_router_drift_m": mean_cv_router,
            "std_cv_router_drift_m": std_cv_router,
            "weighted_cv_router_drift_m": weighted_cv_router,
            "mean_cv_v4d_drift_m": mean_cv_v4d,
            "std_cv_v4d_drift_m": std_cv_v4d,
            "weighted_cv_v4d_drift_m": weighted_cv_v4d,
        },
        "global_best": {
            "tau_yaw": best_global_params[0],
            "tau_accel": best_global_params[1],
            "tau_speed": best_global_params[2],
            "global_drift_m": best_global_drift,
        },
        "v4d_baseline": {
            "overall_drift_m": v4d_overall,
            "per_scenario": {s: float(np.mean(v4d_scen_drifts[s])) for s in v4d_scen_drifts},
        },
        "router_per_scenario": {s: float(np.mean(router_scen_drifts[s])) for s in router_scen_drifts},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "cross_validation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[CV] Results saved -> {out_file}")

    return final_results


if __name__ == "__main__":
    run_cross_validation()
