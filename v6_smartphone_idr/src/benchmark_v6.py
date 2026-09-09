"""
benchmark_v6.py
---------------
Closed-loop 10-second GNSS-outage drift benchmark for PINO-DR v6.

Methodology (matches the "V6 Neural Motion Model Only" / ablation-B run):
  1. For each test scenario journey, slide a 100-step (10 s) outage window.
  2. Before the outage, ground-truth (GNSS) velocity / heading are available.
     During the outage, ONLY the smartphone IMU is used.
  3. The neural model runs at 10 Hz in CLOSED LOOP: the predicted delta-v is
     integrated to update speed, and the predicted yaw rate updates heading.
     The model's own previous-speed estimate feeds back as the v_prev channel.
  4. Position is integrated by dead reckoning (local ENU) and the final drift
     from the true GNSS position is reported per scenario and overall.

The benchmark is deterministic (no augmentation, no scaling noise) so it is a
fair, reproducible measure of real-world dead-reckoning drift.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

from v6_smartphone_idr.src.models_v6 import PINODeadReckoningNetV6


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TEST_SCENARIOS = {
    "motorway": ["vw12"],
    "roundabout": ["vta11"],
    "quick_accel": ["vta12"],
    "hard_brake": ["vw16b", "vw17", "vta9"],
    "sharp_turns": ["vw6", "vw7", "vw8"],
}

DT = 0.1
OUTAGE_STEPS = 100  # 10 seconds at 10 Hz


# ─── Geodesic helpers ────────────────────────────────────────────────────────

EARTH_R = 6378137.0


def geodesic_dist(lat1, lon1, lat2, lon2):
    """WGS84 geodesic distance (m) using the vectorised formula from preprocess."""
    from v6_smartphone_idr.src.preprocess_v6 import compute_vectorized_geodesic_disp
    # Cheap: use the Vincenty-style vectorized helper on two points.
    d = compute_vectorized_geodesic_disp(
        np.array([lat1, lat2], dtype=np.float64),
        np.array([lon1, lon2], dtype=np.float64),
    )
    return float(d[-1])


def local_en_to_latlon(lat0, lon0, east, north):
    """Convert local east/north displacement (m) to new lat/lon (degrees)."""
    dlat = north / EARTH_R * (180.0 / math.pi)
    dlon = east / (EARTH_R * math.cos(math.radians(lat0))) * (180.0 / math.pi)
    return lat0 + dlat, lon0 + dlon


# ─── Closed-loop sequence ────────────────────────────────────────────────────

def build_sequence_window(j, t, v_est_series, scaler_X, window_size=20):
    """
    Build a (20, 6) window at time index t for closed-loop propagation.

    Channels: [a_fwd, w_yaw, a_lat, v_prev, w_yaw_accel, centripetal_res]
    IMU channels are measured and always available. v_prev uses the estimated
    (closed-loop) speed series. centripetal_res is recomputed from the estimate.
    All values are scaled by scaler_X.
    """
    a_fwd = np.clip(j["a_fwd"][t - window_size + 1:t + 1], -8.0, 8.0)
    w_yaw = np.clip(j["w_yaw"][t - window_size + 1:t + 1], -1.2, 1.2)
    a_lat = np.clip(j["a_lat"][t - window_size + 1:t + 1], -8.0, 8.0)
    w_accel = np.clip(j["w_yaw_accel"][t - window_size + 1:t + 1], -4.0, 4.0)

    # v_prev: estimated previous speeds for indices t-20..t-1
    v_prev = np.array(v_est_series[t - window_size:t], dtype=np.float64)
    v_prev = np.clip(v_prev, 0.0, 45.0)

    # centripetal residual aligned to IMU channels (a_lat - v_prev * w_yaw)
    centripetal = np.clip(a_lat - v_prev * w_yaw, -8.0, 8.0)

    win = np.stack([a_fwd, w_yaw, a_lat, v_prev, w_accel, centripetal], axis=-1)
    win = win.reshape(1, -1)
    win = scaler_X.transform(win).reshape(window_size, 6).astype(np.float32)
    return win


def run_sequence(model, journey, t0, scalers, device,
                 outage_steps=OUTAGE_STEPS):
    """
    Run a closed-loop 10 s outage starting at t0.

    Returns a dict with drift at 1s/3s/5s/10s and the estimated path.
    """
    WINDOW = 20
    j = journey
    n = len(j["v_true"])

    # Estimated speed series: ground-truth before outage, model estimate after.
    v_est = np.zeros(n, dtype=np.float64)
    v_est[:t0] = j["v_true"][:t0]

    # Estimated heading (radians) and position (ENU integrated).
    heading0 = math.radians(j["headings"][t0 - 1])  # last known heading (deg->rad)

    # We accumulate east/north from the known start point.
    lat0 = j["lats"][t0 - 1]
    lon0 = j["lons"][t0 - 1]

    east = 0.0
    north = 0.0

    # Keep the model's own v_prev estimate for the last predicted step.
    last_v = float(j["v_true"][t0 - 1]) if t0 > 0 else 0.0

    drifts = {"drift_1s": None, "drift_3s": None, "drift_5s": None, "drift_10s": None}

    heading_rad = heading0
    for k in range(outage_steps):
        t = t0 + k
        if t >= n:
            break

        win = build_sequence_window(j, t, v_est, scalers["X"], WINDOW)
        x_in = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(x_in)
        dv_scaled, w_scaled = preds[0].cpu().numpy(), preds[1].cpu().numpy()

        dv_phys = float(scalers["y_dv"].inverse_transform(dv_scaled)[0, 0])
        w_phys = float(scalers["y_w"].inverse_transform(w_scaled)[0, 0])

        # Update estimated speed (kinematic residual: v += delta-v)
        last_v = max(0.0, last_v + dv_phys)
        v_est[t] = last_v

        # Update heading
        heading_rad += w_phys * DT

        # Integrate position (local ENU)
        east += last_v * math.sin(heading_rad) * DT
        north += last_v * math.cos(heading_rad) * DT

        # Drift at horizon times
        horizon_steps = {1: 10, 3: 30, 5: 50, 10: 100}
        for name, hsteps in horizon_steps.items():
            if (k + 1) == hsteps and t < n:
                est_lat, est_lon = local_en_to_latlon(lat0, lon0, east, north)
                drifts[f"drift_{name}s"] = geodesic_dist(
                    est_lat, est_lon, j["lats"][t], j["lons"][t]
                )

    return drifts


# ─── Benchmark runner ────────────────────────────────────────────────────────

def run_benchmark(model, journeys_by_scenario, scalers, device, stride=10):
    results = {}
    all_drift_10s = []
    all_sequences = 0

    for scenario, journeys in journeys_by_scenario.items():
        scen_drifts = []
        scen_sequences = 0
        for j in journeys:
            n = len(j["v_true"])
            # Slide 100-step outage windows with a stride.
            starts = list(range(OUTAGE_STEPS - 20, n - OUTAGE_STEPS, stride))
            if not starts:
                starts = [OUTAGE_STEPS - 20]
            for t0 in starts:
                if t0 < 20 or t0 >= n:
                    continue
                drifts = run_sequence(model, j, t0, scalers, device)
                d10 = drifts.get("drift_10s")
                if d10 is not None:
                    scen_drifts.append(d10)
                    all_drift_10s.append(d10)
                    scen_sequences += 1
        results[scenario] = {
            "n_sequences": scen_sequences,
            "drift_10s_mean": float(np.mean(scen_drifts)) if scen_drifts else None,
            "drift_10s_std": float(np.std(scen_drifts)) if scen_drifts else None,
        }
        all_sequences += scen_sequences

    overall = float(np.mean(all_drift_10s)) if all_drift_10s else None
    return results, overall, all_sequences


def main():
    parser = argparse.ArgumentParser(description="Closed-loop v6 drift benchmark")
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "best_model_v6.pth"))
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--out", type=str, default=str(ROOT / "results" / "benchmark_summary_v6_closedloop.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA_DIR / "scalers_v6.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v6.pkl", "rb") as f:
        test_scen = pickle.load(f)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = PINODeadReckoningNetV6(
        in_channels=cfg.get("in_channels", 6),
        conv_channels=cfg.get("conv_channels", 32),
        gru_hidden=cfg.get("gru_hidden", 32),
        dropout=cfg.get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[v6 Bench] Loaded checkpoint: {args.ckpt}")

    # Map test scenarios to journeys by name.
    journeys_by_scenario = {}
    for scenario, tags in TEST_SCENARIOS.items():
        tag_set = set(t.lower().replace("-", "").replace("_", "") for t in tags)
        jlist = []
        for j in test_scen.get(scenario, []):
            if j["name"].lower() in tag_set or j["name"].lower() in tag_set:
                jlist.append(j)
        journeys_by_scenario[scenario] = jlist
        print(f"[v6 Bench] {scenario}: {[j['name'] for j in jlist]}")

    results, overall, n_seq = run_benchmark(model, journeys_by_scenario, scalers, device, args.stride)

    summary = {
        "mode": "closed-loop neural-only (ablation B style)",
        "total_sequences": n_seq,
        "overall_drift_10s": overall,
        "scenarios": results,
    }
    print("\n=== CLOSED-LOOP DRIFT (10 s outage) ===")
    print(f"Overall 10s drift: {overall:.3f} m over {n_seq} sequences")
    for scen, r in results.items():
        print(f"  {scen:<14} {r['drift_10s_mean']:8.3f} m  (n={r['n_sequences']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
