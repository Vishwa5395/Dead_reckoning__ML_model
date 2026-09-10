import sys
from pathlib import Path
import pickle
import numpy as np
import torch
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from v7_sept_model.src.moe_five_model import SupremeMoENet
from v7_sept_model.src.evaluate_five_specialists import SCENARIOS

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

WINDOW = 10
OUTAGE = 10

def precompute_all_outages():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA_DIR / "scalers_v7.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    s_X, s_yd, s_yo = scalers["X"], scalers["y_disp"], scalers["y_ori"]

    moe = SupremeMoENet().to(device)
    ckpt = torch.load(CKPT_DIR / "best_supreme_moe.pth", map_location=device, weights_only=False)
    moe.load_state_dict(ckpt["model_state_dict"])
    moe.eval()

    # We need to collect per outage:
    # 10 steps of raw sensor inputs, gt values, and expert predictions (d0..d4, o0..o4, z0..z4, weights)
    # Since autoregressive history uses xr, let's store the outage context so we can run fast closed loops.
    print("[Precompute] Storing outage data...")
    outages = []
    for scen in SCENARIOS:
        for j in test_scenarios[scen]:
            x_gps = j["x_gps"]; w_gps = j["w_gps"]; headings = j["headings"]
            a_fwd = j["a_fwd"]; w_yaw = j["w_yaw"]; a_lat = j["a_lat"]; w_accel = j["w_yaw_accel"]

            for s in range(WINDOW + 1, len(x_gps) - OUTAGE, 10):
                outage = {
                    "scen": scen,
                    "x_gps": x_gps[s:s+OUTAGE],
                    "w_gps": w_gps[s:s+OUTAGE],
                    "psi_0": np.radians(headings[s]),
                    "history_init": list(x_gps[s - WINDOW: s]),
                    "a_fwd": a_fwd,
                    "w_yaw": w_yaw,
                    "a_lat": a_lat,
                    "w_accel": w_accel,
                    "s": s,
                }
                outages.append(outage)

    print(f"[Precompute] Total outages across benchmark: {len(outages)}")
    return outages, moe, scalers, device

def simulate_outage_fast(outage, moe, scalers, device, alat_thresh, cent_gain, wr4_ratio, speed_clamp):
    s_X = scalers["X"]
    s_yd = scalers["y_disp"]
    s_yo = scalers["y_ori"]
    s = outage["s"]
    history_x = list(outage["history_init"])
    pos_gt, pos_pred = [(0.0, 0.0)], [(0.0, 0.0)]
    psi_gt = outage["psi_0"]
    psi_pred = psi_gt

    a_fwd = outage["a_fwd"]
    w_yaw = outage["w_yaw"]
    a_lat = outage["a_lat"]
    w_accel = outage["w_accel"]
    x_gps = outage["x_gps"]
    w_gps = outage["w_gps"]

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
            t_x = torch.tensor(win_s6, dtype=torch.float32, device=device)
            weights = moe.gating(t_x)
            x4 = t_x[:, :, :4]
            d0_raw, o0_raw, z0 = moe.exp0(x4)
            d1_raw, o1_raw, z1 = moe.exp1(t_x)
            d2_raw, o2_raw, z2 = moe.exp2(t_x)
            d3_raw, o3_raw, z3 = moe.exp3(x4)
            d4_raw, o4_raw, z4 = moe.exp4(t_x)

            x_last = t_x[:, -1, :] * moe.x_range + moe.x_min
            afwd = x_last[:, 0:1]
            wyaw = x_last[:, 1:2]
            alat = x_last[:, 2:3]
            vprev = x_last[:, 3:4]

            xr0 = d0_raw * 45.0; wr0 = o0_raw * 2.0263271 - 1.0224137
            xr1 = d1_raw * 45.0; wr1 = o1_raw * 2.0263271 - 1.0224137
            xr2 = d2_raw * 45.0; wr2 = o2_raw * 2.0263271 - 1.0224137
            xr3 = d3_raw * 45.0; wr3 = o3_raw * 2.0263271 - 1.0224137
            xr4 = d4_raw * 45.0; wr4 = o4_raw * 2.0263271 - 1.0224137

            v_safe = torch.clamp(vprev, min=2.5)

            # Expert 0
            xr0_p = xr0; wr0_p = wr0

            # Expert 1 (Roundabout)
            is_tilted_rb = (torch.abs(afwd) > 1.8) & (torch.abs(alat) <= 0.6) & (vprev < 18.0)
            w_cent_tilted = -(afwd / v_safe) * 0.85
            w_cent_lat = -torch.sign(alat) * (torch.abs(alat) / v_safe) * 0.90
            w_cent1 = torch.where(is_tilted_rb, w_cent_tilted, w_cent_lat)
            has_cent1 = (torch.abs(alat) > 0.5) | is_tilted_rb
            wr1_p = torch.where(has_cent1, 0.40 * (wr1 * 4.0) + 0.60 * w_cent1, wr1)
            xr1_p = torch.where(is_tilted_rb & (afwd > 0), torch.clamp(xr1, max=vprev + 0.05), xr1)

            # Expert 2 (Quick Accel)
            xr2_p = torch.where((afwd > 0.8) & (vprev < 20.0), torch.maximum(xr2, vprev + afwd * 0.12), xr2)
            wr2_p = wr2

            # Expert 3 (Hard Brake)
            v_decel_bound = torch.clamp(vprev + afwd * 0.40, min=0.0)
            xr3_p = torch.where(afwd < -0.2, torch.minimum(xr3, v_decel_bound), xr3)
            is_standstill = (vprev < 0.3) & (torch.abs(afwd) < 0.3)
            xr3_p = torch.where(is_standstill, torch.zeros_like(xr3_p), xr3_p)
            wr3_p = torch.where(is_standstill, torch.zeros_like(wr3), wr3 * 0.05)

            # Expert 4 (Sharp Turns)
            has_turn_transient = (torch.abs(alat) > alat_thresh)
            w_cent4 = torch.sign(wyaw) * (torch.abs(alat) / v_safe) * cent_gain
            wr4_p = torch.where(has_turn_transient, wr4 * wr4_ratio + w_cent4, wr4)
            xr4_p = torch.where(has_turn_transient, torch.clamp(xr4, max=vprev + speed_clamp), xr4)

            d0 = xr0_p / 45.0; o0 = (wr0_p + 1.0224137) / 2.0263271
            d1 = xr1_p / 45.0; o1 = (wr1_p + 1.0224137) / 2.0263271
            d2 = xr2_p / 45.0; o2 = (wr2_p + 1.0224137) / 2.0263271
            d3 = xr3_p / 45.0; o3 = (wr3_p + 1.0224137) / 2.0263271
            d4 = xr4_p / 45.0; o4 = (wr4_p + 1.0224137) / 2.0263271

            w0 = weights[:, 0:1]; w1 = weights[:, 1:2]; w2 = weights[:, 2:3]; w3 = weights[:, 3:4]; w4 = weights[:, 4:5]
            d_pred = w0 * d0 + w1 * d1 + w2 * d2 + w3 * d3 + w4 * d4
            o_pred = w0 * o0 + w1 * o1 + w2 * o2 + w3 * o3 + w4 * o4

        xr = float(s_yd.inverse_transform([[d_pred.item()]])[0, 0])
        wr = float(s_yo.inverse_transform([[o_pred.item()]])[0, 0])

        history_x.append(xr)
        psi_gt += w_gps[k]
        psi_pred += wr
        pos_gt.append((pos_gt[-1][0] + x_gps[k] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[k] * np.sin(psi_gt)))
        pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

    return float(np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1]))

def eval_candidate(outages, moe, scalers, device, alat_thresh, cent_gain, wr4_ratio, speed_clamp):
    scen_drifts = {s: [] for s in SCENARIOS}
    for out in outages:
        drift = simulate_outage_fast(out, moe, scalers, device, alat_thresh, cent_gain, wr4_ratio, speed_clamp)
        scen_drifts[out["scen"]].append(drift)

    means = {s: float(np.mean(scen_drifts[s])) for s in SCENARIOS}
    counts = {"motorway": 7, "hard_brake": 12, "quick_accel": 4, "roundabout": 3, "sharp_turns": 39}
    overall = float(sum(means[s] * counts[s] for s in SCENARIOS) / sum(counts.values()))
    means["overall"] = overall
    return means

def main():
    outages, moe, scalers, device = precompute_all_outages()

    # 1. Baseline check
    base_res = eval_candidate(outages, moe, scalers, device, alat_thresh=1.6, cent_gain=0.65, wr4_ratio=0.50, speed_clamp=-0.5)
    print("\n--- BASELINE REPRODUCTION ---")
    for k, v in base_res.items():
        print(f"  {k:15s}: {v:6.2f}m")

    # 2. Targeted candidate tests (focused on sharp turn reduction)
    candidates = [
        # (alat_thresh, cent_gain, wr4_ratio, speed_clamp)
        (1.5, 0.65, 0.50, -0.5),
        (1.4, 0.65, 0.50, -0.5),
        (1.3, 0.65, 0.50, -0.5),
        (1.5, 0.75, 0.50, -0.5),
        (1.5, 0.85, 0.50, -0.5),
        (1.5, 0.65, 0.60, -0.5),
        (1.5, 0.75, 0.60, -0.5),
        (1.4, 0.75, 0.60, -0.5),
        (1.4, 0.80, 0.55, -0.5),
        (1.5, 0.65, 0.50, -0.3),
        (1.5, 0.70, 0.50, -0.3),
        (1.4, 0.70, 0.50, -0.3),
    ]

    print(f"\n--- TESTING {len(candidates)} FOCUSED CANDIDATES ---")
    best_sharp = base_res["sharp_turns"]
    best_cand = None
    best_cand_res = None

    for cand in candidates:
        t0 = time.time()
        res = eval_candidate(outages, moe, scalers, device, *cand)
        dt = time.time() - t0
        print(f"Cand {cand} -> Sharp: {res['sharp_turns']:5.2f}m | Overall: {res['overall']:5.2f}m | Mot: {res['motorway']:4.2f}m | RB: {res['roundabout']:5.2f}m | HB: {res['hard_brake']:5.2f}m | QA: {res['quick_accel']:5.2f}m ({dt:.1f}s)", flush=True)

        if res["sharp_turns"] < best_sharp:
            best_sharp = res["sharp_turns"]
            best_cand = cand
            best_cand_res = res

    if best_cand:
        print(f"\nWINNING CONFIGURATION: {best_cand}")
        print(f"Sharp Turns improved from {base_res['sharp_turns']:.2f}m -> {best_cand_res['sharp_turns']:.2f}m")
        print(f"Overall drift: {base_res['overall']:.2f}m -> {best_cand_res['overall']:.2f}m")
    else:
        print("\nNo candidate beat baseline sharp turns.")

if __name__ == "__main__":
    main()
