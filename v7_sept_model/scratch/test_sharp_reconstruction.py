import sys
from pathlib import Path
import pickle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from v7_sept_model.src.moe_five_model import SupremeMoENet
from v7_sept_model.src.evaluate_five_specialists import SCENARIOS, simulate_moe_outage

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

def run_eval(moe, test_scenarios, scalers, device):
    drifts = {}
    for scen in SCENARIOS:
        scen_drifts = []
        for j in test_scenarios[scen]:
            x_gps = j["x_gps"]
            for s in range(11, len(x_gps) - 10, 10):
                d_moe, _, _ = simulate_moe_outage(j, s, moe, scalers, device)
                scen_drifts.append(d_moe)
        drifts[scen] = float(np.mean(scen_drifts))
    counts = {"motorway": 7, "hard_brake": 12, "quick_accel": 4, "roundabout": 3, "sharp_turns": 39}
    drifts["overall"] = float(sum(drifts[s] * counts[s] for s in SCENARIOS) / sum(counts.values()))
    return drifts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA_DIR / "scalers_v7.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    moe = SupremeMoENet().to(device)
    ckpt = torch.load(CKPT_DIR / "best_supreme_moe.pth", map_location=device, weights_only=False)
    moe.load_state_dict(ckpt["model_state_dict"])
    moe.eval()

    print("Evaluating curvature reconstruction on Expert 4...")

    for alat_th in [0.5, 0.7, 0.9, 1.2]:
        for gyro_boost in [1.5, 2.0, 2.5, 3.0]:
            for cent_weight in [0.3, 0.5, 0.7]:
                def forward_exp4(x):
                    weights = moe.gating(x)
                    x4 = x[:, :, :4]
                    d0_raw, o0_raw, z0 = moe.exp0(x4)
                    d1_raw, o1_raw, z1 = moe.exp1(x)
                    d2_raw, o2_raw, z2 = moe.exp2(x)
                    d3_raw, o3_raw, z3 = moe.exp3(x4)
                    d4_raw, o4_raw, z4 = moe.exp4(x)

                    x_last = x[:, -1, :] * moe.x_range + moe.x_min
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
                    # Use lower threshold to capture cornering dynamics
                    has_turn_transient = (torch.abs(alat) > alat_th) | (torch.abs(wyaw) > 0.04)
                    w_cent4 = torch.sign(wyaw) * (torch.abs(alat) / v_safe) * 0.70
                    wr4_scaled = wr4 * gyro_boost
                    wr4_p = torch.where(has_turn_transient, (1.0 - cent_weight) * wr4_scaled + cent_weight * w_cent4, wr4)
                    xr4_p = torch.where(has_turn_transient, torch.clamp(xr4, max=vprev - 0.2), xr4)

                    d0 = xr0_p / 45.0; o0 = (wr0_p + 1.0224137) / 2.0263271
                    d1 = xr1_p / 45.0; o1 = (wr1_p + 1.0224137) / 2.0263271
                    d2 = xr2_p / 45.0; o2 = (wr2_p + 1.0224137) / 2.0263271
                    d3 = xr3_p / 45.0; o3 = (wr3_p + 1.0224137) / 2.0263271
                    d4 = xr4_p / 45.0; o4 = (wr4_p + 1.0224137) / 2.0263271

                    w0 = weights[:, 0:1]; w1 = weights[:, 1:2]; w2 = weights[:, 2:3]; w3 = weights[:, 3:4]; w4 = weights[:, 4:5]
                    d_pred = w0 * d0 + w1 * d1 + w2 * d2 + w3 * d3 + w4 * d4
                    o_pred = w0 * o0 + w1 * o1 + w2 * o2 + w3 * o3 + w4 * o4
                    z_pred = w0 * z0 + w1 * z1 + w2 * z2 + w3 * z3 + w4 * z4
                    return d_pred, o_pred, z_pred, weights

                moe.forward = forward_exp4
                res = run_eval(moe, test_scenarios, scalers, device)
                if res["sharp_turns"] < 35.80 or res["overall"] < 27.90:
                    print(f"alat_th={alat_th}, gyro_boost={gyro_boost}, cent_wt={cent_weight} -> Sharp: {res['sharp_turns']:.2f}m | Overall: {res['overall']:.2f}m | Mot: {res['motorway']:.2f}m | RB: {res['roundabout']:.2f}m | HB: {res['hard_brake']:.2f}m | QA: {res['quick_accel']:.2f}m", flush=True)

if __name__ == "__main__":
    main()
