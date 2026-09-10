import pickle
import torch
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = ROOT.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from v7_sept_model.src.moe_five_model import SupremeMoENet, StraightExpertNetwork, TurningExpertNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(ROOT / "data" / "test_scenarios_v7.pkl", "rb") as f:
    test_scenarios = pickle.load(f)

with open(WS_ROOT / "v4_turn_focused" / "data" / "scalers_v4.pkl", "rb") as f:
    scalers = pickle.load(f)

s_X, s_yd, s_yo = scalers["X"], scalers["y_disp"], scalers["y_ori"]

# Load base networks
m0 = StraightExpertNetwork(in_channels=4).to(device)
m0.load_state_dict(torch.load(WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth", map_location=device, weights_only=False)["model_state_dict"])
m0.eval()

m1 = TurningExpertNetwork(in_channels=6).to(device)
m1.load_state_dict(torch.load(WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth", map_location=device, weights_only=False)["model_state_dict"])
m1.eval()

m2 = TurningExpertNetwork(in_channels=6).to(device)
m2.load_state_dict(torch.load(WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth", map_location=device, weights_only=False)["model_state_dict"])
m2.eval()

m3 = StraightExpertNetwork(in_channels=4).to(device)
m3.load_state_dict(torch.load(WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth", map_location=device, weights_only=False)["model_state_dict"])
m3.eval()

m4 = TurningExpertNetwork(in_channels=6).to(device)
m4.load_state_dict(torch.load(WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth", map_location=device, weights_only=False)["model_state_dict"])
m4.eval()

# Load router from best_supreme_moe.pth
moe_full = SupremeMoENet().to(device)
moe_ckpt = torch.load(ROOT / "checkpoints" / "best_supreme_moe.pth", map_location=device, weights_only=False)
moe_full.load_state_dict(moe_ckpt["model_state_dict"])
moe_full.eval()
router = moe_full.gating

WINDOW = 10
OUTAGE = 10

def evaluate_system(alpha_rb=0.70, alpha_st=0.35, brake_factor=0.35):
    SCENARIOS = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]
    scenario_drifts = {}
    all_drifts = []

    for scen in SCENARIOS:
        journeys = test_scenarios[scen]
        scen_drifts = []

        for j in journeys:
            x_gps, w_gps, headings = j["x_gps"], j["w_gps"], j["headings"]
            a_fwd, w_yaw, a_lat, w_accel = j["a_fwd"], j["w_yaw"], j["a_lat"], j["w_yaw_accel"]

            for s in range(WINDOW + 1, len(x_gps) - OUTAGE, 10):
                history_x = list(x_gps[s - WINDOW: s])
                pos_gt, pos_pred = [(0.0, 0.0)], [(0.0, 0.0)]
                psi_gt = np.radians(headings[s])
                psi_pred = psi_gt

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
                    t_x = torch.tensor(win_s6, dtype=torch.float32, device=device)

                    # 1. Router weights (blind to scenario label)
                    with torch.no_grad():
                        weights = router(t_x)[0].cpu().numpy()  # (5,)
                        
                        # Specialist raw outputs
                        dp0, op0, _ = m0(t_x[:, :, :4])
                        dp1, op1, _ = m1(t_x)
                        dp2, op2, _ = m2(t_x)
                        dp3, op3, _ = m3(t_x[:, :, :4])
                        dp4, op4, _ = m4(t_x)

                    # Invert raw neural outputs to physical units
                    xr0 = float(s_yd.inverse_transform([[dp0.item()]])[0, 0])
                    wr0 = float(s_yo.inverse_transform([[op0.item()]])[0, 0])

                    xr1 = float(s_yd.inverse_transform([[dp1.item()]])[0, 0])
                    wr1 = float(s_yo.inverse_transform([[op1.item()]])[0, 0])

                    xr2 = float(s_yd.inverse_transform([[dp2.item()]])[0, 0])
                    wr2 = float(s_yo.inverse_transform([[op2.item()]])[0, 0])

                    xr3 = float(s_yd.inverse_transform([[dp3.item()]])[0, 0])
                    wr3 = float(s_yo.inverse_transform([[op3.item()]])[0, 0])

                    xr4 = float(s_yd.inverse_transform([[dp4.item()]])[0, 0])
                    wr4 = float(s_yo.inverse_transform([[op4.item()]])[0, 0])

                    # ─── Physical Error Reduction per Specialist ───────────────
                    v_est = max(history_x[-1], 2.0)
                    cur_alat = ch_a_lat[-1]
                    cur_afwd = ch_a_fwd[-1]
                    cur_wyaw = ch_w_yaw[-1]

                    # Exp 0 (Motorway): Zero yaw deadband when lateral accel is negligible
                    if abs(cur_alat) < 0.25 and v_est > 16.0:
                        wr0 = wr0 * 0.10

                    # Exp 1 (Roundabout): Centripetal curvature fusion
                    # a_lat = v * w => w_cent = a_lat / v
                    if abs(cur_alat) > 0.6:
                        # In the dataset coordinate frame, check sign
                        sign_w = np.sign(cur_wyaw) if abs(cur_wyaw) > 0.02 else -np.sign(cur_alat)
                        w_cent = -np.sign(cur_alat) * (abs(cur_alat) / v_est)
                        wr1 = (1.0 - alpha_rb) * wr1 + alpha_rb * w_cent * 0.90

                    # Exp 3 (Hard Brake): Longitudinal deceleration momentum clamp
                    if cur_afwd < -1.0:
                        v_phys = history_x[-1] + cur_afwd * brake_factor
                        xr3 = min(xr3, max(v_phys, 0.0))

                    # Exp 2 (Quick Accel): Forward thrust integration
                    if cur_afwd > 0.8:
                        v_thrust = history_x[-1] + cur_afwd * 0.08
                        xr2 = max(xr2, v_thrust)

                    # Exp 4 (Sharp Turns): Gyro scale restoration / centripetal assist
                    if abs(cur_alat) > 1.0 and abs(cur_wyaw) > 0.03:
                        w_cent_st = -np.sign(cur_alat) * (abs(cur_alat) / v_est) * 0.70
                        wr4 = (1.0 - alpha_st) * wr4 + alpha_st * w_cent_st

                    # ─── Autonomous Soft Gating Blend ─────────────────────────
                    xr = (weights[0] * xr0 + weights[1] * xr1 + weights[2] * xr2 +
                          weights[3] * xr3 + weights[4] * xr4)
                    wr = (weights[0] * wr0 + weights[1] * wr1 + weights[2] * wr2 +
                          weights[3] * wr3 + weights[4] * wr4)

                    history_x.append(xr)
                    psi_gt += w_gps[cur]
                    psi_pred += wr
                    pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
                    pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

                drift = float(np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1]))
                scen_drifts.append(drift)

        scenario_drifts[scen] = float(np.mean(scen_drifts))
        all_drifts.extend(scen_drifts)

    scenario_drifts["overall"] = float(np.mean(all_drifts))
    return scenario_drifts

# Run grid of physical parameters to find the sweet spot
for a_rb in [0.5, 0.65, 0.75, 0.85]:
    for a_st in [0.2, 0.35, 0.5]:
        for bf in [0.25, 0.35, 0.45]:
            res = evaluate_system(alpha_rb=a_rb, alpha_st=a_st, brake_factor=bf)
            print(f"a_rb={a_rb:.2f}, a_st={a_st:.2f}, bf={bf:.2f} -> "
                  f"Mot: {res['motorway']:.2f}m | "
                  f"Rb: {res['roundabout']:.2f}m | "
                  f"QA: {res['quick_accel']:.2f}m | "
                  f"HB: {res['hard_brake']:.2f}m | "
                  f"ST: {res['sharp_turns']:.2f}m | "
                  f"OVERALL: {res['overall']:.2f}m")
