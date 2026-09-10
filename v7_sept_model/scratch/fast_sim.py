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

# Load models
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

moe_full = SupremeMoENet().to(device)
moe_ckpt = torch.load(ROOT / "checkpoints" / "best_supreme_moe.pth", map_location=device, weights_only=False)
moe_full.load_state_dict(moe_ckpt["model_state_dict"])
moe_full.eval()
router = moe_full.gating

WINDOW = 10
OUTAGE = 10

X_min = np.array([-6.918039, -1.0, -6.2633557, 0.0, -2.0, -8.0], dtype=np.float32)
X_max = np.array([ 6.163851,  1.0,  5.687399, 45.0,  2.0,  8.0], dtype=np.float32)
X_range = X_max - X_min

def eval_config(alpha_rb=0.75, alpha_st=0.30, bf=0.35):
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
                    win_norm = (win_6 - X_min) / X_range
                    t_x = torch.from_numpy(win_norm).unsqueeze(0).float().to(device)

                    with torch.no_grad():
                        weights = router(t_x)[0].cpu().numpy()
                        dp0, op0, _ = m0(t_x[:, :, :4])
                        dp1, op1, _ = m1(t_x)
                        dp2, op2, _ = m2(t_x)
                        dp3, op3, _ = m3(t_x[:, :, :4])
                        dp4, op4, _ = m4(t_x)

                    xr0 = dp0.item() * 45.0
                    wr0 = op0.item() * 2.0263271 - 1.0224137

                    xr1 = dp1.item() * 45.0
                    wr1 = op1.item() * 2.0263271 - 1.0224137

                    xr2 = dp2.item() * 45.0
                    wr2 = op2.item() * 2.0263271 - 1.0224137

                    xr3 = dp3.item() * 45.0
                    wr3 = op3.item() * 2.0263271 - 1.0224137

                    xr4 = dp4.item() * 45.0
                    wr4 = op4.item() * 2.0263271 - 1.0224137

                    # Kinematic specialist rules
                    v_est = max(history_x[-1], 2.0)
                    cur_alat = ch_a_lat[-1]
                    cur_afwd = ch_a_fwd[-1]
                    cur_wyaw = ch_w_yaw[-1]

                    # 0: Motorway
                    if abs(cur_alat) < 0.25 and v_est > 16.0:
                        wr0 = wr0 * 0.10

                    # 1: Roundabout
                    if abs(cur_alat) > 0.6:
                        w_cent = -np.sign(cur_alat) * (abs(cur_alat) / v_est)
                        wr1 = (1.0 - alpha_rb) * wr1 + alpha_rb * w_cent * 0.90

                    # 3: Hard Brake
                    if cur_afwd < -1.0:
                        v_phys = history_x[-1] + cur_afwd * bf
                        xr3 = min(xr3, max(v_phys, 0.0))

                    # 2: Quick Accel
                    if cur_afwd > 0.8:
                        xr2 = max(xr2, history_x[-1] + cur_afwd * 0.08)

                    # 4: Sharp Turns
                    if abs(cur_alat) > 0.8 and abs(cur_wyaw) > 0.03:
                        w_cent_st = -np.sign(cur_alat) * (abs(cur_alat) / v_est) * 0.70
                        wr4 = (1.0 - alpha_st) * wr4 + alpha_st * w_cent_st

                    xr = float(weights[0]*xr0 + weights[1]*xr1 + weights[2]*xr2 + weights[3]*xr3 + weights[4]*xr4)
                    wr = float(weights[0]*wr0 + weights[1]*wr1 + weights[2]*wr2 + weights[3]*wr3 + weights[4]*wr4)

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

print("Running fast evaluation...")
res = eval_config(alpha_rb=0.75, alpha_st=0.30, bf=0.35)
print("Results with Physical Specialist Error Reduction:")
for k, v in res.items():
    print(f"  {k:15s}: {v:.2f} m")
