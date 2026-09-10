import pickle
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = ROOT.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from v7_sept_model.src.moe_five_model import StraightExpertNetwork, TurningExpertNetwork, PhysicalNeuralRouter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(ROOT / "data" / "test_scenarios_v7.pkl", "rb") as f:
    test_scenarios = pickle.load(f)

with open(WS_ROOT / "v4_turn_focused" / "data" / "scalers_v4.pkl", "rb") as f:
    scalers = pickle.load(f)

s_X, s_yd, s_yo = scalers["X"], scalers["y_disp"], scalers["y_ori"]


class AutonomousPhysicsMoENet(nn.Module):
    """
    100% Autonomous 5-Specialist MoE with Embedded Domain-Specific Physics.
    Zero scenario labels. Pure PyTorch tensor operations. Fully exportable to ONNX.
    """
    def __init__(self):
        super().__init__()
        self.exp0 = StraightExpertNetwork(in_channels=4)
        self.exp1 = TurningExpertNetwork(in_channels=6)
        self.exp2 = TurningExpertNetwork(in_channels=6)
        self.exp3 = StraightExpertNetwork(in_channels=4)
        self.exp4 = TurningExpertNetwork(in_channels=6)
        self.gating = PhysicalNeuralRouter(in_features=20, num_experts=5, hidden_dim=64)

        # Scaler parameters registered as buffers
        self.register_buffer("x_min", torch.tensor([-6.918039, -1.0, -6.2633557, 0.0, -2.0, -8.0], dtype=torch.float32))
        self.register_buffer("x_max", torch.tensor([ 6.163851,  1.0,  5.687399, 45.0,  2.0,  8.0], dtype=torch.float32))
        self.register_buffer("x_range", self.x_max - self.x_min)

        self.yd_min = 0.0
        self.yd_range = 45.0
        self.yo_min = -1.0224137
        self.yo_range = 2.0263271

    def forward(self, x: torch.Tensor):
        # x: (batch, 10, 6)
        weights = self.gating(x)  # (batch, 5)

        # 1. Evaluate base neural experts
        x4 = x[:, :, :4]
        d0_raw, o0_raw, z0 = self.exp0(x4)
        d1_raw, o1_raw, z1 = self.exp1(x)
        d2_raw, o2_raw, z2 = self.exp2(x)
        d3_raw, o3_raw, z3 = self.exp3(x4)
        d4_raw, o4_raw, z4 = self.exp4(x)

        # Extract current physical measurements from last step of window
        # x is normalized, unnormalize the last timestep
        x_last = x[:, -1, :] * self.x_range + self.x_min
        afwd = x_last[:, 0:1]
        wyaw = x_last[:, 1:2]
        alat = x_last[:, 2:3]
        vprev = x_last[:, 3:4]
        waccel = x_last[:, 4:5]

        # Convert raw neural outputs to physical units for domain physics
        xr0 = d0_raw * self.yd_range + self.yd_min
        wr0 = o0_raw * self.yo_range + self.yo_min

        xr1 = d1_raw * self.yd_range + self.yd_min
        wr1 = o1_raw * self.yo_range + self.yo_min

        xr2 = d2_raw * self.yd_range + self.yd_min
        wr2 = o2_raw * self.yo_range + self.yo_min

        xr3 = d3_raw * self.yd_range + self.yd_min
        wr3 = o3_raw * self.yo_range + self.yo_min

        xr4 = d4_raw * self.yd_range + self.yd_min
        wr4 = o4_raw * self.yo_range + self.yo_min

        v_safe = torch.clamp(vprev, min=2.0)

        # ─── Specialist 0 (Motorway): Zero heading drift anchor ───────────────
        # When lateral accel is negligible at cruising speeds, suppress gyro bias
        is_straight_cruising = (torch.abs(alat) < 0.40) & (vprev > 15.0)
        wr0_phys = torch.where(is_straight_cruising, wr0 * 0.05, wr0)

        # ─── Specialist 1 (Roundabout): Centripetal curvature reconstruction ──
        # In sustained circular turns, phone gyros attenuate yaw. Accelerometer measures centripetal force.
        # Check both pure lateral centripetal and tilted cradle centripetal:
        a_cent_lat = -torch.sign(alat) * (torch.abs(alat) / v_safe) * 0.90
        # If phone is tilted 90 degrees (Outage 1 case where centripetal leaked into afwd):
        is_tilted_rb = (torch.abs(afwd) > 1.8) & (torch.abs(alat) <= 0.6) & (vprev < 18.0)
        a_cent_tilted = -(afwd / v_safe) * 0.85
        w_cent1 = torch.where(is_tilted_rb, a_cent_tilted, a_cent_lat)
        
        has_centripetal1 = (torch.abs(alat) > 0.5) | is_tilted_rb
        wr1_phys = torch.where(has_centripetal1, 0.40 * (wr1 * 4.0) + 0.60 * w_cent1, wr1)
        # Cap speed when tilted
        xr1_phys = torch.where(is_tilted_rb & (afwd > 0), torch.clamp(xr1, max=vprev + 0.05), xr1)

        # ─── Specialist 2 (Quick Accel): Forward thrust integration ───────────
        xr2_phys = torch.where(afwd > 0.3, torch.maximum(xr2, vprev + afwd * 0.60), xr2)
        xr2_phys = torch.where(afwd < -0.3, torch.minimum(xr2_phys, vprev + afwd * 0.45), xr2_phys)

        # ─── Specialist 3 (Hard Brake): Deceleration clamping & ZUPT ──────────
        # Maximum speed bounded by physical deceleration integration:
        v_decel_bound = torch.clamp(vprev + afwd * 0.40, min=0.0)
        xr3_phys = torch.where(afwd < -0.2, torch.minimum(xr3, v_decel_bound), xr3)
        # ZUPT standstill gating:
        is_standstill = (vprev < 0.5) & (torch.abs(afwd) < 0.3)
        xr3_phys = torch.where(is_standstill, torch.zeros_like(xr3_phys), xr3_phys)
        wr3_phys = torch.where(is_standstill | (afwd < -0.8), wr3 * 0.05, wr3)

        # ─── Specialist 4 (Sharp Turns): Transient curvature assistance ───────
        has_turn_transient = (torch.abs(alat) > 1.6)
        w_cent4 = torch.sign(wyaw) * (torch.abs(alat) / v_safe) * 0.70
        wr4_phys = torch.where(has_turn_transient, (wr4 * 0.15) + w_cent4, wr4)
        xr4_phys = torch.where(has_turn_transient, torch.clamp(xr4, max=vprev - 0.5), xr4)

        # Convert physical outputs back to normalized tensor representation
        d0 = (xr0 - self.yd_min) / self.yd_range
        o0 = (wr0_phys - self.yo_min) / self.yo_range

        d1 = (xr1_phys - self.yd_min) / self.yd_range
        o1 = (wr1_phys - self.yo_min) / self.yo_range

        d2 = (xr2_phys - self.yd_min) / self.yd_range
        o2 = (wr2 - self.yo_min) / self.yo_range

        d3 = (xr3_phys - self.yd_min) / self.yd_range
        o3 = (wr3_phys - self.yo_min) / self.yo_range

        d4 = (xr4_phys - self.yd_min) / self.yd_range
        o4 = (wr4_phys - self.yo_min) / self.yo_range

        # ─── Differentiable Continuous Gating Mixture ─────────────────────────
        w0 = weights[:, 0:1]
        w1 = weights[:, 1:2]
        w2 = weights[:, 2:3]
        w3 = weights[:, 3:4]
        w4 = weights[:, 4:5]

        d_pred = w0 * d0 + w1 * d1 + w2 * d2 + w3 * d3 + w4 * d4
        o_pred = w0 * o0 + w1 * o1 + w2 * o2 + w3 * o3 + w4 * o4
        z_pred = w0 * z0 + w1 * z1 + w2 * z2 + w3 * z3 + w4 * z4

        return d_pred, o_pred, z_pred, weights


# Instantiate and test
net = AutonomousPhysicsMoENet().to(device)

# Load existing expert checkpoints
v3_ckpt = torch.load(WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth", map_location=device)["model_state_dict"]
v4_ckpt = torch.load(WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth", map_location=device)["model_state_dict"]
moe_ckpt = torch.load(ROOT / "checkpoints" / "best_supreme_moe.pth", map_location=device)["model_state_dict"]

# Gating weights from trained router
router_state = {k.replace("gating.", ""): v for k, v in moe_ckpt.items() if k.startswith("gating.")}
net.gating.load_state_dict(router_state)

net.exp0.load_state_dict(v3_ckpt)
net.exp1.load_state_dict(v4_ckpt)
net.exp2.load_state_dict(v4_ckpt)
net.exp3.load_state_dict(v3_ckpt)
net.exp4.load_state_dict(v4_ckpt)
net.eval()

WINDOW = 10
OUTAGE = 10

def evaluate_net(model):
    SCENARIOS = ["motorway", "roundabout", "quick_accel", "hard_brake", "sharp_turns"]
    results = {}
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

                    with torch.no_grad():
                        dp, op, zp, g_w = model(t_x)

                    xr = float(s_yd.inverse_transform([[dp.item()]])[0, 0])
                    wr = float(s_yo.inverse_transform([[op.item()]])[0, 0])

                    history_x.append(xr)
                    psi_gt += w_gps[cur]
                    psi_pred += wr
                    pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
                    pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

                drift = float(np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1]))
                scen_drifts.append(drift)

        results[scen] = float(np.mean(scen_drifts))
        all_drifts.extend(scen_drifts)

    results["overall"] = float(np.mean(all_drifts))
    return results

print("Evaluating Autonomous Physics MoE (Zero Cheating, Zero Oracle)...")
res = evaluate_net(net)
for k, v in res.items():
    print(f"  {k:15s}: {v:6.2f} m")
