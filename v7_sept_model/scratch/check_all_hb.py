import pickle
from pathlib import Path
import numpy as np
import torch
from v7_sept_model.src.models_v7 import StraightSpecialistS1, TurningSpecialistS2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("v7_sept_model/data/scalers_v7.pkl", "rb") as f:
    scalers = pickle.load(f)
with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

s_X = scalers["X"]
s_yd = scalers["y_disp"]
s_yo = scalers["y_ori"]
journeys = scens["hard_brake"]

ckpts = [
    ("v3 Base (4ch)", "v3_pino_dr/checkpoints/best_model.pth", 4),
    ("v4 Ablation-B (6ch)", "v4_turn_focused/checkpoints/best_model_ablation_B_directional_attn.pth", 6),
    ("v4 Ablation-C (6ch)", "v4_turn_focused/checkpoints/best_model_ablation_C_hierarchical_gru.pth", 6),
    ("v4 Ablation-D (6ch)", "v4_turn_focused/checkpoints/best_model_ablation_D_attn_coupling.pth", 6),
    ("v4 Ablation-E (6ch)", "v4_turn_focused/checkpoints/best_model_ablation_E_curvature_loss.pth", 6),
    ("v4 Full (6ch)", "v4_turn_focused/checkpoints/best_model_full_dual_head.pth", 6),
]

for name, ckpt_path, ch in ckpts:
    p = Path(ckpt_path)
    if not p.exists():
        continue
    if ch == 4:
        model = StraightSpecialistS1(in_channels=4).to(device)
    else:
        model = TurningSpecialistS2(in_channels=6).to(device)
    try:
        ckpt = torch.load(p, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as e:
        print(f"Skipping {name}: {e}")
        continue
    model.eval()

    drifts = []
    for j in journeys:
        x_gps = j["x_gps"]
        w_gps = j["w_gps"]
        a_fwd = j["a_fwd"]
        w_yaw = j["w_yaw"]
        a_lat = j["a_lat"]
        w_accel = j["w_yaw_accel"]
        headings = j["headings"]

        for s in range(11, len(x_gps) - 10, 10):
            history_x = list(x_gps[s - 10: s])
            pos_gt = [(0.0, 0.0)]
            pos_pred = [(0.0, 0.0)]
            psi_gt = np.radians(headings[s])
            psi_pred = psi_gt

            for k in range(10):
                cur = s + k
                ch_a_fwd = np.clip(a_fwd[cur-9:cur+1], -8.0, 8.0)
                ch_w_yaw = np.clip(w_yaw[cur-9:cur+1], -1.0, 1.0)
                ch_a_lat = np.clip(a_lat[cur-9:cur+1], -8.0, 8.0)
                ch_v_prev = np.clip(np.array(history_x[-10:]), 0.0, 45.0)
                ch_w_accel = np.clip(w_accel[cur-9:cur+1], -2.0, 2.0)
                ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)

                win_6 = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
                win_s = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

                with torch.no_grad():
                    if ch == 4:
                        d, o, _ = model(torch.tensor(win_s[:, :, :4], dtype=torch.float32, device=device))
                    else:
                        d, o, _ = model(torch.tensor(win_s, dtype=torch.float32, device=device))
                xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
                wr = float(s_yo.inverse_transform([[o.item()]])[0, 0])

                history_x.append(xr)
                psi_gt += w_gps[cur]
                psi_pred += wr

                pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
                pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

            drift_10s = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
            drifts.append(drift_10s)

    print(f"Checkpoint: {name:25s} -> Hard Brake Mean Drift: {np.mean(drifts):.2f}m")
