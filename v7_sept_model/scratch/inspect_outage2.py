import pickle
import numpy as np
import torch
from v7_sept_model.src.models_v7 import TurningSpecialistS2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("v7_sept_model/data/scalers_v7.pkl", "rb") as f:
    scalers = pickle.load(f)
with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

s_X = scalers["X"]
s_yd = scalers["y_disp"]
s_yo = scalers["y_ori"]

s2_qa = TurningSpecialistS2(in_channels=6).to(device)
ckpt_v4 = torch.load("v4_turn_focused/checkpoints/best_model_ablation_D_attn_coupling.pth", map_location=device)
s2_qa.load_state_dict(ckpt_v4["model_state_dict"])
s2_qa.eval()

j = scens["quick_accel"][0]
x_gps = j["x_gps"]
w_gps = j["w_gps"]
a_fwd = j["a_fwd"]
w_yaw = j["w_yaw"]
a_lat = j["a_lat"]
w_accel = j["w_yaw_accel"]
headings = j["headings"]

s = 21  # outage 2
history_x = list(x_gps[s - 10: s])
pos_gt = [(0.0, 0.0)]
pos_pred = [(0.0, 0.0)]
psi_gt = np.radians(headings[s])
psi_pred = psi_gt

print(f"OUTAGE 2 DETAILED STEP ANALYSIS:")
for k in range(10):
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
        d, o, _ = s2_qa(torch.tensor(win_s6, dtype=torch.float32, device=device))
    xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
    wr = float(s_yo.inverse_transform([[o.item()]])[0, 0])

    if a_fwd[cur] > 0.4:
        xr = max(xr, history_x[-1] + a_fwd[cur] * 0.70)
    elif a_fwd[cur] < -0.4:
        xr = min(xr, history_x[-1] + a_fwd[cur] * 0.40)

    history_x.append(xr)
    psi_gt += w_gps[cur]
    psi_pred += wr

    pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
    pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

    step_drift = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
    head_err_deg = np.degrees(abs(psi_pred - psi_gt))
    print(f"k={k:02d} | GT_v={x_gps[cur]:5.2f}, Pred_v={xr:5.2f} | a_fwd={a_fwd[cur]:5.2f}, a_lat={a_lat[cur]:5.2f} | w_gps={w_gps[cur]:+6.3f}, wr={wr:+6.3f} | HeadErr={head_err_deg:5.2f}°, Drift={step_drift:5.2f}m")
