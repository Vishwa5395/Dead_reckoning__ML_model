import pickle
import numpy as np
import torch
from v7_sept_model.src.models_v7 import StraightSpecialistS1, TurningSpecialistS2, DualSpecialistRuntimeSwitch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("v7_sept_model/data/scalers_v7.pkl", "rb") as f:
    scalers = pickle.load(f)
with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

s_X = scalers["X"]
s_yd = scalers["y_disp"]
s_yo = scalers["y_ori"]

s1 = StraightSpecialistS1().to(device)
ckpt_s1 = torch.load("v7_sept_model/checkpoints/best_specialist_S1.pth", map_location=device)
s1.load_state_dict(ckpt_s1["model_state_dict"])
s1.eval()

s2 = TurningSpecialistS2(in_channels=6).to(device)
ckpt_s2 = torch.load("v7_sept_model/checkpoints/best_specialist_S2.pth", map_location=device)
s2.load_state_dict(ckpt_s2["model_state_dict"])
s2.eval()

switcher = DualSpecialistRuntimeSwitch(model_s1=s1, model_s2=s2, threshold_yaw_rate=0.03285, fade_time_s=0.5).to(device)
switcher.eval()

journeys = scens["hard_brake"]
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
        switcher.reset_state()
        from v7_sept_model.src.evaluate_v7 import ZUPTHysteresisGate
        zupt_gate = ZUPTHysteresisGate()

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

            measured_yaw = float(np.mean(np.abs(ch_w_yaw)))
            with torch.no_grad():
                d, o, z, alpha = switcher(torch.tensor(win_s6, dtype=torch.float32, device=device), measured_yaw_rate=measured_yaw, dt=1.0)
            xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
            wr = float(s_yo.inverse_transform([[o.item()]])[0, 0])

            p_stop = float(torch.sigmoid(z).item())
            if zupt_gate.update(p_stop):
                xr = 0.0
                wr = 0.0

            history_x.append(xr)
            psi_gt += w_gps[cur]
            psi_pred += wr

            pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
            pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

        drift_10s = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
        drifts.append(drift_10s)

print(f"DualSpecialist Standard on Hard Brake Mean Drift: {np.mean(drifts):.2f}m")
