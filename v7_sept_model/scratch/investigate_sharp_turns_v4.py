import pickle
import numpy as np
import torch
from v7_sept_model.src.models_v7 import TurningSpecialistS2

class ZUPTHysteresisGate:
    def __init__(self, threshold_high=0.7, threshold_low=0.3, n_enter=3, n_exit=2):
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.n_enter = n_enter
        self.n_exit = n_exit
        self.is_stopped = False
        self.high_count = 0
        self.low_count = 0

    def reset(self):
        self.is_stopped = False
        self.high_count = 0
        self.low_count = 0

    def update(self, p_stop: float) -> bool:
        if not self.is_stopped:
            if p_stop > self.threshold_high:
                self.high_count += 1
                self.low_count = 0
                if self.high_count >= self.n_enter:
                    self.is_stopped = True
            else:
                self.high_count = 0
        else:
            if p_stop < self.threshold_low:
                self.low_count += 1
                self.high_count = 0
                if self.low_count >= self.n_exit:
                    self.is_stopped = False
            else:
                self.low_count = 0
        return self.is_stopped

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('v7_sept_model/data/scalers_v7.pkl', 'rb') as f: scalers = pickle.load(f)
with open('v7_sept_model/data/test_scenarios_v7.pkl', 'rb') as f: scens = pickle.load(f)

s_X = scalers['X']; s_yd = scalers['y_disp']; s_yo = scalers['y_ori']

model = TurningSpecialistS2(in_channels=6).to(device)
ckpt = torch.load('v4_turn_focused/checkpoints/best_model_ablation_D_attn_coupling.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

journeys = scens['sharp_turns']

drifts_pure_v4 = []
drifts_v7 = []

for j_idx, j in enumerate(journeys):
    x_gps = j['x_gps']
    a_fwd = j['a_fwd']
    w_yaw = j['w_yaw']
    a_lat = j['a_lat']
    w_accel = j['w_yaw_accel']
    w_gps = j['w_gps']
    headings = j['headings']

    for s in range(11, len(x_gps) - 10, 10):
        # 1. Pure v4 evaluation
        history_x = list(x_gps[s - 10: s])
        pos_gt = [(0.0, 0.0)]
        pos_pred = [(0.0, 0.0)]
        psi_gt = np.radians(headings[s])
        psi_pred = psi_gt
        zupt_gate = ZUPTHysteresisGate()

        for k in range(10):
            cur = s + k
            win_6 = np.stack([
                np.clip(a_fwd[cur-9:cur+1], -8, 8),
                np.clip(w_yaw[cur-9:cur+1], -1, 1),
                np.clip(a_lat[cur-9:cur+1], -8, 8),
                np.clip(np.array(history_x[-10:]), 0, 45),
                np.clip(w_accel[cur-9:cur+1], -2, 2),
                np.clip(a_lat[cur-9:cur+1] - np.array(history_x[-10:]) * w_yaw[cur-9:cur+1], -8, 8)
            ], axis=-1)
            win_s6 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

            with torch.no_grad():
                d, o, z = model(torch.tensor(win_s6, dtype=torch.float32, device=device))
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

        drift_pure_v4 = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
        drifts_pure_v4.append(drift_pure_v4)

print(f"Pure v4 Mean Drift on Sharp Turns: {np.mean(drifts_pure_v4):.2f} m")
