import pickle
import numpy as np
import torch
from v7_sept_model.src.models_v7 import StraightSpecialistS1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("v7_sept_model/data/scalers_v7.pkl", "rb") as f:
    scalers = pickle.load(f)
with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

s_X = scalers["X"]
s_yd = scalers["y_disp"]
s_yo = scalers["y_ori"]
j = scens["hard_brake"][0]
s = 41

model = StraightSpecialistS1(in_channels=4).to(device)
ckpt = torch.load("v3_pino_dr/checkpoints/best_model.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

x_gps = j["x_gps"]
w_gps = j["w_gps"]
a_fwd = j["a_fwd"]
w_yaw = j["w_yaw"]
a_lat = j["a_lat"]
w_accel = j["w_yaw_accel"]
headings = j["headings"]

history_x = list(x_gps[s - 10: s])
print(f"HARD BRAKE J0 s=41 DETAILED STEP ANALYSIS:")
for k in range(10):
    cur = s + k
    ch_a_fwd = np.clip(a_fwd[cur-9:cur+1], -8.0, 8.0)
    ch_w_yaw = np.clip(w_yaw[cur-9:cur+1], -1.0, 1.0)
    ch_a_lat = np.clip(a_lat[cur-9:cur+1], -8.0, 8.0)
    ch_v_prev = np.clip(np.array(history_x[-10:]), 0.0, 45.0)
    ch_w_accel = np.clip(w_accel[cur-9:cur+1], -2.0, 2.0)
    ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)

    win_6 = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
    win_s4 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6)[:, :, :4].astype(np.float32)

    with torch.no_grad():
        d, o, _ = model(torch.tensor(win_s4, dtype=torch.float32, device=device))
    xr = float(s_yd.inverse_transform([[d.item()]])[0, 0])
    wr = float(s_yo.inverse_transform([[o.item()]])[0, 0])

    print(f"k={k:02d} | GT_v={x_gps[cur]:5.2f}, Pred_v={xr:5.2f}, err={xr - x_gps[cur]:+5.2f} | a_fwd={a_fwd[cur]:+5.2f}, a_lat={a_lat[cur]:+5.2f}")
    history_x.append(xr)
