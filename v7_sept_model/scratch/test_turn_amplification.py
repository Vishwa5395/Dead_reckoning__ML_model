import pickle, torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from v7_sept_model.src.models_v7 import TurningSpecialistS2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = np.load('v7_sept_model/data/dataset_splits_v7.npz')

X_tr = torch.tensor(data['X_tr_s2_replay'], dtype=torch.float32, device=device)
yd_tr = torch.tensor(data['y_d_tr_s2_replay'], dtype=torch.float32, device=device)
yo_tr = torch.tensor(data['y_o_tr_s2_replay'], dtype=torch.float32, device=device)
yz_tr = torch.tensor(data['y_z_tr_s2_replay'], dtype=torch.float32, device=device)

X_va = torch.tensor(data['X_va_s2'], dtype=torch.float32, device=device)
yd_va = torch.tensor(data['y_d_va_s2'], dtype=torch.float32, device=device)
yo_va = torch.tensor(data['y_o_va_s2'], dtype=torch.float32, device=device)
yz_va = torch.tensor(data['y_z_va_s2'], dtype=torch.float32, device=device)

# Curvature weights
diff_tr = torch.abs(yo_tr - 0.504565)
weights_tr = (1.0 + 10.0 * torch.clamp(diff_tr / 0.12, 0.0, 3.0) ** 2).to(device)

loader = DataLoader(TensorDataset(X_tr, yd_tr, yo_tr, yz_tr, weights_tr), batch_size=256, shuffle=True)
va_loader = DataLoader(TensorDataset(X_va, yd_va, yo_va, yz_va), batch_size=256, shuffle=False)

# Load pretrained base Ablation-D
model = TurningSpecialistS2(in_channels=6).to(device)
ckpt = torch.load('v4_turn_focused/checkpoints/best_model_ablation_D_attn_coupling.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])

optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)
loss_huber = nn.HuberLoss(delta=0.1, reduction='none')
loss_zupt = nn.BCEWithLogitsLoss()

print('Starting Turn-Amplification Training for S2 (35 epochs)...')
for ep in range(1, 36):
    model.train()
    tot_loss = 0.0
    for X, yd, yo, yz, w in loader:
        optimizer.zero_grad()
        d_p, o_p, z_p = model(X)
        l_d = loss_huber(d_p, yd).mean()
        # Weighted orientation loss: sharp turns get up to 10x penalty for underprediction!
        l_o = (loss_huber(o_p, yo) * w).mean()
        l_z = loss_zupt(z_p, yz)
        loss = l_d + 6.0 * l_o + 0.25 * l_z
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot_loss += loss.item()
    scheduler.step()
    
    if ep % 5 == 0 or ep == 35:
        model.eval()
        va_ori_mae = 0.0; va_b = 0
        with torch.no_grad():
            for X, yd, yo, yz in va_loader:
                _, o_p, _ = model(X)
                va_ori_mae += nn.functional.l1_loss(o_p, yo).item()
                va_b += 1
        va_ori_mae /= max(va_b, 1)
        print(f'Epoch {ep:02d} | Train Loss: {tot_loss/len(loader):.5f} | Val Ori MAE: {va_ori_mae:.5f}')

# Now evaluate on Roundabout
with open('v7_sept_model/data/scalers_v7.pkl', 'rb') as f: scalers = pickle.load(f)
with open('v7_sept_model/data/test_scenarios_v7.pkl', 'rb') as f: scens = pickle.load(f)
s_X = scalers['X']; s_yd = scalers['y_disp']; s_yo = scalers['y_ori']

model.eval()
rb_j = scens['roundabout'][0]
x_gps = rb_j['x_gps']; w_gps = rb_j['w_gps']; a_fwd = rb_j['a_fwd']; w_yaw = rb_j['w_yaw']; a_lat = rb_j['a_lat']; w_accel = rb_j['w_yaw_accel']; headings = rb_j['headings']

drifts = []
for start_idx in range(11, len(x_gps) - 10, 10):
    history_x = list(x_gps[start_idx - 10: start_idx])
    pos_gt = [(0.0, 0.0)]; pos_pred = [(0.0, 0.0)]
    psi_gt = np.radians(headings[start_idx]); psi_pred = psi_gt
    for k in range(10):
        cur = start_idx + k
        ch_a_fwd = np.clip(a_fwd[cur-9:cur+1], -8.0, 8.0)
        ch_w_yaw = np.clip(w_yaw[cur-9:cur+1], -1.0, 1.0)
        ch_a_lat = np.clip(a_lat[cur-9:cur+1], -8.0, 8.0)
        ch_v_prev = np.clip(np.array(history_x[-10:]), 0.0, 45.0)
        ch_w_accel = np.clip(w_accel[cur-9:cur+1], -2.0, 2.0)
        ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)
        win = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
        win_s = s_X.transform(win.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)
        with torch.no_grad():
            d_p, o_p, _ = model(torch.tensor(win_s, dtype=torch.float32, device=device))
        xr = float(s_yd.inverse_transform([[d_p.item()]])[0, 0])
        wr = float(s_yo.inverse_transform([[o_p.item()]])[0, 0])
        history_x.append(xr)
        psi_gt += w_gps[cur]; psi_pred += wr
        pos_gt.append((pos_gt[-1][0] + x_gps[cur]*np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur]*np.sin(psi_gt)))
        pos_pred.append((pos_pred[-1][0] + xr*np.cos(psi_pred), pos_pred[-1][1] + xr*np.sin(psi_pred)))
    d = np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1])
    drifts.append(d)

print(f'\n>>> PRO-TUNED S2 ROUNDABOUT MEAN DRIFT: {np.mean(drifts):.2f}m (vs previous 55.36m / 60.41m) <<<')
