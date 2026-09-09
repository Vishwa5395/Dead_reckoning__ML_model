import pickle, torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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

# Focal turn weight
diff_tr = torch.abs(yo_tr - 0.504565)
weights_tr = (1.0 + 12.0 * torch.clamp(diff_tr / 0.10, 0.0, 4.0) ** 2).to(device)

loader = DataLoader(TensorDataset(X_tr, yd_tr, yo_tr, yz_tr, weights_tr), batch_size=256, shuffle=True)
va_loader = DataLoader(TensorDataset(X_va, yd_va, yo_va, yz_va), batch_size=256, shuffle=False)

# Enhanced Architecture for S2
class EnhancedTurningSpecialist(nn.Module):
    def __init__(self, in_channels=6, conv_dim=48, gru_dim=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(conv_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.gru = nn.GRU(
            input_size=conv_dim,
            hidden_size=gru_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.20
        )
        gru_out = gru_dim * 2 # 128
        
        # 4-head attention
        self.attn = nn.MultiheadAttention(embed_dim=gru_out, num_heads=4, batch_first=True, dropout=0.1)
        
        # Specialized heads
        self.disp_head = nn.Sequential(
            nn.Linear(gru_out, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1)
        )
        self.ori_head = nn.Sequential(
            nn.Linear(gru_out, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.zupt_head = nn.Sequential(
            nn.Linear(gru_out, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, 10, 6)
        h = x.permute(0, 2, 1)
        h1 = self.conv1(h)
        h2 = self.conv2(h1)
        h = (h1 + h2).permute(0, 2, 1)
        
        gru_out, _ = self.gru(h)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        context = attn_out[:, -1, :] # pool last timestep
        
        delta_v = self.disp_head(context)
        v_prev_last = x[:, -1, 3:4]
        disp = torch.clamp(v_prev_last + delta_v, 0.0, 1.0)
        ori = self.ori_head(context)
        zupt = self.zupt_head(context)
        return disp, ori, zupt

model = EnhancedTurningSpecialist().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-6)
loss_huber = nn.HuberLoss(delta=0.08, reduction='none')
loss_zupt = nn.BCEWithLogitsLoss()

print('Training Enhanced S2 Architecture (45 epochs)...')
for ep in range(1, 46):
    model.train()
    tot_l = 0.0
    for X, yd, yo, yz, w in loader:
        optimizer.zero_grad()
        d_p, o_p, z_p = model(X)
        l_d = loss_huber(d_p, yd).mean()
        l_o = (loss_huber(o_p, yo) * w).mean()
        l_z = loss_zupt(z_p, yz)
        loss = l_d + 8.0 * l_o + 0.25 * l_z
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot_l += loss.item()
    scheduler.step()
    
    if ep % 10 == 0 or ep == 45:
        print(f'Epoch {ep:02d} | Train Loss: {tot_l/len(loader):.5f}')

# Evaluate on Roundabout and Sharp Turns
with open('v7_sept_model/data/scalers_v7.pkl', 'rb') as f: scalers = pickle.load(f)
with open('v7_sept_model/data/test_scenarios_v7.pkl', 'rb') as f: scens = pickle.load(f)
s_X = scalers['X']; s_yd = scalers['y_disp']; s_yo = scalers['y_ori']

model.eval()
for test_scen in ['roundabout', 'sharp_turns', 'hard_brake']:
    journeys = scens[test_scen]
    drifts = []
    for j in journeys:
        x_gps = j['x_gps']; w_gps = j['w_gps']; a_fwd = j['a_fwd']; w_yaw = j['w_yaw']; a_lat = j['a_lat']; w_accel = j['w_yaw_accel']; headings = j['headings']
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
    print(f'Enhanced S2 on {test_scen.upper()}: Mean Drift = {np.mean(drifts):.2f}m')
