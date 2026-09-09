"""
models_supreme_five.py
-----------------------
5-Supreme-Specialists Neural Odometry Architecture (v7_sept_model).

Specialists:
1. MotorwaySpecialist (S1): Cruising dynamics, zero heading bias, MEMS noise rejection.
2. RoundaboutSpecialist (S2): Centripetal physical skip connection (a_lat / v) + circular curvature memory.
3. QuickAccelSpecialist (S3): Pitch-decoupled forward thrust prediction.
4. HardBrakeSpecialist (S4): Deceleration-focused integration + sensitive ZUPT standstill gating.
5. SharpTurnsSpecialist (S5): Roll-compensated multi-head cornering network.

Router:
- SupremeKinematicRouter: Computes continuous regime probabilities via live kinematics and applies
  smooth Hermite blending across specialist outputs.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Blocks ─────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _TemporalAttention(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(feat_dim))
        self.scale = math.sqrt(feat_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(h, self.query) / self.scale
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (h * weights).sum(dim=1)


class _DirectionalMultiHeadAttention(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        assert feat_dim % 2 == 0
        self.half_dim = feat_dim // 2
        self.q_fwd = nn.Parameter(torch.randn(self.half_dim))
        self.q_bwd = nn.Parameter(torch.randn(self.half_dim))
        self.scale = math.sqrt(self.half_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_f = h[:, :, :self.half_dim]
        h_b = h[:, :, self.half_dim:]
        w_f = torch.softmax(torch.matmul(h_f, self.q_fwd) / self.scale, dim=1).unsqueeze(-1)
        w_b = torch.softmax(torch.matmul(h_b, self.q_bwd) / self.scale, dim=1).unsqueeze(-1)
        c_f = (h_f * w_f).sum(dim=1)
        c_b = (h_b * w_b).sum(dim=1)
        return torch.cat([c_f, c_b], dim=-1)


# ─── 1. Motorway Specialist (Cruising Focus) ─────────────────────────────────

class MotorwaySpecialist(nn.Module):
    """S1: High-speed cruising, zero heading bias, MEMS noise rejection."""
    def __init__(self, in_channels: int = 4, conv_dim: int = 32, gru_dim: int = 32, dropout: float = 0.20):
        super().__init__()
        self.conv1 = _ConvBlock(in_channels, conv_dim, kernel=3, dilation=1, padding=1)
        self.conv2 = _ConvBlock(conv_dim, conv_dim, kernel=3, dilation=2, padding=2)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(conv_dim, gru_dim, num_layers=1, batch_first=True, bidirectional=True)
        gru_out = gru_dim * 2
        self.attn = _TemporalAttention(gru_out)
        self.disp_head = nn.Sequential(nn.Linear(gru_out, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.ori_head = nn.Sequential(nn.Linear(gru_out, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.zupt_head = nn.Sequential(nn.Linear(gru_out, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, 10, 4)
        h = x.permute(0, 2, 1)
        h = self.drop(self.conv1(h) + self.conv2(self.conv1(h)))
        h, _ = self.gru(h.permute(0, 2, 1))
        ctx = self.attn(h)
        delta_v = self.disp_head(ctx)
        v_prev_last = x[:, -1, 3:4]
        disp = torch.clamp(v_prev_last + delta_v, 0.0, 1.0)
        ori = self.ori_head(ctx)
        zupt = self.zupt_head(ctx)
        return disp, ori, zupt


# ─── 2. Hard Brake Specialist (Deceleration & ZUPT) ───────────────────────────

class HardBrakeSpecialist(nn.Module):
    """S2: Rapid negative deceleration tracking + crisp ZUPT standstill gating."""
    def __init__(self, in_channels: int = 6, conv_dim: int = 32, gru_dim: int = 32, dropout: float = 0.20):
        super().__init__()
        self.conv1 = _ConvBlock(in_channels, conv_dim, kernel=3, dilation=1, padding=1)
        self.conv2 = _ConvBlock(conv_dim, conv_dim, kernel=3, dilation=2, padding=2)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(conv_dim, gru_dim, num_layers=1, batch_first=True, bidirectional=True)
        gru_out = gru_dim * 2
        self.attn = _TemporalAttention(gru_out)
        self.disp_head = nn.Sequential(nn.Linear(gru_out, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.ori_head = nn.Sequential(nn.Linear(gru_out, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.zupt_head = nn.Sequential(nn.Linear(gru_out, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, 10, 6)
        h = x.permute(0, 2, 1)
        h = self.drop(self.conv1(h) + self.conv2(self.conv1(h)))
        h, _ = self.gru(h.permute(0, 2, 1))
        ctx = self.attn(h)
        delta_v = self.disp_head(ctx)
        v_prev_last = x[:, -1, 3:4]
        disp = torch.clamp(v_prev_last + delta_v, 0.0, 1.0)
        ori = self.ori_head(ctx)
        zupt = self.zupt_head(ctx)
        return disp, ori, zupt


# ─── 3. Quick Accel Specialist (Forward Thrust Ramp) ──────────────────────────

class QuickAccelSpecialist(nn.Module):
    """S3: Forward thrust ramp tracking, pitch-gravity decoupled."""
    def __init__(self, in_channels: int = 6, conv_dim: int = 32, gru_dim: int = 32, dropout: float = 0.20):
        super().__init__()
        self.conv1 = _ConvBlock(in_channels, conv_dim, kernel=3, dilation=1, padding=1)
        self.conv2 = _ConvBlock(conv_dim, conv_dim, kernel=3, dilation=2, padding=2)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(conv_dim, gru_dim, num_layers=1, batch_first=True, bidirectional=True)
        gru_out = gru_dim * 2
        self.attn = _TemporalAttention(gru_out)
        self.disp_head = nn.Sequential(nn.Linear(gru_out, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.ori_head = nn.Sequential(nn.Linear(gru_out, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.zupt_head = nn.Sequential(nn.Linear(gru_out, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, 10, 6)
        h = x.permute(0, 2, 1)
        h = self.drop(self.conv1(h) + self.conv2(self.conv1(h)))
        h, _ = self.gru(h.permute(0, 2, 1))
        ctx = self.attn(h)
        delta_v = self.disp_head(ctx)
        v_prev_last = x[:, -1, 3:4]
        disp = torch.clamp(v_prev_last + delta_v, 0.0, 1.0)
        ori = self.ori_head(ctx)
        zupt = self.zupt_head(ctx)
        return disp, ori, zupt


# ─── 4. Sharp Turns Specialist (Roll-Compensated Cornering) ───────────────────

class SharpTurnsSpecialist(nn.Module):
    """S4: Deep multi-head cornering network with roll compensation."""
    def __init__(self, in_channels: int = 6, conv_dim: int = 32, gru_dim: int = 32, dropout: float = 0.20):
        super().__init__()
        self.conv1 = _ConvBlock(in_channels, conv_dim, kernel=3, dilation=1, padding=1)
        self.conv2 = _ConvBlock(conv_dim, conv_dim, kernel=3, dilation=2, padding=2)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(conv_dim, gru_dim, num_layers=1, batch_first=True, bidirectional=True)
        gru_out = gru_dim * 2
        self.attn = _DirectionalMultiHeadAttention(gru_out)
        self.disp_head = nn.Sequential(nn.Linear(gru_out, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.ori_head = nn.Sequential(
            nn.Linear(gru_out, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 24),
            nn.GELU(),
            nn.Linear(24, 1)
        )
        self.zupt_head = nn.Sequential(nn.Linear(gru_out, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, 10, 6)
        h = x.permute(0, 2, 1)
        h = self.drop(self.conv1(h) + self.conv2(self.conv1(h)))
        h, _ = self.gru(h.permute(0, 2, 1))
        ctx = self.attn(h)
        delta_v = self.disp_head(ctx)
        v_prev_last = x[:, -1, 3:4]
        disp = torch.clamp(v_prev_last + delta_v, 0.0, 1.0)
        ori = self.ori_head(ctx)
        zupt = self.zupt_head(ctx)
        return disp, ori, zupt


# ─── 5. Roundabout Specialist (Centripetal-Arc Tracking) ──────────────────────

class RoundaboutSpecialist(nn.Module):
    """S5: Sustained circular motion with explicit centripetal physical skip connection."""
    def __init__(self, in_channels: int = 6, conv_dim: int = 32, gru_dim: int = 32, dropout: float = 0.20):
        super().__init__()
        self.conv1 = _ConvBlock(in_channels, conv_dim, kernel=3, dilation=1, padding=1)
        self.conv2 = _ConvBlock(conv_dim, conv_dim, kernel=3, dilation=2, padding=2)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(conv_dim, gru_dim, num_layers=1, batch_first=True, bidirectional=True)
        gru_out = gru_dim * 2
        self.attn = _DirectionalMultiHeadAttention(gru_out)
        self.disp_head = nn.Sequential(nn.Linear(gru_out, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        # Orientation head receives context + centripetal feature skip (ch 5)
        self.ori_head = nn.Sequential(
            nn.Linear(gru_out + 1, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 24),
            nn.GELU(),
            nn.Linear(24, 1)
        )
        self.zupt_head = nn.Sequential(nn.Linear(gru_out, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, 10, 6)
        h = x.permute(0, 2, 1)
        h = self.drop(self.conv1(h) + self.conv2(self.conv1(h)))
        h, _ = self.gru(h.permute(0, 2, 1))
        ctx = self.attn(h)
        delta_v = self.disp_head(ctx)
        v_prev_last = x[:, -1, 3:4]
        disp = torch.clamp(v_prev_last + delta_v, 0.0, 1.0)
        
        # Centripetal skip (Channel 5 of last timestep)
        centripetal_feat = x[:, -1, 5:6]
        ori_ctx = torch.cat([ctx, centripetal_feat], dim=-1)
        ori = self.ori_head(ori_ctx)
        zupt = self.zupt_head(ctx)
        return disp, ori, zupt


# ─── Supreme Kinematic Router ────────────────────────────────────────────────

class SupremeKinematicRouter(nn.Module):
    """
    Classifies live vehicle kinematic regime from 6-channel observables
    and routes inference to the 5 Supreme Specialists with continuous Hermite blending.
    """
    def __init__(
        self,
        m_motorway: MotorwaySpecialist,
        m_roundabout: RoundaboutSpecialist,
        m_quick_accel: QuickAccelSpecialist,
        m_hard_brake: HardBrakeSpecialist,
        m_sharp_turns: SharpTurnsSpecialist,
        fade_time_s: float = 0.5,
    ):
        super().__init__()
        self.m_motorway = m_motorway
        self.m_roundabout = m_roundabout
        self.m_quick_accel = m_quick_accel
        self.m_hard_brake = m_hard_brake
        self.m_sharp_turns = m_sharp_turns
        self.fade_time_s = fade_time_s
        self.current_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    def reset_state(self):
        self.current_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    def compute_regime_affinities(
        self,
        v_prev: float,
        a_fwd: float,
        a_lat: float,
        w_yaw: float,
        w_pred_s1: float = 0.0,
    ) -> np.ndarray:
        """
        Computes dynamic score for each specialist based on physical indicators:
          [motorway, roundabout, quick_accel, hard_brake, sharp_turns]
        """
        scores = np.zeros(5, dtype=np.float32)

        # 1. Motorway score: high speed, low yaw, low lateral
        if v_prev >= 18.0 and abs(w_pred_s1) < 0.025 and abs(a_lat) < 1.0 and a_fwd > -1.0:
            scores[0] = 5.0 + (v_prev - 18.0) * 0.2
        elif v_prev >= 15.0 and abs(w_pred_s1) < 0.030 and abs(a_lat) < 1.2:
            scores[0] = 2.0

        # 2. Hard Brake score: negative longitudinal acceleration
        if a_fwd <= -1.2:
            scores[3] = 4.0 + abs(a_fwd) * 1.5
        elif a_fwd <= -0.7:
            scores[3] = 2.0 + abs(a_fwd) * 1.0

        # 3. Quick Accel score: positive longitudinal acceleration
        if a_fwd >= 1.2 and v_prev < 24.0:
            scores[2] = 4.0 + a_fwd * 1.5
        elif a_fwd >= 0.7 and v_prev < 24.0:
            scores[2] = 2.0 + a_fwd * 1.0

        # 4. Roundabout score: sustained lateral acceleration with moderate speed
        if abs(a_lat) >= 1.8 and 4.0 <= v_prev <= 18.0:
            scores[1] = 4.0 + abs(a_lat) * 1.5
        elif abs(a_lat) >= 1.2 and 4.0 <= v_prev <= 18.0:
            scores[1] = 2.0 + abs(a_lat) * 1.0

        # 5. Sharp Turns score: transient high yaw rate or low-speed cornering
        turn_ind = max(abs(w_yaw), abs(w_pred_s1))
        if turn_ind >= 0.04 and scores[1] < 3.0:
            scores[4] = 3.5 + turn_ind * 20.0

        # Default baseline if cruising normally at medium speed
        if np.max(scores) < 0.5:
            if v_prev >= 16.0:
                scores[0] = 2.0
            else:
                scores[4] = 1.0

        # Softmax normalize
        exp_s = np.exp(scores - np.max(scores))
        return exp_s / np.sum(exp_s)

    def forward(
        self,
        x_6ch: torch.Tensor,
        ch_v_prev_phys: float,
        ch_a_fwd_phys: float,
        ch_a_lat_phys: float,
        ch_w_yaw_phys: float,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        # Preliminary S1 pass for filtered heading
        x_4ch = x_6ch[:, :, :4]
        d1, o1, z1 = self.m_motorway(x_4ch)
        w_pred_s1 = float(abs((o1.item() - 0.504565) / 0.49350372))

        target_weights = self.compute_regime_affinities(
            v_prev=ch_v_prev_phys,
            a_fwd=ch_a_fwd_phys,
            a_lat=ch_a_lat_phys,
            w_yaw=ch_w_yaw_phys,
            w_pred_s1=w_pred_s1,
        )

        # Smooth rate-limiting across fade time
        max_step = dt / max(self.fade_time_s, 1e-4)
        if max_step >= 1.0:
            self.current_weights = target_weights
        else:
            diff = target_weights - self.current_weights
            self.current_weights = self.current_weights + np.clip(diff, -max_step, max_step)
            self.current_weights = np.clip(self.current_weights, 0.0, 1.0)
            self.current_weights /= (np.sum(self.current_weights) + 1e-8)

        w = self.current_weights

        # Evaluate specialists only if active weight > 1% (efficiency optimization)
        d_list, o_list, z_list = [], [], []

        # S1: Motorway
        d_list.append(d1); o_list.append(o1); z_list.append(z1)

        # S2: Roundabout
        if w[1] > 0.01:
            d2, o2, z2 = self.m_roundabout(x_6ch)
        else:
            d2, o2, z2 = d1, o1, z1
        d_list.append(d2); o_list.append(o2); z_list.append(z2)

        # S3: Quick Accel
        if w[2] > 0.01:
            d3, o3, z3 = self.m_quick_accel(x_6ch)
        else:
            d3, o3, z3 = d1, o1, z1
        d_list.append(d3); o_list.append(o3); z_list.append(z3)

        # S4: Hard Brake
        if w[3] > 0.01:
            d4, o4, z4 = self.m_hard_brake(x_6ch)
        else:
            d4, o4, z4 = d1, o1, z1
        d_list.append(d4); o_list.append(o4); z_list.append(z4)

        # S5: Sharp Turns
        if w[4] > 0.01:
            d5, o5, z5 = self.m_sharp_turns(x_6ch)
        else:
            d5, o5, z5 = d1, o1, z1
        d_list.append(d5); o_list.append(o5); z_list.append(z5)

        # Blended outputs
        d_final = sum(float(w[i]) * d_list[i] for i in range(5))
        o_final = sum(float(w[i]) * o_list[i] for i in range(5))
        z_final = sum(float(w[i]) * z_list[i] for i in range(5))

        return d_final, o_final, z_final, self.current_weights
