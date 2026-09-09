"""
models_specialists.py
---------------------
Genuinely Independent Specialist Architectures for PINO-DR:
  1. TurningBackboneModelA:
     - Specialized in yaw-rate estimation, turning dynamics, centripetal acceleration, lateral stability.
     - Primary outputs: w (yaw rate, zero-centered), bias_gyro, uncertainty_w, coupled delta_v.
  2. LongitudinalBackboneModelB:
     - Specialized in longitudinal acceleration, braking deceleration, speed propagation, ZUPT standstill.
     - Primary outputs: delta_v (velocity change, zero-centered), zupt (standstill logit), bias_accel, uncertainty_v, coupled w.

Architectural Principles:
  - Strictly Causal (Causal Conv1D left-padding, unidirectional past-to-present GRU, causal attention)
  - Zero arbitrary parameter cap initially (ample capacity to master maneuver families without underfitting)
  - ZERO shared weights, ZERO shared latent representations, ZERO gradient leakage.
  - Zero-centered linear output heads (NO artificial [0, 1] clamping!).
"""

from __future__ import annotations

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Causal Conv1D Block ───────────────────────────────────────────────────

class _CausalConvBlock(nn.Module):
    """
    1D Convolution with explicit left-padding for strict causality.
    At timestep t, it only sees timesteps <= t.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.kernel = kernel
        self.dilation = dilation
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=0, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self._init()

    def _init(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        # Left-pad only along time dimension
        x_padded = F.pad(x, (self.pad, 0))
        return self.act(self.bn(self.conv(x_padded)))


# ─── Causal Temporal Self-Attention ────────────────────────────────────────

class _CausalTemporalAttention(nn.Module):
    """
    Multi-head temporal self-attention over the GRU hidden representations.
    Attends across past sequence timesteps to capture long-range maneuver context.
    """

    def __init__(self, feat_dim: int, num_heads: int = 2):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, L, feat_dim)
        B, L, D = h.shape
        q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        k = self.k_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        v = self.v_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        # Causal mask: query at step t can only attend to key at step <= t
        causal_mask = torch.triu(torch.ones(L, L, device=h.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)  # (B, H, L, d)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)

        # Focus on the most recent step context combined with residual
        out = self.out_proj(ctx)[:, -1, :] + h[:, -1, :]  # (B, D)
        return out


# ============================================================================
# MODEL A — Turning Specialist Backbone
# ============================================================================

class TurningBackboneModelA(nn.Module):
    """
    Turning Specialist Neural Network.
    Trained strictly on Roundabout and Sharp Turn dynamics.
    Primary objective: Master yaw rate (w), heading changes, lateral dynamics, gyro bias.
    """

    def __init__(
        self,
        in_channels: int = 6,
        conv_channels: int = 48,
        gru_hidden: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Causal Conv1D Stem
        self.conv1 = _CausalConvBlock(in_channels, conv_channels, kernel=3, dilation=1)
        self.conv2 = _CausalConvBlock(conv_channels, conv_channels, kernel=3, dilation=2)
        self.conv_drop = nn.Dropout(dropout)

        # Causal Unidirectional GRU (strictly forward in time)
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Causal Temporal Attention
        self.attention = _CausalTemporalAttention(feat_dim=gru_hidden, num_heads=2)

        # Specialist Output Heads
        # Primary: Yaw rate head (w, zero-centered)
        self.head_w = nn.Sequential(
            nn.Linear(gru_hidden, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

        # Secondary: Coupled speed change (delta_v, zero-centered)
        self.head_dv = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Gyro bias residual head (zero-centered)
        self.head_bias_gyro = nn.Sequential(
            nn.Linear(gru_hidden, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )

        # Predictive uncertainty for yaw rate (log variance)
        self.head_log_var_w = nn.Sequential(
            nn.Linear(gru_hidden, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.head_w, self.head_dv, self.head_bias_gyro, self.head_log_var_w]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, L=20, C=6) temporal IMU window.
        Returns:
            w_pred:       (B, 1) yaw rate prediction (zero-centered, standardized)
            dv_pred:      (B, 1) coupled delta-v prediction (zero-centered, standardized)
            bw_pred:      (B, 1) gyro bias prediction (zero-centered)
            log_var_w:    (B, 1) log-variance for yaw rate
        """
        h = x.permute(0, 2, 1)  # (B, C, L)
        h = self.conv1(h)
        h = self.conv2(h) + h
        h = self.conv_drop(h)

        h = h.permute(0, 2, 1)  # (B, L, conv_channels)
        h, _ = self.gru(h)      # (B, L, gru_hidden)

        ctx = self.attention(h) # (B, gru_hidden)

        w_pred = self.head_w(ctx)
        dv_pred = self.head_dv(ctx)
        bw_pred = self.head_bias_gyro(ctx)
        log_var_w = torch.clamp(self.head_log_var_w(ctx), -4.0, 4.0)

        return w_pred, dv_pred, bw_pred, log_var_w


# ============================================================================
# MODEL B — Longitudinal Specialist Backbone
# ============================================================================

class LongitudinalBackboneModelB(nn.Module):
    """
    Longitudinal Specialist Neural Network.
    Trained strictly on Motorway, Quick Acceleration, and Hard Braking dynamics.
    Primary objective: Master speed change (delta_v), acceleration/braking, ZUPT standstill, accel bias.
    """

    def __init__(
        self,
        in_channels: int = 6,
        conv_channels: int = 48,
        gru_hidden: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Causal Conv1D Stem
        self.conv1 = _CausalConvBlock(in_channels, conv_channels, kernel=3, dilation=1)
        self.conv2 = _CausalConvBlock(conv_channels, conv_channels, kernel=3, dilation=2)
        self.conv_drop = nn.Dropout(dropout)

        # Causal Unidirectional GRU
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Causal Temporal Attention
        self.attention = _CausalTemporalAttention(feat_dim=gru_hidden, num_heads=2)

        # Specialist Output Heads
        # Primary: Forward velocity change (delta_v, zero-centered)
        self.head_dv = nn.Sequential(
            nn.Linear(gru_hidden, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

        # Secondary: Coupled yaw rate (w, zero-centered)
        self.head_w = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Primary: ZUPT standstill detector (raw logit for BCEWithLogits)
        self.head_zupt = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Accelerometer motion residual head (zero-centered)
        self.head_bias_accel = nn.Sequential(
            nn.Linear(gru_hidden, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )

        # Predictive uncertainty for velocity (log variance)
        self.head_log_var_v = nn.Sequential(
            nn.Linear(gru_hidden, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.head_dv, self.head_w, self.head_zupt, self.head_bias_accel, self.head_log_var_v]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, L=20, C=6) temporal IMU window.
        Returns:
            dv_pred:      (B, 1) delta-v prediction (zero-centered, standardized)
            w_pred:       (B, 1) coupled yaw rate prediction (zero-centered, standardized)
            z_logit:      (B, 1) ZUPT logit (raw log-odds)
            ba_pred:      (B, 1) accel bias prediction (zero-centered)
            log_var_v:    (B, 1) log-variance for delta-v
        """
        h = x.permute(0, 2, 1)  # (B, C, L)
        h = self.conv1(h)
        h = self.conv2(h) + h
        h = self.conv_drop(h)

        h = h.permute(0, 2, 1)  # (B, L, conv_channels)
        h, _ = self.gru(h)      # (B, L, gru_hidden)

        ctx = self.attention(h) # (B, gru_hidden)

        dv_pred = self.head_dv(ctx)
        w_pred = self.head_w(ctx)
        z_logit = self.head_zupt(ctx)
        ba_pred = self.head_bias_accel(ctx)
        log_var_v = torch.clamp(self.head_log_var_v(ctx), -4.0, 4.0)

        return dv_pred, w_pred, z_logit, ba_pred, log_var_v


if __name__ == "__main__":
    mA = TurningBackboneModelA()
    mB = LongitudinalBackboneModelB()
    pA = sum(p.numel() for p in mA.parameters() if p.requires_grad)
    pB = sum(p.numel() for p in mB.parameters() if p.requires_grad)
    print(f"TurningBackboneModelA params:      {pA:,}")
    print(f"LongitudinalBackboneModelB params: {pB:,}")
    print(f"Combined params:                   {pA + pB:,}")

    dummy = torch.randn(4, 20, 6)
    w_A, dv_A, bw_A, lv_w = mA(dummy)
    dv_B, w_B, z_B, ba_B, lv_v = mB(dummy)
    print("Model A forward pass OK:", w_A.shape, dv_A.shape, bw_A.shape, lv_w.shape)
    print("Model B forward pass OK:", dv_B.shape, w_B.shape, z_B.shape, ba_B.shape, lv_v.shape)
