"""
models_five_specialists.py
--------------------------
Five Genuinely Independent Specialist Models for Smartphone PINO-DR:
  - M1: VelocitySpecialistM1 (CNN + GRU, T=25, 14 features -> v, delta_v)
  - M2: YawSpecialistM2 (Rotation-focused CNN + GRU, T=20, 12 features -> w, delta_psi)
  - M3: AccelBiasSpecialistM3 (Temporal Dilated CNN + GRU, T=50, 6 features -> b_a)
  - M4: GyroBiasSpecialistM4 (Temporal Dilated CNN + GRU, T=50, 6 features -> b_g)
  - M5: UncertaintyAdapterM5 (Lightweight Temporal CNN, T=15, 9 features -> sigma_v, sigma_w, sigma_ba, sigma_bg)

Zero shared weights, zero shared embeddings, zero cross-talk between backbones.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Causal Building Blocks ──────────────────────────────────────────────────

class CausalConvBlock(nn.Module):
    """Causal Conv1D with left-padding to strictly prevent future temporal leakage."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.act(self.bn(self.conv(x)))


class CausalTemporalAttention(nn.Module):
    """Learnable temporal attention pooling over sequence states."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(feat_dim))
        self.scale = math.sqrt(feat_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, feat_dim)
        scores = torch.matmul(h, self.query) / self.scale  # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (h * weights).sum(dim=1)  # (B, feat_dim)


# ─── M1: Velocity Specialist ─────────────────────────────────────────────────

class VelocitySpecialistM1(nn.Module):
    """
    Specialist M1: Forward Velocity / Speed Propagation.
    Inputs: [ax, ay, az, gx, gy, gz, uz_x, uz_y, uz_z, a_fwd, a_lat, w_yaw, jerk, v_prev] (14 features)
    Window: T = 25 steps (2.5 seconds)
    Outputs: Forward speed (v), velocity increment (delta_v)
    """

    def __init__(self, in_features: int = 14, conv_dim: int = 32, gru_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.conv1 = CausalConvBlock(in_features, conv_dim, kernel_size=3, dilation=1)
        self.conv2 = CausalConvBlock(conv_dim, conv_dim, kernel_size=3, dilation=2)
        self.drop = nn.Dropout(dropout)

        self.gru = nn.GRU(conv_dim, gru_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.attn = CausalTemporalAttention(gru_dim)

        self.head_v = nn.Sequential(
            nn.Linear(gru_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.head_dv = nn.Sequential(
            nn.Linear(gru_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, 14) -> permute to (B, 14, T)
        c = self.drop(self.conv2(self.conv1(x.transpose(1, 2))))
        h, _ = self.gru(c.transpose(1, 2))  # (B, T, gru_dim)
        feat = self.attn(h)                # (B, gru_dim)
        v = self.head_v(feat)
        dv = self.head_dv(feat)
        return v, dv


# ─── M2: Yaw Rate Specialist ─────────────────────────────────────────────────

class YawSpecialistM2(nn.Module):
    """
    Specialist M2: Orientation & Rotational Dynamics.
    Inputs: [gx, gy, gz, ax, ay, az, uz_x, uz_y, uz_z, a_lat, a_fwd, w_yaw] (12 features)
    Window: T = 20 steps (2.0 seconds)
    Outputs: Yaw rate (w), heading increment (delta_psi)
    """

    def __init__(self, in_features: int = 12, conv_dim: int = 32, gru_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.conv1 = CausalConvBlock(in_features, conv_dim, kernel_size=3, dilation=1)
        self.conv2 = CausalConvBlock(conv_dim, conv_dim, kernel_size=3, dilation=2)
        self.drop = nn.Dropout(dropout)

        self.gru = nn.GRU(conv_dim, gru_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.attn = CausalTemporalAttention(gru_dim)

        self.head_w = nn.Sequential(
            nn.Linear(gru_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.head_dpsi = nn.Sequential(
            nn.Linear(gru_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, 12) -> permute to (B, 12, T)
        c = self.drop(self.conv2(self.conv1(x.transpose(1, 2))))
        h, _ = self.gru(c.transpose(1, 2))  # (B, T, gru_dim)
        feat = self.attn(h)                # (B, gru_dim)
        w = self.head_w(feat)
        dpsi = self.head_dpsi(feat)
        return w, dpsi


# ─── M3: Accelerometer Bias Specialist ───────────────────────────────────────

class AccelBiasSpecialistM3(nn.Module):
    """
    Specialist M3: Accelerometer Bias / Longitudinal Residual.
    Inputs: [a_fwd, a_norm, uz_x, uz_y, uz_z, v_prev] (6 features)
    Window: T = 50 steps (5.0 seconds - low-frequency emphasis)
    Outputs: Slowly varying acceleration bias (b_a)
    """

    def __init__(self, in_features: int = 6, conv_dim: int = 24, gru_dim: int = 32, dropout: float = 0.10):
        super().__init__()
        self.conv1 = CausalConvBlock(in_features, conv_dim, kernel_size=3, dilation=1)
        self.conv2 = CausalConvBlock(conv_dim, conv_dim, kernel_size=3, dilation=2)
        self.conv3 = CausalConvBlock(conv_dim, conv_dim, kernel_size=3, dilation=4)
        self.drop = nn.Dropout(dropout)

        self.gru = nn.GRU(conv_dim, gru_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.head_ba = nn.Sequential(
            nn.Linear(gru_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 6)
        c = self.drop(self.conv3(self.conv2(self.conv1(x.transpose(1, 2)))))
        h, _ = self.gru(c.transpose(1, 2))
        ba = self.head_ba(h[:, -1, :])  # Final causal step
        return ba


# ─── M4: Gyroscope Bias Specialist ───────────────────────────────────────────

class GyroBiasSpecialistM4(nn.Module):
    """
    Specialist M4: Gyroscope Bias / Yaw Correction.
    Inputs: [w_yaw, g_norm, w_accel, uz_x, uz_y, uz_z] (6 features)
    Window: T = 50 steps (5.0 seconds - low-frequency emphasis)
    Outputs: Slowly varying vertical gyro bias (b_g)
    """

    def __init__(self, in_features: int = 6, conv_dim: int = 24, gru_dim: int = 32, dropout: float = 0.10):
        super().__init__()
        self.conv1 = CausalConvBlock(in_features, conv_dim, kernel_size=3, dilation=1)
        self.conv2 = CausalConvBlock(conv_dim, conv_dim, kernel_size=3, dilation=2)
        self.conv3 = CausalConvBlock(conv_dim, conv_dim, kernel_size=3, dilation=4)
        self.drop = nn.Dropout(dropout)

        self.gru = nn.GRU(conv_dim, gru_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.head_bg = nn.Sequential(
            nn.Linear(gru_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 6)
        c = self.drop(self.conv3(self.conv2(self.conv1(x.transpose(1, 2)))))
        h, _ = self.gru(c.transpose(1, 2))
        bg = self.head_bg(h[:, -1, :])  # Final causal step
        return bg


# ─── M5: Uncertainty / Motion Confidence Adapter ─────────────────────────────

class UncertaintyAdapterM5(nn.Module):
    """
    Specialist M5: Pure Temporal CNN Adapter for Filter Measurement Covariances.
    Inputs: Raw IMU window [ax, ay, az, gx, gy, gz, uz_x, uz_y, uz_z] (9 features)
    Window: T = 15 steps (1.5 seconds)
    Outputs: [sigma_v, sigma_w, sigma_ba, sigma_bg] (strictly positive via Softplus)
    """

    def __init__(self, in_features: int = 9, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # (B, channels*2, 1)
            nn.Flatten(),
            nn.Linear(channels * 2, 32),
            nn.GELU(),
            nn.Linear(32, 4),
            nn.Softplus()  # Guarantees strictly positive standard deviations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 9) -> permute to (B, 9, T)
        sigmas = self.net(x.transpose(1, 2)) + 1e-4  # Minimum variance floor
        return sigmas  # (B, 4) -> [sigma_v, sigma_w, sigma_ba, sigma_bg]


if __name__ == "__main__":
    # Smoke test dimensions and parameter counts
    m1 = VelocitySpecialistM1()
    m2 = YawSpecialistM2()
    m3 = AccelBiasSpecialistM3()
    m4 = GyroBiasSpecialistM4()
    m5 = UncertaintyAdapterM5()

    print("M1 Params:", sum(p.numel() for p in m1.parameters()))
    print("M2 Params:", sum(p.numel() for p in m2.parameters()))
    print("M3 Params:", sum(p.numel() for p in m3.parameters()))
    print("M4 Params:", sum(p.numel() for p in m4.parameters()))
    print("M5 Params:", sum(p.numel() for p in m5.parameters()))

    x1 = torch.randn(2, 25, 14)
    x2 = torch.randn(2, 20, 12)
    x3 = torch.randn(2, 50, 6)
    x4 = torch.randn(2, 50, 6)
    x5 = torch.randn(2, 15, 9)

    v, dv = m1(x1)
    w, dpsi = m2(x2)
    ba = m3(x3)
    bg = m4(x4)
    sig = m5(x5)

    print("M1 output:", v.shape, dv.shape)
    print("M2 output:", w.shape, dpsi.shape)
    print("M3 output:", ba.shape)
    print("M4 output:", bg.shape)
    print("M5 output:", sig.shape)
    print("All forward passes successful!")
