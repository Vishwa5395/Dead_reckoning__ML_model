"""
models_v7.py
------------
Dual-Specialist Neural Odometry Architecture for PINO-DR v7 (v7_sept_model).

Specialists:
1. StraightSpecialistS1:
   - Initialized from v3 PINO-DR weights (21,667 params).
   - 4 input channels: [a_fwd, w_yaw, a_lat, v_prev].
   - BiGRU + Learnable Single-Query Temporal Attention + Multi-Task Heads.
   - Optimized strictly on straight/low-yaw driving dynamics (|yaw_rate| < 60th percentile).

2. TurningSpecialistS2:
   - Initialized from v4 Ablation-D weights (21,941 params).
   - 6 input channels: [a_fwd, w_yaw, a_lat, v_prev, w_yaw_accel, centripetal_residual].
   - BiGRU + Directional 2-Head Temporal Attention + Cross-Task Coupling + Multi-Task Heads.
   - Optimized strictly on cornering/high-yaw dynamics (|yaw_rate| >= 60th percentile).

3. DualSpecialistRuntimeSwitch:
   - Computes window |yaw_rate| measure live.
   - Soft crossover blending between S1 and S2 over ~0.5s transition window to prevent kinematic discontinuities.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Tuple, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn


# ─── Common Building Blocks ──────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Conv1D -> BatchNorm -> GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 dilation: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self._init()

    def _init(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _SingleTemporalAttention(nn.Module):
    """Learnable query-based attention pooling over time dimension (S1)."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(feat_dim))
        self.scale = math.sqrt(feat_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (batch, seq_len, feat_dim)
        scores = torch.matmul(h, self.query) / self.scale
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (h * weights).sum(dim=1)


class _DirectionalMultiHeadTemporalAttention(nn.Module):
    """
    2-head directional temporal attention (S2):
    - Head 1 attends over forward GRU half (steady-state dynamics).
    - Head 2 attends over backward GRU half (turn initiation / exit transients).
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        assert feat_dim % 2 == 0, "feat_dim must be even for directional splitting"
        self.half_dim = feat_dim // 2
        self.query_fwd = nn.Parameter(torch.randn(self.half_dim))
        self.query_bwd = nn.Parameter(torch.randn(self.half_dim))
        self.scale = math.sqrt(self.half_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_fwd = h[:, :, :self.half_dim]
        h_bwd = h[:, :, self.half_dim:]

        scores_fwd = torch.matmul(h_fwd, self.query_fwd) / self.scale
        w_fwd = torch.softmax(scores_fwd, dim=1).unsqueeze(-1)
        c_fwd = (h_fwd * w_fwd).sum(dim=1)

        scores_bwd = torch.matmul(h_bwd, self.query_bwd) / self.scale
        w_bwd = torch.softmax(scores_bwd, dim=1).unsqueeze(-1)
        c_bwd = (h_bwd * w_bwd).sum(dim=1)

        return torch.cat([c_fwd, c_bwd], dim=-1)


class _CrossTaskCoupling(nn.Module):
    """
    Mutual coupling layer between preliminary velocity and yaw predictions (S2).
    Strictly uses model predictions with zero-initialized residual layer.
    """

    def __init__(self):
        super().__init__()
        self.coupling = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 2),
        )
        for m in self.coupling.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, v_init: torch.Tensor, w_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([v_init, w_init], dim=-1)
        delta = self.coupling(inp)
        v_coupled = torch.clamp(v_init + delta[:, 0:1], 0.0, 1.0)
        w_coupled = w_init + delta[:, 1:2]
        return v_coupled, w_coupled


# ─── Specialist S1: Straight-Driving Specialist ───────────────────────────────

class StraightSpecialistS1(nn.Module):
    """
    Straight-Driving Specialist (S1).
    Initialized from v3 PINO-DR weights (21,667 params).
    Input: (batch, 10, 4) -> [a_fwd, w_yaw, a_lat, v_prev].
    """

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels: int = 32,
        gru_hidden: int = 32,
        num_gru_layers: int = 1,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.gru_hidden = gru_hidden

        self.conv1 = _ConvBlock(in_channels, conv_channels, kernel=3, dilation=1, padding=1)
        self.conv2 = _ConvBlock(conv_channels, conv_channels, kernel=3, dilation=2, padding=2)
        self.conv_drop = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        gru_out_dim = gru_hidden * 2  # 64

        self.attention = _SingleTemporalAttention(gru_out_dim)

        self.disp_head = nn.Sequential(
            nn.Linear(gru_out_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.ori_head = nn.Sequential(
            nn.Linear(gru_out_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.zupt_head = nn.Sequential(
            nn.Linear(gru_out_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self._init_heads()

    def _init_heads(self):
        for module in [self.disp_head, self.ori_head, self.zupt_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, 10, 4)
        h = x.permute(0, 2, 1)
        h1 = self.conv1(h)
        h2 = self.conv2(h1)
        h = self.conv_drop(h1 + h2)

        h = h.permute(0, 2, 1)
        h, _ = self.gru(h)

        context = self.attention(h)

        delta_v = self.disp_head(context)
        v_prev_last = x[:, -1, 3:4]
        disp = torch.clamp(v_prev_last + delta_v, 0.0, 1.0)
        ori = self.ori_head(context)
        zupt = self.zupt_head(context)

        return disp, ori, zupt


# ─── Specialist S2: Turning Specialist ───────────────────────────────────────

class TurningSpecialistS2(nn.Module):
    """
    Turning Specialist (S2).
    Initialized from v4 Ablation-D weights (21,941 params).
    Input: (batch, 10, 6) -> [a_fwd, w_yaw, a_lat, v_prev, w_yaw_accel, centripetal_res].
    """

    def __init__(
        self,
        in_channels: int = 6,
        conv_channels: int = 32,
        gru_hidden: int = 32,
        num_gru_layers: int = 1,
        dropout: float = 0.20,
        use_multihead_attention: bool = True,
        use_cross_task_coupling: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.gru_hidden = gru_hidden
        self.use_multihead_attention = use_multihead_attention
        self.use_cross_task_coupling = use_cross_task_coupling

        self.conv1 = _ConvBlock(in_channels, conv_channels, kernel=3, dilation=1, padding=1)
        self.conv2 = _ConvBlock(conv_channels, conv_channels, kernel=3, dilation=2, padding=2)
        self.conv_drop = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        gru_out_dim = gru_hidden * 2  # 64

        if use_multihead_attention:
            self.attention = _DirectionalMultiHeadTemporalAttention(gru_out_dim)
        else:
            self.attention = _SingleTemporalAttention(gru_out_dim)

        self.disp_head = nn.Sequential(
            nn.Linear(gru_out_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.ori_head = nn.Sequential(
            nn.Linear(gru_out_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.zupt_head = nn.Sequential(
            nn.Linear(gru_out_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

        if use_cross_task_coupling:
            self.coupling = _CrossTaskCoupling()
        else:
            self.coupling = None

        self._init_heads()

    def _init_heads(self):
        for module in [self.disp_head, self.ori_head, self.zupt_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, 10, 6)
        h = x.permute(0, 2, 1)
        h1 = self.conv1(h)
        h2 = self.conv2(h1)
        h = self.conv_drop(h1 + h2)

        h = h.permute(0, 2, 1)
        h, _ = self.gru(h)

        context = self.attention(h)

        delta_v = self.disp_head(context)
        v_prev_last = x[:, -1, 3:4]
        v_init = torch.clamp(v_prev_last + delta_v, 0.0, 1.0)
        w_init = self.ori_head(context)

        if self.coupling is not None:
            disp, ori = self.coupling(v_init, w_init)
        else:
            disp, ori = v_init, w_init

        zupt = self.zupt_head(context)
        return disp, ori, zupt


# ─── Runtime Switching Container ─────────────────────────────────────────────

class DualSpecialistRuntimeSwitch(nn.Module):
    """
    Combined runtime switching system.
    Evaluates both specialists and blends their predictions linearly based on measured |yaw_rate|:
      - Below (threshold - margin) -> 100% S1 (Straight Specialist)
      - Above (threshold + margin) -> 100% S2 (Turning Specialist)
      - Within [threshold - margin, threshold + margin] -> linear blend
      - Over temporal sequences, blends smoothly with rate-limiting / ~0.5s transition time constant.
    """

    def __init__(
        self,
        model_s1: StraightSpecialistS1,
        model_s2: TurningSpecialistS2,
        threshold_yaw_rate: float = 0.03285,  # 60th percentile (~1.88 deg/s)
        transition_margin: float = 0.0050,    # transition band margin (+- 0.29 deg/s)
        fade_time_s: float = 0.5,             # linear transition duration
    ):
        super().__init__()
        self.model_s1 = model_s1
        self.model_s2 = model_s2
        self.threshold = threshold_yaw_rate
        self.margin = transition_margin
        self.fade_time_s = fade_time_s
        self.current_alpha = 0.0

    def reset_state(self):
        """Reset temporal smoothing state."""
        self.current_alpha = 0.0

    def compute_turn_weight(self, measured_yaw_rate: float, dt: float = 1.0) -> float:
        """
        Computes the trust weight for S2 (0.0 = full S1, 1.0 = full S2).
        Includes static margin interpolation and temporal rate-limiting over fade_time_s.
        """
        low = self.threshold - self.margin
        high = self.threshold + self.margin

        if measured_yaw_rate <= low:
            target_alpha = 0.0
        elif measured_yaw_rate >= high:
            target_alpha = 1.0
        else:
            target_alpha = (measured_yaw_rate - low) / (high - low + 1e-8)

        # Temporal slew-rate limit over fade_time_s (~0.5s)
        max_delta = dt / max(self.fade_time_s, 1e-4)
        if max_delta >= 1.0:
            # If time step dt >= fade_time (e.g. 1.0s step), transition reaches target
            self.current_alpha = target_alpha
        else:
            diff = target_alpha - self.current_alpha
            self.current_alpha = float(np.clip(self.current_alpha + np.clip(diff, -max_delta, max_delta), 0.0, 1.0))

        return self.current_alpha

    def forward(
        self,
        x_6ch: torch.Tensor,
        measured_yaw_rate: Optional[float] = None,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        x_4ch = x_6ch[:, :, :4]
        d1, o1, z1 = self.model_s1(x_4ch)
        d2, o2, z2 = self.model_s2(x_6ch)

        if measured_yaw_rate is None:
            w_scaled = x_6ch[:, :, 1]
            w_phys = w_scaled * 2.0 - 1.0
            measured_yaw_rate = float(torch.mean(torch.abs(w_phys)).item())

        alpha = self.compute_turn_weight(measured_yaw_rate, dt=dt)
        disp = (1.0 - alpha) * d1 + alpha * d2
        ori = (1.0 - alpha) * o1 + alpha * o2
        zupt = (1.0 - alpha) * z1 + alpha * z2
        return disp, ori, zupt, alpha


# ─── Production Kinematic Turn Estimator ──────────────────────────────────────

class KinematicTurnEstimator:
    """
    Production-grade Kinematic Turn Estimator with Learned Model Gating & Hysteresis Lock.

    Solves the highway vibration trap where sensor noise floor (~0.065 rad/s)
    falsely triggers cornering models on straight roads.

    Features:
      1. Uses learned temporal orientation signal w_pred_s1 from S1 (which filters out 10-second vibration).
      2. Schmitt-trigger hysteresis (tau_enter = 0.011 rad/s, tau_exit = 0.007 rad/s).
      3. Smooth Hermite S-curve blending: alpha_blend = 3*alpha^2 - 2*alpha^3.
    """

    def __init__(
        self,
        tau_enter: float = 0.011,  # ~0.63 deg/s
        tau_exit: float = 0.007,   # ~0.40 deg/s
        fade_time_s: float = 0.5,
    ):
        self.tau_enter = tau_enter
        self.tau_exit = tau_exit
        self.fade_time_s = fade_time_s
        self.is_turning = False
        self.alpha = 0.0

    def reset(self):
        self.is_turning = False
        self.alpha = 0.0

    def update(
        self,
        ch_w_yaw: np.ndarray,
        ch_a_lat: np.ndarray,
        v_prev: float,
        w_pred_s1: float = 0.0,
        dt: float = 1.0,
    ) -> Tuple[bool, float, float]:
        """
        Updates turn state and returns (is_turning, alpha, turn_metric).
        """
        turn_metric = float(abs(w_pred_s1))

        # Schmitt-trigger hysteresis lock
        if not self.is_turning:
            if turn_metric >= self.tau_enter:
                self.is_turning = True
        else:
            if turn_metric <= self.tau_exit:
                self.is_turning = False

        # Target alpha and slew-rate limiting (0.5s fade time)
        target_alpha = 1.0 if self.is_turning else 0.0
        max_step = dt / max(self.fade_time_s, 1e-4)

        if max_step >= 1.0:
            self.alpha = target_alpha
        else:
            diff = target_alpha - self.alpha
            self.alpha = float(np.clip(self.alpha + np.clip(diff, -max_step, max_step), 0.0, 1.0))

        # Hermite smooth step (continuous first derivative)
        smooth_alpha = float(3.0 * (self.alpha ** 2) - 2.0 * (self.alpha ** 3))
        return self.is_turning, smooth_alpha, turn_metric


# ─── Production Dual-Specialist Fusion Container ─────────────────────────────

class ProductionDualSpecialistFusion(nn.Module):
    """
    Production-grade Dual-Specialist Fusion Container.
    Combines:
      - Straight Specialist S1 (v3 architecture & weights)
      - Turning Specialist S2 (v4 Ablation-D architecture & weights)
      - KinematicTurnEstimator with noise-rejecting hysteresis
      - Smooth Hermite blending across displacement, orientation, and ZUPT
    """

    def __init__(
        self,
        model_s1: StraightSpecialistS1,
        model_s2: TurningSpecialistS2,
        tau_enter: float = 0.011,
        tau_exit: float = 0.007,
        fade_time_s: float = 0.5,
    ):
        super().__init__()
        self.model_s1 = model_s1
        self.model_s2 = model_s2
        self.turn_estimator = KinematicTurnEstimator(
            tau_enter=tau_enter,
            tau_exit=tau_exit,
            fade_time_s=fade_time_s,
        )

    def reset_state(self):
        self.turn_estimator.reset()

    def forward(
        self,
        x_6ch: torch.Tensor,
        ch_w_yaw_phys: np.ndarray,
        ch_a_lat_phys: np.ndarray,
        v_prev_phys: float,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
        """
        Runs dual-specialist forward pass with kinematic turn estimation.
        """
        x_4ch = x_6ch[:, :, :4]

        # Evaluate S1 preliminary predictions
        d1, o1, z1 = self.model_s1(x_4ch)

        # Unscale orientation prediction to physical rad/s
        w_pred_s1 = float(abs((o1.item() - 0.504565) / 0.49350372))
        is_turning, alpha, turn_metric = self.turn_estimator.update(
            ch_w_yaw=ch_w_yaw_phys,
            ch_a_lat=ch_a_lat_phys,
            v_prev=v_prev_phys,
            w_pred_s1=w_pred_s1,
            dt=dt,
        )

        # Evaluate S2 only if in turn or transitional zone (saves compute on highway)
        if alpha > 0.01:
            d2, o2, z2 = self.model_s2(x_6ch)
        else:
            d2, o2, z2 = d1, o1, z1

        # Smooth state fusion
        disp = (1.0 - alpha) * d1 + alpha * d2
        ori = (1.0 - alpha) * o1 + alpha * o2
        zupt = (1.0 - alpha) * z1 + alpha * z2

        return disp, ori, zupt, alpha, is_turning


# ─── Export Utilities ────────────────────────────────────────────────────────

def export_onnx_specialist(model: nn.Module, output_path: str, in_channels: int, device: torch.device):
    """Export an individual specialist to ONNX."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 10, in_channels, device=device)
    try:
        torch.onnx.export(
            model, dummy, output_path,
            export_params=True, opset_version=18,
            do_constant_folding=True,
            input_names=["imu_window"],
            output_names=["displacement", "orientation", "zupt_logit"],
            dynamic_axes={"imu_window": {0: "batch"}, "displacement": {0: "batch"},
                          "orientation": {0: "batch"}, "zupt_logit": {0: "batch"}},
        )
        print(f"[v7][Export] ONNX: {output_path}")
    except Exception as e:
        print(f"[v7][Export] ONNX failed: {e}")


def export_torchscript_specialist(model: nn.Module, output_path: str, in_channels: int, device: torch.device):
    """Export an individual specialist to TorchScript."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 10, in_channels, device=device)
    try:
        traced = torch.jit.trace(model, dummy)
        traced.save(output_path)
        print(f"[v7][Export] TorchScript: {output_path}")
    except Exception as e:
        print(f"[v7][Export] TorchScript failed: {e}")


if __name__ == "__main__":
    s1 = StraightSpecialistS1()
    s2 = TurningSpecialistS2()
    n_s1 = sum(p.numel() for p in s1.parameters() if p.requires_grad)
    n_s2 = sum(p.numel() for p in s2.parameters() if p.requires_grad)
    print(f"S1 parameters: {n_s1:,} (~21.7k)")
    print(f"S2 parameters: {n_s2:,} (~21.9k)")
    switch = DualSpecialistRuntimeSwitch(s1, s2)
    dummy = torch.randn(2, 10, 6)
    d, o, z, a = switch(dummy)
    print(f"Runtime switch output: disp={d.shape}, ori={o.shape}, zupt={z.shape}, alpha={a:.3f}")
