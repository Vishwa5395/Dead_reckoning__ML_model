"""
models_v4.py
------------
PINO-DR v4: Physics-Informed Neural Odometry with Turn-Aware Dynamics.

Key architectural innovations:
1. Flexible Input Channels (4, 5, or 6):
   Supports [a_fwd, w_yaw, a_lat, v_prev, (w_yaw_accel), (centripetal_residual)]
   for full ablation sequence.
2. Directional 2-Head Temporal Attention:
   - Head 1: Attends over causal forward GRU states (steady kinematic accumulation).
   - Head 2: Attends over anti-causal backward GRU states (turn transition dynamics).
   Parameter-efficient: splits 64-dim hidden state into two 32-dim sub-spaces with zero projection overhead.
3. Predicted Cross-Task Coupling:
   Lightweight residual MLP that allows predicted displacement and orientation to mutually
   condition each other strictly using model predictions (zero train/inference mismatch).
   Zero-initialized so it starts as a pure identity bypass.
4. Compact Mobile Footprint:
   ~21,941 parameters (well below the <= 25,000 budget).
"""

from __future__ import annotations

import math
import os
import sys
from typing import Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn


# ─── Building Blocks ─────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Conv1D → BatchNorm → GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 dilation: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self._init()

    def _init(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _SingleTemporalAttention(nn.Module):
    """Standard learnable query attention pooling over time dimension."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(feat_dim))
        self.scale = math.sqrt(feat_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (batch, seq_len, feat_dim)
        scores = torch.matmul(h, self.query) / self.scale  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        return (h * weights).sum(dim=1)  # (batch, feat_dim)


class _DirectionalMultiHeadTemporalAttention(nn.Module):
    """
    2-head temporal attention:
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
        # h: (batch, seq_len, feat_dim)
        h_fwd = h[:, :, :self.half_dim]
        h_bwd = h[:, :, self.half_dim:]

        scores_fwd = torch.matmul(h_fwd, self.query_fwd) / self.scale
        w_fwd = torch.softmax(scores_fwd, dim=1).unsqueeze(-1)
        c_fwd = (h_fwd * w_fwd).sum(dim=1)

        scores_bwd = torch.matmul(h_bwd, self.query_bwd) / self.scale
        w_bwd = torch.softmax(scores_bwd, dim=1).unsqueeze(-1)
        c_bwd = (h_bwd * w_bwd).sum(dim=1)

        return torch.cat([c_fwd, c_bwd], dim=-1)  # (batch, feat_dim)


class _CrossTaskCoupling(nn.Module):
    """
    Mutual coupling layer between preliminary displacement and orientation predictions.
    Strictly uses model predictions (zero train/inference mismatch).
    Zero-initialized so that at initialization, delta = 0 (identity residual).
    """

    def __init__(self):
        super().__init__()
        self.coupling = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 2),
        )
        # Zero-initialize to start as pure identity
        for m in self.coupling.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, v_init: torch.Tensor, w_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([v_init, w_init], dim=-1)  # (B, 2)
        delta = self.coupling(inp)                  # (B, 2)
        v_coupled = torch.clamp(v_init + delta[:, 0:1], 0.0, 1.0)
        w_coupled = w_init + delta[:, 1:2]
        return v_coupled, w_coupled


# ─── Main V4 Model ──────────────────────────────────────────────────────────

class PINODeadReckoningNetV4(nn.Module):
    """
    Physics-Informed Neural Odometry v4 (Turn-Focused).

    Args:
        in_channels: Number of input channels (4, 5, or 6, default 6).
        conv_channels: Channels in conv stem (default 32).
        gru_hidden: Hidden size per direction in BiGRU (default 32 → output 64).
        num_gru_layers: Number of GRU layers (default 1).
        dropout: Dropout rate (default 0.20).
        use_multihead_attention: Use 2-head directional attention if True, else single.
        use_cross_task_coupling: Use predicted coupling layer if True, else direct.
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

        # ── Conv1D stem (input: batch, channels, seq_len) ──
        self.conv1 = _ConvBlock(in_channels, conv_channels, kernel=3, dilation=1, padding=1)
        self.conv2 = _ConvBlock(conv_channels, conv_channels, kernel=3, dilation=2, padding=2)
        self.conv_drop = nn.Dropout(dropout)

        # ── BiGRU ──
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        gru_out_dim = gru_hidden * 2  # 64

        # ── Temporal Attention ──
        if use_multihead_attention:
            self.attention = _DirectionalMultiHeadTemporalAttention(gru_out_dim)
        else:
            self.attention = _SingleTemporalAttention(gru_out_dim)

        # ── Multi-Task Heads ──
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

        # ── Cross-Task Coupling ──
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
        """
        Args:
            x: (batch, seq_len=10, channels) temporal input tensor.

        Returns:
            disp_pred: (batch, 1) displacement prediction (scaled [0, 1]).
            ori_pred:  (batch, 1) orientation rate prediction (scaled [0, 1]).
            zupt_logit: (batch, 1) raw ZUPT logit.
        """
        # Conv1D expects (batch, channels, seq_len)
        h = x.permute(0, 2, 1)  # (B, C, 10)

        h1 = self.conv1(h)      # (B, 32, 10)
        h2 = self.conv2(h1)     # (B, 32, 10)
        h = h1 + h2             # Residual connection
        h = self.conv_drop(h)

        # GRU expects (batch, seq_len, features)
        h = h.permute(0, 2, 1)  # (B, 10, 32)
        h, _ = self.gru(h)      # (B, 10, 64)

        # Temporal attention pooling
        context = self.attention(h)  # (B, 64)

        # Preliminary predictions via residual velocity link
        delta_v = self.disp_head(context)           # (B, 1) — predicted delta v
        v_prev_last = x[:, -1, 3:4]                 # (B, 1) — channel 3 is always v_prev
        v_init = torch.clamp(v_prev_last + delta_v, 0.0, 1.0)
        w_init = self.ori_head(context)             # (B, 1)

        # Cross-task coupling (if enabled)
        if self.coupling is not None:
            disp, ori = self.coupling(v_init, w_init)
        else:
            disp, ori = v_init, w_init

        zupt = self.zupt_head(context)              # (B, 1)

        return disp, ori, zupt


# ─── Export Utilities ────────────────────────────────────────────────────────

def export_onnx_v4(model: PINODeadReckoningNetV4, output_path: str, device: torch.device, in_channels: int = 6):
    """Export model to ONNX format."""
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
        print(f"[v4][Export] ONNX: {output_path}")
    except Exception as e:
        print(f"[v4][Export] ONNX failed: {e}")


def export_torchscript_v4(model: PINODeadReckoningNetV4, output_path: str, device: torch.device, in_channels: int = 6):
    """Export model to TorchScript format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 10, in_channels, device=device)
    try:
        traced = torch.jit.trace(model, dummy)
        traced.save(output_path)
        print(f"[v4][Export] TorchScript: {output_path}")
    except Exception as e:
        print(f"[v4][Export] TorchScript failed: {e}")


if __name__ == "__main__":
    model = PINODeadReckoningNetV4(in_channels=6)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PINO-DR v4 parameters: {n_params:,}")
    dummy = torch.randn(2, 10, 6)
    d, o, z = model(dummy)
    print(f"Output shapes: disp={d.shape}, ori={o.shape}, zupt={z.shape}")
