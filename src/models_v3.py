"""
models_v3.py
------------
PINO-DR: Physics-Informed Neural Odometry for Dead Reckoning.

Architecture:
  Input (batch, 10, 4) → Conv1D Stem → BiGRU → Temporal Attention → Multi-Task Heads
    - Displacement head: Δs_t (m per 1-second step, range [0, 45])
    - Orientation head: ω_t (rad/s, range [-1.2, 1.2])
    - ZUPT head: p_stop ∈ [0, 1]

Estimated parameters: ~18,500 (compact enough for mobile, with temporal weight sharing).
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


# ─── Building blocks ─────────────────────────────────────────────────────────

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


class _TemporalAttention(nn.Module):
    """Learnable query-based attention pooling over the time dimension."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(feat_dim))
        self.scale = math.sqrt(feat_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (batch, seq_len, feat_dim)
        scores = torch.matmul(h, self.query) / self.scale  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = (h * weights).sum(dim=1)  # (batch, feat_dim)
        return context


# ─── Main model ──────────────────────────────────────────────────────────────

class PINODeadReckoningNet(nn.Module):
    """
    Physics-Informed Neural Odometry for Dead Reckoning.

    Args:
        in_channels: Number of input channels (default 4: a_fwd, w_yaw, a_lat, v_prev).
        conv_channels: Number of channels in the conv stem (default 32).
        gru_hidden: Hidden size per direction in the BiGRU (default 32 → output 64).
        num_gru_layers: Number of GRU layers (default 1).
        dropout: Dropout rate (default 0.20).
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
            dropout=0.0,  # no inter-layer dropout for single layer
        )
        gru_out_dim = gru_hidden * 2  # bidirectional → 2x hidden

        # ── Temporal attention ──
        self.attention = _TemporalAttention(gru_out_dim)

        # ── Task heads ──
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
        """
        Args:
            x: (batch, seq_len=10, channels=4) temporal input tensor.

        Returns:
            disp_pred: (batch, 1) displacement prediction (scaled).
            ori_pred:  (batch, 1) orientation rate prediction (scaled).
            zupt_logit: (batch, 1) raw ZUPT logit (apply sigmoid externally for BCE).
        """
        # Conv1D expects (batch, channels, seq_len)
        h = x.permute(0, 2, 1)  # (B, 4, 10)

        h1 = self.conv1(h)      # (B, 32, 10)
        h2 = self.conv2(h1)     # (B, 32, 10)
        h = h1 + h2             # Residual connection
        h = self.conv_drop(h)

        # GRU expects (batch, seq_len, features)
        h = h.permute(0, 2, 1)  # (B, 10, 32)
        h, _ = self.gru(h)      # (B, 10, 64)

        # Temporal attention pooling
        context = self.attention(h)  # (B, 64)

        # Multi-task heads: residual velocity connection
        delta_v = self.disp_head(context)           # (B, 1) — predicted change in velocity
        v_prev_last = x[:, -1, 3:4]                 # (B, 1) — last known velocity in the window (scaled)
        disp = torch.clamp(v_prev_last + delta_v, 0.0, 1.0)
        ori = self.ori_head(context)                # (B, 1)
        zupt = self.zupt_head(context)              # (B, 1) — raw logit

        return disp, ori, zupt


# ─── Export utilities ─────────────────────────────────────────────────────────

def export_onnx_v3(model: PINODeadReckoningNet, output_path: str, device: torch.device):
    """Export model to ONNX format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 10, 4, device=device)
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
        print(f"[v3][Export] ONNX: {output_path}")
    except Exception as e:
        print(f"[v3][Export] ONNX failed: {e}")


def export_torchscript_v3(model: PINODeadReckoningNet, output_path: str, device: torch.device):
    """Export model to TorchScript format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 10, 4, device=device)
    try:
        traced = torch.jit.trace(model, dummy)
        traced.save(output_path)
        print(f"[v3][Export] TorchScript: {output_path}")
    except Exception as e:
        print(f"[v3][Export] TorchScript failed: {e}")


if __name__ == "__main__":
    model = PINODeadReckoningNet()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PINO-DR parameters: {n_params:,}")
    dummy = torch.randn(2, 10, 4)
    d, o, z = model(dummy)
    print(f"Output shapes: disp={d.shape}, ori={o.shape}, zupt={z.shape}")
