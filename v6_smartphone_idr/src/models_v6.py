"""
models_v6.py
------------
PINO-DR v6: Physics-Informed Neural Odometry with Smartphone-Only IMU.

Architecture (reconstructed from best_model_v6.pth checkpoint):
  - Input: (B, 20, 6) 10 Hz window, 6 channels
      [a_fwd, w_yaw, a_lat, v_prev, w_yaw_accel, centripetal_res]
  - Conv1D stem: 2 residual conv blocks (32 channels) -> BatchNorm -> GELU
  - BiGRU: hidden 32 (bidirectional -> output 64)
  - Directional 2-head temporal attention (forward/backward sub-spaces)
  - Multi-task heads:
      head_dv            -> delta-v prediction (64 -> 28 -> 1)
      head_w             -> yaw rate prediction (64 -> 28 -> 1)
      head_zupt          -> standstill logit (64 -> 16 -> 1)
      head_bias_accel    -> accel bias residual (64 -> 16 -> 1)
      head_bias_gyro     -> gyro bias residual (64 -> 16 -> 1)
      head_uncertainty   -> per-task log-variance (64 -> 16 -> 2)
  - Cross-task coupling MLP (2 -> 16 -> 2, zero-initialized identity)

Parameter budget: ~24,601 (<= 25,000 mobile budget).
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
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=padding,
                              dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self._init()

    def _init(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _DirectionalMultiHeadTemporalAttention(nn.Module):
    """
    2-head temporal attention over the BiGRU output.

    Head 1 attends over the forward-GRU half (steady kinematic accumulation).
    Head 2 attends over the backward-GRU half (turn initiation / transients).
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        assert feat_dim % 2 == 0, "feat_dim must be even for directional split"
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
    Mutual coupling between preliminary delta-v and yaw-rate predictions.
    Strictly uses model predictions (zero train/inference mismatch).
    Zero-initialized so at initialization delta = 0 (identity residual).
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

    def forward(self, v_init: torch.Tensor,
                w_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([v_init, w_init], dim=-1)  # (B, 2)
        delta = self.coupling(inp)                  # (B, 2)
        dv = v_init + delta[:, 0:1]
        w = w_init + delta[:, 1:2]
        return dv, w


# ─── Main V6 Model ──────────────────────────────────────────────────────────

class PINODeadReckoningNetV6(nn.Module):
    """
    Physics-Informed Neural Odometry v6 (Smartphone-Only).

    Args:
        in_channels:  Number of input channels (default 6).
        conv_channels: Channels in conv stem (default 32).
        gru_hidden:    Hidden size per direction in BiGRU (default 32 -> 64).
        num_gru_layers: Number of GRU layers (default 1).
        dropout:       Dropout rate (default 0.15).
    """

    def __init__(
        self,
        in_channels: int = 6,
        conv_channels: int = 32,
        gru_hidden: int = 32,
        num_gru_layers: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.gru_hidden = gru_hidden
        self.num_gru_layers = num_gru_layers
        self.dropout = dropout

        # ── Conv1D stem ──
        self.conv1 = _ConvBlock(in_channels, conv_channels, kernel=3,
                                dilation=1, padding=1)
        self.conv2 = _ConvBlock(conv_channels, conv_channels, kernel=3,
                                dilation=2, padding=2)
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
        self.attention = _DirectionalMultiHeadTemporalAttention(gru_out_dim)

        # ── Heads ──
        self.head_dv = nn.Sequential(
            nn.Linear(gru_out_dim, 28),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(28, 1),
        )
        self.head_w = nn.Sequential(
            nn.Linear(gru_out_dim, 28),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(28, 1),
        )
        self.head_zupt = nn.Sequential(
            nn.Linear(gru_out_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self.head_bias_accel = nn.Sequential(
            nn.Linear(gru_out_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self.head_bias_gyro = nn.Sequential(
            nn.Linear(gru_out_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self.head_uncertainty = nn.Sequential(
            nn.Linear(gru_out_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2),
        )

        # ── Cross-Task Coupling ──
        self.coupling = _CrossTaskCoupling()

        self._init_heads()

    def _init_heads(self):
        for module in [
            self.head_dv, self.head_w, self.head_zupt,
            self.head_bias_accel, self.head_bias_gyro,
            self.head_uncertainty,
        ]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """
        Args:
            x: (batch, seq_len=20, channels=6) temporal input tensor.

        Returns:
            dv_pred:    (batch, 1) delta-v prediction (scaled [0, 1]).
            w_pred:     (batch, 1) yaw-rate prediction (scaled [0, 1]).
            z_logit:    (batch, 1) raw ZUPT logit.
            ba_pred:    (batch, 1) accel bias residual (scaled [0, 1]).
            bw_pred:    (batch, 1) gyro bias residual (scaled [0, 1]).
            log_var:    (batch, 2) per-task log-variance (dv, w).
        """
        # Conv1D expects (batch, channels, seq_len)
        h = x.permute(0, 2, 1)  # (B, C, 20)

        h1 = self.conv1(h)      # (B, 32, 20)
        h2 = self.conv2(h1)     # (B, 32, 20)
        h = h1 + h2             # Residual connection
        h = self.conv_drop(h)

        # GRU expects (batch, seq_len, features)
        h = h.permute(0, 2, 1)  # (B, 20, 32)
        h, _ = self.gru(h)      # (B, 20, 64)

        context = self.attention(h)  # (B, 64)

        dv_init = self.head_dv(context)      # (B, 1)
        w_init = self.head_w(context)        # (B, 1)

        dv, w = self.coupling(dv_init, w_init)
        dv = torch.clamp(dv, 0.0, 1.0)       # delta-v lives in [0, 1] scaled space
        w = torch.clamp(w, 0.0, 1.0)

        z_logit = self.head_zupt(context)    # (B, 1)
        ba_pred = self.head_bias_accel(context)  # (B, 1)
        bw_pred = self.head_bias_gyro(context)   # (B, 1)
        log_var = self.head_uncertainty(context) # (B, 2)

        return dv, w, z_logit, ba_pred, bw_pred, log_var


# ─── Export Utilities ────────────────────────────────────────────────────────

def export_onnx_v6(model: PINODeadReckoningNetV6, output_path: str,
                   device: torch.device, in_channels: int = 6,
                   seq_len: int = 20):
    """Export model to ONNX format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    dummy = torch.randn(1, seq_len, in_channels, device=device)
    try:
        torch.onnx.export(
            model, dummy, output_path,
            export_params=True, opset_version=18,
            do_constant_folding=True,
            dynamo=False,
            input_names=["imu_window"],
            output_names=["delta_v", "yaw_rate", "zupt_logit",
                          "bias_accel", "bias_gyro", "log_var"],
            dynamic_axes={
                "imu_window": {0: "batch"},
                "delta_v": {0: "batch"}, "yaw_rate": {0: "batch"},
                "zupt_logit": {0: "batch"}, "bias_accel": {0: "batch"},
                "bias_gyro": {0: "batch"}, "log_var": {0: "batch"},
            },
        )
        print(f"[v6][Export] ONNX: {output_path}")
    except Exception as e:
        print(f"[v6][Export] ONNX failed: {e}")


def export_torchscript_v6(model: PINODeadReckoningNetV6, output_path: str,
                          device: torch.device, in_channels: int = 6,
                          seq_len: int = 20):
    """Export model to TorchScript format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    dummy = torch.randn(1, seq_len, in_channels, device=device)
    try:
        traced = torch.jit.trace(model, dummy)
        traced.save(output_path)
        print(f"[v6][Export] TorchScript: {output_path}")
    except Exception as e:
        print(f"[v6][Export] TorchScript failed: {e}")


if __name__ == "__main__":
    model = PINODeadReckoningNetV6(in_channels=6)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PINO-DR v6 parameters: {n_params:,}")
    dummy = torch.randn(2, 20, 6)
    d, w, z, ba, bw, lv = model(dummy)
    print(f"Output shapes: dv={d.shape}, w={w.shape}, zupt={z.shape}, "
          f"b_a={ba.shape}, b_w={bw.shape}, log_var={lv.shape}")
