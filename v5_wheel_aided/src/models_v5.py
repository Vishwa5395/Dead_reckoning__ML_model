"""
models_v5.py
------------
PINO-DR v5 neural architecture supporting 7-channel inputs (including rear wheel speed).

Key components:
- 7-channel Conv1D stem with dilated receptive field (kernel 3, dilation 1 & 2).
- BiGRU (hidden_size=32/dir -> 64-dim sequence representations).
- 2-Head Directional Temporal Attention:
    * Head 1 (q_fwd): attends over forward causal states (steady kinematics).
    * Head 2 (q_bwd): attends over backward anti-causal states (transient cornering).
- Multi-Task Heads: Displacement (Δv), Orientation (ω), and ZUPT (stationary logit).
- Predicted Cross-Task Coupling: 2 -> 16 -> 2 residual MLP coupling Δv and ω.
- Mobile footprint: ~22,037 parameters (<= 25,000 mobile budget).
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int = 3, dilation: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=kernel, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DirectionalTwoHeadTemporalAttention(nn.Module):
    """
    2-Head Directional Temporal Attention over BiGRU outputs.
    BiGRU output: (B, T, 64) -> [fwd: 32 dims, bwd: 32 dims]
    - Head 1 (q_fwd, 32): attends over forward causal states (steady motion).
    - Head 2 (q_bwd, 32): attends over backward anti-causal states (transients).
    """

    def __init__(self, gru_hidden: int = 32):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.q_fwd = nn.Parameter(torch.randn(gru_hidden) * 0.02)
        self.q_bwd = nn.Parameter(torch.randn(gru_hidden) * 0.02)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (B, T, 2 * gru_hidden)
        H_fwd = H[:, :, :self.gru_hidden]
        H_bwd = H[:, :, self.gru_hidden:]

        # Attention weights
        score_fwd = torch.einsum("bte,e->bt", H_fwd, self.q_fwd) / (self.gru_hidden ** 0.5)
        score_bwd = torch.einsum("bte,e->bt", H_bwd, self.q_bwd) / (self.gru_hidden ** 0.5)

        w_fwd = F.softmax(score_fwd, dim=1).unsqueeze(-1)
        w_bwd = F.softmax(score_bwd, dim=1).unsqueeze(-1)

        ctx_fwd = torch.sum(H_fwd * w_fwd, dim=1)
        ctx_bwd = torch.sum(H_bwd * w_bwd, dim=1)

        return torch.cat([ctx_fwd, ctx_bwd], dim=-1)  # (B, 64)


class SingleTemporalAttention(nn.Module):
    """Single-head attention baseline."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("bte,e->bt", H, self.query) / (H.shape[-1] ** 0.5)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(H * weights, dim=1)


class PredictedCrossTaskCoupling(nn.Module):
    """
    Lightweight residual MLP (2 -> 16 -> 2) coupling velocity and yaw rate.
    Initialized with zeros so it acts as identity link at initialization.
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        # Initialize final layer weights and bias to exact zero
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, v_init: torch.Tensor, w_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([v_init, w_init], dim=-1)  # (B, 2)
        delta = self.mlp(x)                      # (B, 2)
        v_out = v_init + delta[:, 0:1]
        w_out = w_init + delta[:, 1:2]
        return v_out, w_out


class PINODeadReckoningNetV5(nn.Module):
    """
    PINO-DR v5 Architecture:
    Conv1D -> BiGRU -> 2-Head Directional Attention -> MLP Heads -> Predicted Coupling
    """

    def __init__(
        self,
        in_channels: int = 7,
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

        # Conv1D stem (input: batch, channels, seq_len)
        self.conv1 = _ConvBlock(in_channels, conv_channels, kernel=3, dilation=1, padding=1)
        self.conv2 = _ConvBlock(conv_channels, conv_channels, kernel=3, dilation=2, padding=2)
        self.conv_drop = nn.Dropout(dropout)

        # BiGRU
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.gru_drop = nn.Dropout(dropout)

        gru_out_dim = gru_hidden * 2  # 64

        # Attention pooling
        if use_multihead_attention:
            self.attn = DirectionalTwoHeadTemporalAttention(gru_hidden=gru_hidden)
        else:
            self.attn = SingleTemporalAttention(hidden_dim=gru_out_dim)

        # Heads
        self.disp_head = nn.Sequential(
            nn.Linear(gru_out_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self.ori_head = nn.Sequential(
            nn.Linear(gru_out_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self.zupt_head = nn.Sequential(
            nn.Linear(gru_out_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        # Cross-Task Coupling
        if use_cross_task_coupling:
            self.coupling = PredictedCrossTaskCoupling(hidden_dim=16)
        else:
            self.coupling = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T=10, C=7) -> transpose for Conv1d: (B, C, T)
        x_conv = x.transpose(1, 2)
        c1 = self.conv1(x_conv)
        c2 = self.conv2(c1) + c1  # residual add
        c2 = self.conv_drop(c2)

        # BiGRU: (B, T, C)
        h_gru, _ = self.gru(c2.transpose(1, 2))
        h_gru = self.gru_drop(h_gru)

        # Attention Context: (B, 64)
        context = self.attn(h_gru)

        # Task Heads
        v_init = self.disp_head(context)           # (B, 1) - delta velocity update
        w_init = self.ori_head(context)            # (B, 1) - yaw rate

        # Cross-Task Coupling
        if self.coupling is not None:
            disp, ori = self.coupling(v_init, w_init)
        else:
            disp, ori = v_init, w_init

        zupt = self.zupt_head(context)             # (B, 1) - stationary logit

        return disp, ori, zupt


# ─── Export Utilities ────────────────────────────────────────────────────────

def export_onnx_v5(model: PINODeadReckoningNetV5, output_path: str, device: torch.device, in_channels: int = 7):
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
        print(f"[v5][Export] ONNX: {output_path}")
    except Exception as e:
        print(f"[v5][Export] ONNX failed: {e}")


def export_torchscript_v5(model: PINODeadReckoningNetV5, output_path: str, device: torch.device, in_channels: int = 7):
    """Export model to TorchScript format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 10, in_channels, device=device)
    try:
        traced = torch.jit.trace(model, dummy)
        traced.save(output_path)
        print(f"[v5][Export] TorchScript: {output_path}")
    except Exception as e:
        print(f"[v5][Export] TorchScript failed: {e}")
