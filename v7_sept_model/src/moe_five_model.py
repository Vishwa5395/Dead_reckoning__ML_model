"""
moe_five_model.py
-----------------
Supreme 5-Expert Neural Mixture-of-Experts (MoE) Architecture for PINO-DR v7.

Key Design Elements:
1. 5 Recurrent Neural Experts:
   - Expert 0: Motorway Cruising Specialist (4-channel: a_fwd, w_yaw, a_lat, v_prev; optimized for high-speed zero drift)
   - Expert 1: Roundabout Specialist (6-channel: BiGRU + 2-Head Attention + Coupling for sustained circular curvature)
   - Expert 2: Quick Accel Specialist (6-channel: forward thrust tracking & acceleration dynamics)
   - Expert 3: Hard Brake Specialist (4-channel: rapid deceleration & sensitive ZUPT standstill gating)
   - Expert 4: Sharp Turns Specialist (6-channel: directional attention over sharp urban cornering transients)

2. Physical Neural Gating Router:
   - Input: 20 kinematic physical features extracted from the 10-step IMU window (deviations from physical centers, speeds, accelerations, angular rates, centripetal residuals).
   - 2-layer MLP with LayerNorm, GELU, and Dropout.
   - Output: 5 continuous softmax routing weights g = [g0, g1, g2, g3, g4] with sum(g) = 1.0.

3. Differentiable Soft Mixture:
   - y_disp = sum(g_i * d_i)
   - y_ori  = sum(g_i * o_i)
   - y_zupt = sum(g_i * z_i)
   Eliminates all discrete switching friction and trajectory discontinuities.

4. 100% Autonomous & Production Ready:
   - Pure neural network computation from IMU inputs. Zero scenario labels, zero test-set multipliers.
   - Single forward pass execution: model(imu_tensor) -> (disp, ori, zupt, router_weights).
   - Exportable directly to ONNX (< 2.5 MB) and TorchScript (< 800 KB).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Blocks for Experts ─────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Conv1D -> BatchNorm -> GELU."""

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


class _SingleTemporalAttention(nn.Module):
    """Single-head temporal attention over GRU hidden states."""

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
    """2-head directional temporal attention over forward and backward GRU halves."""

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

        return torch.cat([c_fwd, c_bwd], dim=-1)


class _CrossTaskCoupling(nn.Module):
    """Mutual coupling between preliminary displacement and orientation predictions."""

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


# ─── Expert 4-Channel Architecture (Cruising / Highway & Decel) ──────────────

class StraightExpertNetwork(nn.Module):
    """
    Specialist Expert for straight-line cruising and braking (21,667 params).
    Input: (batch, 10, 4) -> [a_fwd, w_yaw, a_lat, v_prev].
    Matches StraightSpecialistS1 from v3_pino_dr.
    """

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels: int = 32,
        gru_hidden: int = 32,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.conv1 = _ConvBlock(in_channels, conv_channels, kernel=3, dilation=1, padding=1)
        self.conv2 = _ConvBlock(conv_channels, conv_channels, kernel=3, dilation=2, padding=2)
        self.conv_drop = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len=10, 4)
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


# ─── Expert 6-Channel Architecture (Roundabout, Quick Accel, Sharp Turns) ─────

class TurningExpertNetwork(nn.Module):
    """
    Specialist Expert for dynamic turning and acceleration maneuvers (21,941 params).
    Input: (batch, 10, 6) -> [a_fwd, w_yaw, a_lat, v_prev, w_yaw_accel, centripetal_residual].
    Matches TurningSpecialistS2 from v4_turn_focused (Ablation D).
    """

    def __init__(
        self,
        in_channels: int = 6,
        conv_channels: int = 32,
        gru_hidden: int = 32,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.conv1 = _ConvBlock(in_channels, conv_channels, kernel=3, dilation=1, padding=1)
        self.conv2 = _ConvBlock(conv_channels, conv_channels, kernel=3, dilation=2, padding=2)
        self.conv_drop = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        gru_out_dim = gru_hidden * 2  # 64
        self.attention = _DirectionalMultiHeadTemporalAttention(gru_out_dim)

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
        self.coupling = _CrossTaskCoupling()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len=10, 6)
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

        disp, ori = self.coupling(v_init, w_init)
        zupt = self.zupt_head(context)
        return disp, ori, zupt


# ─── Physical Neural Gating Router ───────────────────────────────────────────

class PhysicalNeuralRouter(nn.Module):
    """
    Physical Neural Gating Router that continuously maps 10-step IMU signals
    to continuous mixture-of-experts affinities across the 5 driving regimes.
    """

    def __init__(self, in_features: int = 20, num_experts: int = 5, hidden_dim: int = 64):
        super().__init__()
        # Normalization centers for the 6 channels (scaled value of 0.0 in physical units)
        self.register_buffer(
            "centers",
            torch.tensor([0.52882564, 0.50000000, 0.52409708, 0.00000000, 0.50000000, 0.50000000]),
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_experts),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len=10, channels=6)
        dev = x - self.centers
        speed = x[:, :, 3]
        v_mean = speed.mean(dim=1, keepdim=True)
        v_last = speed[:, -1:]

        abs_yaw = torch.abs(dev[:, :, 1])
        yaw_mean = abs_yaw.mean(dim=1, keepdim=True)
        yaw_max = abs_yaw.max(dim=1, keepdim=True).values
        yaw_last = abs_yaw[:, -1:]

        abs_alat = torch.abs(dev[:, :, 2])
        alat_mean = abs_alat.mean(dim=1, keepdim=True)
        alat_max = abs_alat.max(dim=1, keepdim=True).values
        alat_last = abs_alat[:, -1:]

        fwd_accel = dev[:, :, 0]
        acc_mean = fwd_accel.mean(dim=1, keepdim=True)
        acc_last = fwd_accel[:, -1:]
        acc_pos_max = torch.clamp(fwd_accel, min=0.0).max(dim=1, keepdim=True).values
        acc_neg_max = torch.clamp(-fwd_accel, min=0.0).max(dim=1, keepdim=True).values

        abs_yacc = torch.abs(dev[:, :, 4])
        yacc_mean = abs_yacc.mean(dim=1, keepdim=True)
        yacc_max = abs_yacc.max(dim=1, keepdim=True).values

        abs_cent = torch.abs(dev[:, :, 5])
        cent_mean = abs_cent.mean(dim=1, keepdim=True)
        cent_max = abs_cent.max(dim=1, keepdim=True).values

        raw_mean = dev.mean(dim=1)
        raw_std = dev.std(dim=1, unbiased=False)

        feats = torch.cat([
            v_mean, v_last,
            yaw_mean, yaw_max, yaw_last,
            alat_mean, alat_max, alat_last,
            acc_mean, acc_last, acc_pos_max, acc_neg_max,
            yacc_mean, yacc_max,
            cent_mean, cent_max,
            raw_mean[:, :2], raw_std[:, :2],
        ], dim=-1)
        return feats

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.mlp(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        logits = self.mlp(feats)
        return torch.softmax(logits, dim=-1)


# ─── Supreme 5-Expert Mixture-of-Experts ──────────────────────────────────────

class SupremeMoENet(nn.Module):
    """
    Supreme 5-Expert Neural Mixture-of-Experts for PINO-DR v7.

    Contains:
      - exp0: Motorway Specialist (4-channel BiGRU + Attention)
      - exp1: Roundabout Specialist (6-channel BiGRU + Directional Attention + Coupling)
      - exp2: Quick Accel Specialist (6-channel BiGRU + Directional Attention + Coupling)
      - exp3: Hard Brake Specialist (4-channel BiGRU + Attention)
      - exp4: Sharp Turns Specialist (6-channel BiGRU + Directional Attention + Coupling)
      - gating: PhysicalNeuralRouter (20 physical kinematic features -> 5 affinities)
    """

    EXPERT_NAMES = [
        "motorway",     # Expert 0
        "roundabout",   # Expert 1
        "quick_accel",  # Expert 2
        "hard_brake",   # Expert 3
        "sharp_turns",  # Expert 4
    ]

    def __init__(self):
        super().__init__()
        self.exp0 = StraightExpertNetwork(in_channels=4)
        self.exp1 = TurningExpertNetwork(in_channels=6)
        self.exp2 = TurningExpertNetwork(in_channels=6)
        self.exp3 = StraightExpertNetwork(in_channels=4)
        self.exp4 = TurningExpertNetwork(in_channels=6)
        self.gating = PhysicalNeuralRouter(in_features=20, num_experts=5, hidden_dim=64)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Differentiable forward pass.

        Args:
            x: (batch, 10, 6) input tensor
            temperature: Softmax temperature

        Returns:
            d_pred: (batch, 1) blended displacement prediction
            o_pred: (batch, 1) blended orientation rate prediction
            z_pred: (batch, 1) blended ZUPT logits
            weights: (batch, 5) continuous router weights
        """
        weights = self.gating(x)  # (batch, 5)

        x4 = x[:, :, :4]
        d0, o0, z0 = self.exp0(x4)
        d1, o1, z1 = self.exp1(x)
        d2, o2, z2 = self.exp2(x)
        d3, o3, z3 = self.exp3(x4)
        d4, o4, z4 = self.exp4(x)

        w0 = weights[:, 0:1]
        w1 = weights[:, 1:2]
        w2 = weights[:, 2:3]
        w3 = weights[:, 3:4]
        w4 = weights[:, 4:5]

        d_pred = w0 * d0 + w1 * d1 + w2 * d2 + w3 * d3 + w4 * d4
        o_pred = w0 * o0 + w1 * o1 + w2 * o2 + w3 * o3 + w4 * o4
        z_pred = w0 * z0 + w1 * z1 + w2 * z2 + w3 * z3 + w4 * z4

        return d_pred, o_pred, z_pred, weights

    def load_expert_weights(
        self,
        v3_ckpt_path: str | Path,
        v4_ckpt_path: str | Path,
        device: torch.device,
    ):
        """Initializes the 5 recurrent neural experts from verified checkpoints."""
        ckpt_v3 = torch.load(v3_ckpt_path, map_location=device, weights_only=False)
        state_v3 = ckpt_v3["model_state_dict"]

        ckpt_v4 = torch.load(v4_ckpt_path, map_location=device, weights_only=False)
        state_v4 = ckpt_v4["model_state_dict"]

        # Exp 0 (Motorway): v3 weights
        self.exp0.load_state_dict(state_v3)
        # Exp 1 (Roundabout): v4-D weights
        self.exp1.load_state_dict(state_v4)
        # Exp 2 (Quick Accel): v4-D weights
        self.exp2.load_state_dict(state_v4)
        # Exp 3 (Hard Brake): v3 weights
        self.exp3.load_state_dict(state_v3)
        # Exp 4 (Sharp Turns): v4-D weights
        self.exp4.load_state_dict(state_v4)

        print("[SupremeMoE] Expert 0 (motorway)    <- v3 (7.13m baseline)")
        print("[SupremeMoE] Expert 1 (roundabout)  <- v4-D (55.37m baseline)")
        print("[SupremeMoE] Expert 2 (quick_accel) <- v4-D (19.58m baseline)")
        print("[SupremeMoE] Expert 3 (hard_brake)  <- v3 (17.15m baseline)")
        print("[SupremeMoE] Expert 4 (sharp_turns) <- v4-D (36.39m baseline)")


# ─── Export Helpers for Mobile App Integration ───────────────────────────────

def export_onnx_moe(model: SupremeMoENet, out_path: Path, device: torch.device):
    """Exports SupremeMoENet to ONNX for mobile deployment (iOS CoreML / Android ONNX Runtime)."""
    model.eval()
    dummy_input = torch.randn(1, 10, 6, dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["imu_input_6ch"],
        output_names=["disp_pred", "ori_pred", "zupt_logits", "router_weights"],
        dynamic_axes={"imu_input_6ch": {0: "batch_size"}},
        opset_version=14,
        dynamo=False,
    )
    print(f"[Export][ONNX] Saved -> {out_path.name}")


def export_torchscript_moe(model: SupremeMoENet, out_path: Path, device: torch.device):
    """Exports SupremeMoENet to TorchScript."""
    model.eval()
    dummy_input = torch.randn(1, 10, 6, dtype=torch.float32, device=device)
    traced = torch.jit.trace(model, dummy_input)
    traced.save(str(out_path))
    print(f"[Export][TorchScript] Saved -> {out_path.name}")
