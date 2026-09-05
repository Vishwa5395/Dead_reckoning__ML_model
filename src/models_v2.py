"""
models_v2.py
------------
Improved, production-grade Input Delay Neural Network (IDNN) architectures for:
1. Displacement Estimation (closed-loop feedback model)
2. Orientation Rate Estimation (angular velocity model)

Upgrades over the original 2x32 IDNN:
- Three hidden layers (64 -> 64 -> 32) for added capacity.
- BatchNorm1d after every hidden layer for stable, faster convergence.
- GELU activation (smooth, better gradient flow) instead of ReLU.
- He/Kaiming weight initialisation and zero bias init.
- Dropout retained as a regulariser (10-20%).

The models remain fully-differentiable MLPs suitable for ONNX / TorchScript
export, preserving the IDNN "input-delay taps" design of Onyekpe et al. (2021).
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn


class _IDNNBlock(nn.Module):
    """Linear -> BatchNorm -> GELU -> Dropout block."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.linear.weight, a=math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu")
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.linear(x))))


class DisplacementIDNNV2(nn.Module):
    """Displacement predictor. Input: 10 accel + 10 feedback displacement = 20."""

    def __init__(self, input_dim: int = 20, hidden_dims=(64, 64, 32), dropout: float = 0.20):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(_IDNNBlock(prev, h, dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


class OrientationIDNNV2(nn.Module):
    """Orientation (yaw-rate) predictor. Input: 10 gravity-aligned yaw rates."""

    def __init__(self, input_dim: int = 10, hidden_dims=(64, 64, 32), dropout: float = 0.20):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(_IDNNBlock(prev, h, dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


def export_to_onnx_v2(model: nn.Module, dummy_input: torch.Tensor, output_path: str,
                      input_name: str = "imu_input", output_name: str = "prediction") -> None:
    """Export a model to ONNX and TorchScript for edge/mobile deployment."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    ts_path = output_path.replace(".onnx", "_torchscript.pt")
    try:
        traced = torch.jit.trace(model, dummy_input)
        traced.save(ts_path)
        print(f"[v2][Export] Saved TorchScript: {ts_path}")
    except Exception as e:  # pragma: no cover
        print(f"[v2][Export] TorchScript failed: {e}")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes={input_name: {0: "batch_size"}, output_name: {0: "batch_size"}},
        )
        print(f"[v2][Export] Saved ONNX: {output_path}")
    except Exception as e:  # pragma: no cover
        print(f"[v2][Export] ONNX failed: {e}")


if __name__ == "__main__":
    d = DisplacementIDNNV2()
    o = OrientationIDNNV2()
    print("Displacement params:", sum(p.numel() for p in d.parameters() if p.requires_grad))
    print("Orientation params:", sum(p.numel() for p in o.parameters() if p.requires_grad))
