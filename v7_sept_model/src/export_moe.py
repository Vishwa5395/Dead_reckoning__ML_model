"""
export_moe.py
-------------
Exports the trained best_supreme_moe.pth checkpoint to TorchScript and ONNX.
"""

import os
import sys
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"

import sys
WS_ROOT = ROOT.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from v7_sept_model.src.moe_five_model import SupremeMoENet

def export_all():
    device = torch.device("cpu")
    moe_path = CKPT_DIR / "best_supreme_moe.pth"
    print(f"Loading {moe_path}...")
    model = SupremeMoENet().to(device)
    ckpt = torch.load(moe_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 10, 6, dtype=torch.float32, device=device)

    # 1. TorchScript Export
    ts_path = CKPT_DIR / "best_supreme_moe_torchscript.pt"
    try:
        traced = torch.jit.trace(model, dummy_input)
        traced.save(str(ts_path))
        print(f"[Export] TorchScript saved successfully -> {ts_path.name} ({ts_path.stat().st_size:,} bytes)")
    except Exception as e:
        print(f"[Export] TorchScript error: {e}")

    # 2. Standalone Single-File ONNX Export
    onnx_path = CKPT_DIR / "best_supreme_moe.onnx"
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["imu_input_6ch"],
            output_names=["disp_pred", "ori_pred", "zupt_logits", "router_weights"],
            dynamic_axes={"imu_input_6ch": {0: "batch_size"}},
            opset_version=14,
            export_params=True,
            do_constant_folding=True,
            dynamo=False,
        )
        print(f"[Export] ONNX saved successfully -> {onnx_path.name} ({onnx_path.stat().st_size:,} bytes)")
    except Exception as e:
        print(f"[Export] ONNX error: {e}")

if __name__ == "__main__":
    export_all()
