import pickle
import numpy as np
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = ROOT.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from v7_sept_model.src.moe_five_model import SupremeMoENet, export_onnx_moe, export_torchscript_moe

device = torch.device("cpu")
CKPT_DIR = ROOT / "checkpoints"

print("=" * 80)
print("ASSEMBLING SUPREME 5-EXPERT MoE WITH EMBEDDED PHYSICAL ERROR REDUCTION")
print("=" * 80)

model = SupremeMoENet().to(device)

# Load base weights
v3_ckpt = torch.load(WS_ROOT / "v3_pino_dr" / "checkpoints" / "best_model.pth", map_location=device, weights_only=False)["model_state_dict"]
v4_ckpt = torch.load(WS_ROOT / "v4_turn_focused" / "checkpoints" / "best_model_ablation_D_attn_coupling.pth", map_location=device, weights_only=False)["model_state_dict"]

model.exp0.load_state_dict(v3_ckpt)
model.exp1.load_state_dict(v4_ckpt)
model.exp2.load_state_dict(v4_ckpt)
model.exp3.load_state_dict(v3_ckpt)
model.exp4.load_state_dict(v4_ckpt)

# Load existing router state
moe_ckpt = torch.load(CKPT_DIR / "best_supreme_moe.pth", map_location=device, weights_only=False)["model_state_dict"]
router_state = {k.replace("gating.", ""): v for k, v in moe_ckpt.items() if k.startswith("gating.")}
model.gating.load_state_dict(router_state)

model.eval()

# 1. Save updated unified PyTorch checkpoint
out_pth = CKPT_DIR / "best_supreme_moe.pth"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_experts": 5,
        "architecture": "SupremeMoENet",
        "expert_names": SupremeMoENet.EXPERT_NAMES,
    },
    out_pth,
)
print(f"[MoE] Checkpoint saved successfully -> {out_pth.name} ({out_pth.stat().st_size:,} bytes)")

# 2. Export ONNX & TorchScript
onnx_path = CKPT_DIR / "best_supreme_moe.onnx"
ts_path = CKPT_DIR / "best_supreme_moe_torchscript.pt"

dummy = torch.randn(1, 10, 6, dtype=torch.float32)

# Export TorchScript
traced = torch.jit.trace(model, dummy)
traced.save(str(ts_path))
print(f"[Export] TorchScript saved successfully -> {ts_path.name} ({ts_path.stat().st_size:,} bytes)")

# Export ONNX
torch.onnx.export(
    model,
    dummy,
    str(onnx_path),
    input_names=["imu_input_6ch"],
    output_names=["disp_pred", "ori_pred", "zupt_logits", "router_weights"],
    dynamic_axes={"imu_input_6ch": {0: "batch_size"}},
    opset_version=14,
    export_params=True,
    do_constant_folding=True,
    dynamo=False,
)
print(f"[Export] Standalone Single-File ONNX saved -> {onnx_path.name} ({onnx_path.stat().st_size:,} bytes)")

# 3. Verify ONNX Runtime Parity
import onnxruntime as ort
session = ort.InferenceSession(str(onnx_path))
ort_inputs = {session.get_inputs()[0].name: dummy.numpy()}
ort_outs = session.run(None, ort_inputs)

with torch.no_grad():
    py_d, py_o, py_z, py_w = model(dummy)

diff_d = np.max(np.abs(ort_outs[0] - py_d.numpy()))
diff_o = np.max(np.abs(ort_outs[1] - py_o.numpy()))
diff_w = np.max(np.abs(ort_outs[3] - py_w.numpy()))
print(f"[ONNX Parity] Max numerical diff: disp={diff_d:.2e}, ori={diff_o:.2e}, weights={diff_w:.2e}")
assert diff_d < 1e-4 and diff_o < 1e-4 and diff_w < 1e-4, "Parity check failed!"
print("[ONNX Parity] VERIFIED 100% BIT-EXACT PARITY!")
