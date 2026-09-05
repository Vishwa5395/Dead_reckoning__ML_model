"""
models.py
---------
Input Delay Neural Network (IDNN) PyTorch architectures for:
1. Displacement Estimation (Closed-Loop / Feedback model)
2. Orientation Rate Estimation (Angular velocity model)

Hyperparameters per Table 5 in Onyekpe et al. (2021):
- 2 hidden layers, 32 neurons per layer
- 10% dropout
- Linear output
"""

import os
import torch
import torch.nn as nn


class DisplacementIDNN(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=32, dropout=0.10):
        super(DisplacementIDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class OrientationIDNN(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, dropout=0.10):
        super(OrientationIDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


def export_to_onnx(model, dummy_input, output_path, input_name='imu_input', output_name='prediction'):
    """
    Exports PyTorch model to ONNX and TorchScript formats for edge and mobile deployment.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    # TorchScript mobile export
    ts_path = output_path.replace('.onnx', '_torchscript.pt')
    try:
        traced = torch.jit.trace(model, dummy_input)
        traced.save(ts_path)
        print(f"[Export] Saved TorchScript model to: {ts_path}")
    except Exception as e:
        print(f"[Export Warning] TorchScript export failed: {e}")

    # ONNX export
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
            dynamic_axes={
                input_name: {0: 'batch_size'},
                output_name: {0: 'batch_size'}
            }
        )
        print(f"[Export] Saved ONNX model to: {output_path}")
    except Exception as e:
        print(f"[Export Warning] ONNX export failed: {e}")


if __name__ == '__main__':
    disp_model = DisplacementIDNN(input_dim=20, hidden_dim=32)
    ori_model = OrientationIDNN(input_dim=10, hidden_dim=32)
    print("Displacement IDNN parameters:", sum(p.numel() for p in disp_model.parameters() if p.requires_grad))
    print("Orientation IDNN parameters:", sum(p.numel() for p in ori_model.parameters() if p.requires_grad))
