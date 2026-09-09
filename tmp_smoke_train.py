# Smoke test: verify train_v6 imports, loss is non-negative, and
# a tiny 1-batch forward/backward pass runs correctly.
import torch
import numpy as np
import pickle, json

# 1. Verify loss is non-negative on random data
import sys
sys.path.insert(0, '.')
from v6_smartphone_idr.src.train_v6 import (
    compute_v6_loss, V6AugmentedDataset, compute_physical_mae,
)
from v6_smartphone_idr.src.models_v6 import PINODeadReckoningNetV6

model = PINODeadReckoningNetV6()
loss_weights = {"velocity": 1.0, "heading": 1.5, "zupt": 0.25, "bias": 0.05, "uncertainty": 0.01}

# Random batch (scaled [0,1] inputs, targets scaled [0,1])
bx = torch.rand(8, 20, 6)
b_dv = torch.rand(8, 1)
b_w = torch.rand(8, 1)
b_z = torch.rand(8, 1)
b_ba = torch.rand(8, 1)
b_bw = torch.rand(8, 1)
b_wgt = torch.ones(8, 1)

preds = model(bx)
targets = (b_dv, b_w, b_z, b_ba, b_bw)
loss, metrics = compute_v6_loss(preds, targets, b_wgt, turn_tau=1.0, loss_weights=loss_weights)
print("loss=", float(loss), "non-negative:", float(loss) >= 0)
print("metrics:", metrics)

# 2. Verify a tiny train_v6 run executes (subsample huge, 1 epoch)
import os
# Temporarily reduce max_epochs via env not needed; call train_v6 with tiny params
# We'll just test the module level by running 1 epoch with subsample=50 and epochs=1.
from v6_smartphone_idr.src import train_v6 as tv
# Monkeypatch config max_epochs is read at call; call directly:
try:
    tv.train_v6(max_epochs=1, batch_size=64, lr=8e-4, subsample=100,
                use_smote=True, use_augmentation=True, use_batch_lr=True)
    print("SMOKE TRAIN OK")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("SMOKE TRAIN FAILED:", e)
