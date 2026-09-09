"""
Test differentiable 10-step closed-loop trajectory training.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
from v7_sept_model.src.models_v7 import TurningSpecialistS2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("v7_sept_model/data/scalers_v7.pkl", "rb") as f:
    scalers = pickle.load(f)
with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

s_X = scalers["X"]
s_yd = scalers["y_disp"]
s_yo = scalers["y_ori"]

# Extract scale factors for differentiable unscaling
disp_scale = float(s_yd.scale_[0])
disp_min = float(s_yd.min_[0])
ori_scale = float(s_yo.scale_[0])
ori_min = float(s_yo.min_[0])

# To unscale: xr = (d_norm - disp_min) / disp_scale
# wr = (o_norm - ori_min) / ori_scale
print(f"Disp unscale: x = (d - {disp_min:.4f}) / {disp_scale:.4f}")
print(f"Ori unscale:  w = (o - {ori_min:.4f}) / {ori_scale:.4f}")
