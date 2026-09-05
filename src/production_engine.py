"""
production_engine.py
--------------------
Self-contained streaming dead-reckoning engine for real-time mobile/web apps.

Usage:
    engine = InertialDeadReckoningEngine("checkpoints_v3/best_model.onnx",
                                         "data/preprocessed/v3/scalers_v3.pkl")
    engine.reset(lat=52.41, lon=-1.52, heading_deg=90.0)

    while gnss_is_unavailable:
        state = engine.update(accel_xyz=[ax, ay, az],
                              gyro_xyz=[wx, wy, wz],
                              gravity_xyz=[gx, gy, gz],
                              dt=0.1)
        print(state)  # {"x": ..., "y": ..., "velocity": ..., "heading_deg": ..., ...}
"""

from __future__ import annotations

import math
import pickle
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np


class InertialDeadReckoningEngine:
    """
    Real-time streaming dead-reckoning engine powered by PINO-DR v3.
    Processes raw smartphone IMU readings and maintains a 2D position estimate.
    """

    CLIP_ACCEL = (-8.0, 8.0)
    CLIP_GYRO = (-1.0, 1.0)
    CLIP_DISP = (0.0, 45.0)
    CLIP_YAW = (-1.2, 1.2)

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        window_size: int = 10,
        zupt_high: float = 0.7,
        zupt_low: float = 0.3,
        zupt_n_enter: int = 3,
        zupt_n_exit: int = 2,
        use_onnx: bool = True,
    ):
        self.window_size = window_size
        self.use_onnx = use_onnx

        # Load scalers
        with open(scaler_path, "rb") as f:
            self.scalers = pickle.load(f)

        # Load model
        if use_onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
        else:
            import torch
            from src.models_v3 import PINODeadReckoningNet
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {})
            self.model = PINODeadReckoningNet(
                in_channels=cfg.get("in_channels", 4),
                conv_channels=cfg.get("conv_channels", 32),
                gru_hidden=cfg.get("gru_hidden", 32),
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

        # ZUPT hysteresis state
        self.zupt_high = zupt_high
        self.zupt_low = zupt_low
        self.zupt_n_enter = zupt_n_enter
        self.zupt_n_exit = zupt_n_exit

        self.reset()

    def reset(self, lat: float = 0.0, lon: float = 0.0, heading_deg: float = 0.0):
        """Reset the engine state."""
        self.x = 0.0
        self.y = 0.0
        self.velocity = 0.0
        self.heading_rad = math.radians(heading_deg)
        self.is_stopped = False
        self.zupt_high_count = 0
        self.zupt_low_count = 0
        self.step_count = 0

        # Circular buffers for the 10-step window (1 Hz accumulation)
        self.buf_a_fwd = deque(maxlen=self.window_size)
        self.buf_w_yaw = deque(maxlen=self.window_size)
        self.buf_a_lat = deque(maxlen=self.window_size)
        self.buf_v_prev = deque(maxlen=self.window_size)

        # Pre-fill with zeros
        for _ in range(self.window_size):
            self.buf_a_fwd.append(0.0)
            self.buf_w_yaw.append(0.0)
            self.buf_a_lat.append(0.0)
            self.buf_v_prev.append(0.0)

        # 10 Hz accumulator for 1-second averaging
        self._hz_a_fwd = []
        self._hz_w_yaw = []
        self._hz_a_lat = []
        self._fwd_dir = np.array([0.0, 1.0, 0.0])  # default forward direction

    def _level_sample(self, accel_xyz, gyro_xyz, gravity_xyz):
        """Online gravity leveling for a single 10 Hz sample."""
        ax, ay, az = accel_xyz
        gx, gy, gz = gravity_xyz
        wx, wy, wz = gyro_xyz

        g_norm = math.sqrt(gx**2 + gy**2 + gz**2)
        if g_norm < 1e-3:
            g_norm = 9.80665
        uz = np.array([gx / g_norm, gy / g_norm, gz / g_norm])

        a_lin = np.array([ax - gx, ay - gy, az - gz])
        a_vert = np.dot(a_lin, uz)
        a_horiz = a_lin - a_vert * uz

        a_fwd = float(np.dot(a_horiz, self._fwd_dir))

        # Lateral direction
        lat_dir = np.cross(uz, self._fwd_dir)
        lat_norm = np.linalg.norm(lat_dir)
        if lat_norm > 1e-6:
            lat_dir = lat_dir / lat_norm
        a_lat = float(np.dot(a_horiz, lat_dir))

        # Yaw rate around vertical
        w_gyro = np.array([wx, wy, wz])
        w_yaw = float(np.dot(w_gyro, uz))

        return a_fwd, w_yaw, a_lat

    def update(
        self,
        accel_xyz: list,
        gyro_xyz: list,
        gravity_xyz: list,
        dt: float = 0.1,
    ) -> Optional[dict]:
        """
        Process a single 10 Hz IMU sample.
        Returns a state dict every 10 samples (1 second), or None otherwise.
        """
        a_fwd, w_yaw, a_lat = self._level_sample(accel_xyz, gyro_xyz, gravity_xyz)
        self._hz_a_fwd.append(a_fwd)
        self._hz_w_yaw.append(w_yaw)
        self._hz_a_lat.append(a_lat)

        # Only produce output at 1 Hz (every 10 samples)
        if len(self._hz_a_fwd) < 10:
            return None

        # Average over the last 10 readings
        a_fwd_1s = float(np.clip(np.mean(self._hz_a_fwd), *self.CLIP_ACCEL))
        w_yaw_1s = float(np.clip(np.mean(self._hz_w_yaw), *self.CLIP_GYRO))
        a_lat_1s = float(np.clip(np.mean(self._hz_a_lat), *self.CLIP_ACCEL))
        self._hz_a_fwd.clear()
        self._hz_w_yaw.clear()
        self._hz_a_lat.clear()

        self.buf_a_fwd.append(a_fwd_1s)
        self.buf_w_yaw.append(w_yaw_1s)
        self.buf_a_lat.append(a_lat_1s)
        self.buf_v_prev.append(self.velocity)

        # Build (1, 10, 4) window
        window = np.stack([
            np.array(self.buf_a_fwd),
            np.array(self.buf_w_yaw),
            np.array(self.buf_a_lat),
            np.array(self.buf_v_prev),
        ], axis=-1).astype(np.float32)  # (10, 4)

        # Scale
        s_X = self.scalers["X"]
        window_flat = window.reshape(1, -1)
        window_scaled = s_X.transform(window_flat).reshape(1, 10, 4).astype(np.float32)

        # Inference
        if self.use_onnx:
            outputs = self.session.run(None, {self.input_name: window_scaled})
            d_pred_s, o_pred_s, z_logit = outputs[0][0, 0], outputs[1][0, 0], outputs[2][0, 0]
        else:
            import torch
            with torch.no_grad():
                inp = torch.tensor(window_scaled)
                d, o, z = self.model(inp)
            d_pred_s, o_pred_s, z_logit = d.item(), o.item(), z.item()

        # Inverse transform
        x_pred = float(np.clip(
            self.scalers["y_disp"].inverse_transform([[d_pred_s]])[0, 0],
            *self.CLIP_DISP
        ))
        w_pred = float(np.clip(
            self.scalers["y_ori"].inverse_transform([[o_pred_s]])[0, 0],
            *self.CLIP_YAW
        ))
        p_stop = 1.0 / (1.0 + math.exp(-z_logit))  # sigmoid

        # ZUPT hysteresis
        if not self.is_stopped:
            if p_stop > self.zupt_high:
                self.zupt_high_count += 1
                self.zupt_low_count = 0
                if self.zupt_high_count >= self.zupt_n_enter:
                    self.is_stopped = True
            else:
                self.zupt_high_count = 0
        else:
            if p_stop < self.zupt_low:
                self.zupt_low_count += 1
                self.zupt_high_count = 0
                if self.zupt_low_count >= self.zupt_n_exit:
                    self.is_stopped = False
            else:
                self.zupt_low_count = 0

        if self.is_stopped:
            x_pred = 0.0
            w_pred = 0.0

        # Integrate
        self.velocity = x_pred
        self.heading_rad += w_pred
        self.x += x_pred * math.cos(self.heading_rad)
        self.y += x_pred * math.sin(self.heading_rad)
        self.step_count += 1

        return {
            "x": self.x,
            "y": self.y,
            "velocity": self.velocity,
            "heading_deg": math.degrees(self.heading_rad) % 360,
            "is_stopped": self.is_stopped,
            "p_stop": p_stop,
            "step": self.step_count,
        }
