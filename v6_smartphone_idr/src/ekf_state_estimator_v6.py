"""
ekf_state_estimator_v6.py
-------------------------
6-State Extended Kalman Filter (EKF) with ZUPT Hysteresis & Soft NHC for 10 Hz Smartphone IDR.

State Vector:
  x = [px, py, v, psi, b_a, b_w]^T
  - px, py: 2D local Cartesian positions (meters)
  - v:      forward vehicle velocity (m/s)
  - psi:    vehicle heading angle (radians)
  - b_a:    forward accelerometer bias (m/s²)
  - b_w:    vertical gyroscope bias (rad/s)

Measurement Updates:
  1. Neural Velocity & Yaw Rate (dynamically weighted by predicted uncertainty)
  2. Neural Bias Corrections
  3. Non-Holonomic Constraints (v_lat ≈ 0)
  4. Zero Velocity Updates (ZUPT) with Hysteresis Lock
"""

from __future__ import annotations

import math
import numpy as np


class ZUPTHysteresisGateV6:
    """Hysteresis gating to prevent rapid switching between stationary and moving states."""

    def __init__(self, p_enter: float = 0.70, p_exit: float = 0.30, n_enter: int = 3, n_exit: int = 2):
        self.p_enter = p_enter
        self.p_exit = p_exit
        self.n_enter = n_enter
        self.n_exit = n_exit
        self.is_stopped = False
        self.high_count = 0
        self.low_count = 0

    def reset(self):
        self.is_stopped = False
        self.high_count = 0
        self.low_count = 0

    def update(self, p_stop: float, a_fwd_mag: float, w_mag: float) -> bool:
        # Fuse neural probability with direct IMU energy check
        confident_still = (p_stop > self.p_enter) and (a_fwd_mag < 0.25) and (w_mag < 0.05)

        if not self.is_stopped:
            if confident_still:
                self.high_count += 1
                self.low_count = 0
                if self.high_count >= self.n_enter:
                    self.is_stopped = True
            else:
                self.high_count = 0
        else:
            confident_moving = (p_stop < self.p_exit) or (a_fwd_mag > 0.50) or (w_mag > 0.12)
            if confident_moving:
                self.low_count += 1
                self.high_count = 0
                if self.low_count >= self.n_exit:
                    self.is_stopped = False
            else:
                self.low_count = 0

        return self.is_stopped


class ExtendedKalmanFilterV6:
    def __init__(
        self,
        dt: float = 0.1,
        q_pos: float = 0.04,
        q_vel: float = 0.12,
        q_psi: float = 0.02,
        q_bias_a: float = 0.001,
        q_bias_w: float = 0.0005,
    ):
        self.dt = dt
        # State: [px, py, v, psi, b_a, b_w]
        self.x = np.zeros(6, dtype=np.float64)

        # State Covariance P
        self.P = np.diag([1.0, 1.0, 0.5, 0.05, 0.05, 0.01])

        # Process Noise Covariance Q
        self.Q = np.diag([
            q_pos * dt,
            q_pos * dt,
            q_vel * dt,
            q_psi * dt,
            q_bias_a * dt,
            q_bias_w * dt,
        ])

        self.zupt_gate = ZUPTHysteresisGateV6()

    def reset(self, x0: float = 0.0, y0: float = 0.0, v0: float = 0.0, psi0: float = 0.0):
        self.x = np.array([x0, y0, v0, psi0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 0.5, 0.05, 0.05, 0.01])
        self.zupt_gate.reset()

    def predict(self, a_fwd_meas: float, w_meas: float):
        """
        EKF Time Update (Prediction Step) at 10 Hz using IMU kinematic integration.
        """
        px, py, v, psi, b_a, b_w = self.x
        dt = self.dt
        self.psi_prev = psi

        # Unbiased inputs
        a_fwd = a_fwd_meas - b_a
        w_yaw = w_meas - b_w

        # State propagation
        psi_mid = psi + 0.5 * w_yaw * dt
        px_next = px + v * math.cos(psi_mid) * dt
        py_next = py + v * math.sin(psi_mid) * dt
        v_next = max(0.0, v + a_fwd * dt)
        psi_next = (psi + w_yaw * dt + math.pi) % (2 * math.pi) - math.pi

        self.x = np.array([px_next, py_next, v_next, psi_next, b_a, b_w], dtype=np.float64)

        # State Transition Jacobian F = df/dx
        F = np.eye(6, dtype=np.float64)
        F[0, 2] = math.cos(psi_mid) * dt
        F[0, 3] = -v * math.sin(psi_mid) * dt
        F[0, 5] = -0.5 * v * math.sin(psi_mid) * dt * dt
        F[1, 2] = math.sin(psi_mid) * dt
        F[1, 3] = v * math.cos(psi_mid) * dt
        F[1, 5] = 0.5 * v * math.cos(psi_mid) * dt * dt
        F[2, 4] = -dt
        F[3, 5] = -dt

        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q

    def update_neural_motion(
        self,
        v_neural: float,
        w_neural: float,
        b_a_neural: float,
        b_w_neural: float,
        var_v: float = 0.25,
        var_w: float = 0.04,
    ):
        """
        EKF Measurement Update fusing neural motion predictions.
        Measurement Vector z = [v, psi_meas, b_a, b_w]^T
        """
        # For w_neural, update heading relative to pre-step psi_prev
        psi_base = getattr(self, "psi_prev", self.x[3])
        psi_meas = (psi_base + w_neural * self.dt + math.pi) % (2 * math.pi) - math.pi
        z_extended = np.array([v_neural, psi_meas, b_a_neural, b_w_neural], dtype=np.float64)

        # Measurement matrix H (4x6)
        H = np.zeros((4, 6), dtype=np.float64)
        H[0, 2] = 1.0  # v
        H[1, 3] = 1.0  # psi
        H[2, 4] = 1.0  # b_a
        H[3, 5] = 1.0  # b_w

        R = np.diag([max(0.05, var_v), max(0.01, var_w * self.dt**2), 0.08, 0.02])

        # Innovation
        y = z_extended - H @ self.x
        # Angle wrap for psi innovation
        y[1] = (y[1] + math.pi) % (2 * math.pi) - math.pi

        # Innovation covariance
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[3] = (self.x[3] + math.pi) % (2 * math.pi) - math.pi
        self.P = (np.eye(6) - K @ H) @ self.P

    def apply_zupt(self, p_stop: float, a_fwd_mag: float, w_mag: float):
        """
        Zero Velocity Update: Clamps velocity and heading drift when stopped.
        """
        is_stopped = self.zupt_gate.update(p_stop, a_fwd_mag, w_mag)
        if is_stopped:
            # Direct pseudo-measurement: v = 0 with high certainty
            H_zupt = np.zeros((1, 6), dtype=np.float64)
            H_zupt[0, 2] = 1.0
            R_zupt = np.array([[0.001]], dtype=np.float64)

            y_zupt = np.array([0.0]) - self.x[2]
            S = H_zupt @ self.P @ H_zupt.T + R_zupt
            K = self.P @ H_zupt.T @ np.linalg.inv(S)

            self.x = self.x + (K @ y_zupt).squeeze()
            self.x[2] = 0.0  # Hard clamp
            self.P = (np.eye(6) - K @ H_zupt) @ self.P

        return is_stopped

    def get_state(self) -> dict:
        return {
            "px": float(self.x[0]),
            "py": float(self.x[1]),
            "v": float(self.x[2]),
            "psi": float(self.x[3]),
            "b_a": float(self.x[4]),
            "b_w": float(self.x[5]),
            "is_stopped": self.zupt_gate.is_stopped,
            "pos_std": float(math.sqrt(max(0.0, self.P[0, 0] + self.P[1, 1]))),
        }
