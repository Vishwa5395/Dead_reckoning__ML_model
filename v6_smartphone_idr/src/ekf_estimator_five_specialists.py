"""
ekf_estimator_five_specialists.py
---------------------------------
Error-State Extended Kalman Filter (EKF) Estimator fusing:
  - IMU Kinematics Process Model
  - Specialist M1: Forward Velocity (v)
  - Specialist M2: Yaw Rate / Heading Increment (w)
  - Specialist M3: Accelerometer Bias (b_a)
  - Specialist M4: Gyroscope Bias (b_g)
  - Specialist M5: Dynamic Measurement Covariances (sigma_v, sigma_w, sigma_ba, sigma_bg)
  - Classical Physics ZUPT (Zero-Velocity Update) Detector

State Vector:
  x = [p_x, p_y, v, psi, b_a, b_g]^T  in R^6
"""

from __future__ import annotations

import math
import numpy as np


def wrap_angle(angle: float) -> float:
    """Wraps angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class FiveSpecialistEKF:
    """
    Continuous-Discrete Extended Kalman Filter for Dead Reckoning with Five Specialists.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt

        # State: [p_x, p_y, v, psi, b_a, b_g]
        self.x = np.zeros(6, dtype=np.float64)

        # Covariance Matrix P
        self.P = np.diag([
            1.0,    # p_x (m^2)
            1.0,    # p_y (m^2)
            0.5,    # v ((m/s)^2)
            0.05,   # psi (rad^2)
            0.1,    # b_a ((m/s^2)^2)
            0.01,   # b_g ((rad/s)^2)
        ])

        # Base Process Noise Q
        self.Q = np.diag([
            0.01,   # p_x
            0.01,   # p_y
            0.05,   # v
            0.005,  # psi
            1e-4,   # b_a (slow random walk)
            1e-5,   # b_g (slow random walk)
        ])

    def reset(self, init_px: float = 0.0, init_py: float = 0.0, init_v: float = 0.0, init_psi: float = 0.0):
        """Resets state vector to known initial condition."""
        self.x = np.array([init_px, init_py, init_v, init_psi, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([0.1, 0.1, 0.1, 0.01, 0.05, 0.005])

    def predict(self, a_fwd_raw: float, w_yaw_raw: float):
        """
        Propagate state using kinematic equations with estimated biases:
          p_x  <- p_x + v * cos(psi) * dt
          p_y  <- p_y + v * sin(psi) * dt
          v    <- v + (a_fwd - b_a) * dt
          psi  <- psi + (w_yaw - b_g) * dt
        """
        px, py, v, psi, ba, bg = self.x
        dt = self.dt

        a_eff = a_fwd_raw - ba
        w_eff = w_yaw_raw - bg

        # State propagation
        px_new = px + v * math.cos(psi) * dt
        py_new = py + v * math.sin(psi) * dt
        v_new = max(0.0, v + a_eff * dt)
        psi_new = wrap_angle(psi + w_eff * dt)

        self.x[0] = px_new
        self.x[1] = py_new
        self.x[2] = v_new
        self.x[3] = psi_new

        # Jacobian F = df/dx
        F = np.eye(6, dtype=np.float64)
        F[0, 2] = math.cos(psi) * dt
        F[0, 3] = -v * math.sin(psi) * dt
        F[1, 2] = math.sin(psi) * dt
        F[1, 3] = v * math.cos(psi) * dt
        F[2, 4] = -dt
        F[3, 5] = -dt

        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q

    def update_velocity(self, v_m1: float, sigma_v: float):
        """Update state using M1 velocity prediction with M5 dynamic variance."""
        v_meas = max(0.0, float(v_m1))
        r_v = max(1e-3, float(sigma_v)**2)

        # H = [0, 0, 1, 0, 0, 0]
        H = np.zeros((1, 6), dtype=np.float64)
        H[0, 2] = 1.0

        y = v_meas - self.x[2]  # Innovation
        S = H @ self.P @ H.T + r_v
        K = (self.P @ H.T) / S[0, 0]

        self.x = self.x + K.flatten() * y
        self.x[2] = max(0.0, self.x[2])  # Non-negative forward velocity
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K * r_v @ K.T

    def update_heading_rate(self, w_m2: float, sigma_w: float):
        """Update state using M2 yaw rate prediction with M5 dynamic variance."""
        dt = self.dt
        dpsi_meas = float(w_m2) * dt
        r_psi = max(1e-5, (float(sigma_w) * dt)**2)

        # Heading innovation
        H = np.zeros((1, 6), dtype=np.float64)
        H[0, 3] = 1.0

        y = wrap_angle(dpsi_meas - (self.x[5] * dt))  # Innovation
        S = H @ self.P @ H.T + r_psi
        K = (self.P @ H.T) / S[0, 0]

        self.x = self.x + K.flatten() * y
        self.x[3] = wrap_angle(self.x[3])
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K * r_psi @ K.T

    def update_biases(self, ba_m3: float, bg_m4: float, sigma_ba: float, sigma_bg: float):
        """Update state using M3 accel bias and M4 gyro bias predictions."""
        # Update b_a
        r_ba = max(1e-4, float(sigma_ba)**2)
        H_ba = np.zeros((1, 6), dtype=np.float64)
        H_ba[0, 4] = 1.0
        y_ba = float(ba_m3) - self.x[4]
        S_ba = H_ba @ self.P @ H_ba.T + r_ba
        K_ba = (self.P @ H_ba.T) / S_ba[0, 0]
        self.x = self.x + K_ba.flatten() * y_ba
        I_K = np.eye(6) - K_ba @ H_ba
        self.P = I_K @ self.P @ I_K.T + K_ba * r_ba @ K_ba.T

        # Update b_g
        r_bg = max(1e-5, float(sigma_bg)**2)
        H_bg = np.zeros((1, 6), dtype=np.float64)
        H_bg[0, 5] = 1.0
        y_bg = float(bg_m4) - self.x[5]
        S_bg = H_bg @ self.P @ H_bg.T + r_bg
        K_bg = (self.P @ H_bg.T) / S_bg[0, 0]
        self.x = self.x + K_bg.flatten() * y_bg
        I_K2 = np.eye(6) - K_bg @ H_bg
        self.P = I_K2 @ self.P @ I_K2.T + K_bg * r_bg @ K_bg.T

    def apply_zupt(self):
        """Hard zero-velocity constraint when vehicle is stationary."""
        self.x[2] = 0.0  # Velocity = 0
        self.P[2, 2] = 1e-4

    def get_state(self) -> dict:
        """Returns current estimated vehicle state."""
        return {
            "px": float(self.x[0]),
            "py": float(self.x[1]),
            "v": float(self.x[2]),
            "psi": float(self.x[3]),
            "ba": float(self.x[4]),
            "bg": float(self.x[5]),
        }
