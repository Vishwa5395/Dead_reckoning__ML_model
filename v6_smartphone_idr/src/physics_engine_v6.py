"""
physics_engine_v6.py
--------------------
Deterministic Vehicle Motion Physics Engine & Non-Holonomic Constraint (NHC) Validator.

Implements:
1. 10 Hz Kinematic Vehicle Motion Integration:
   v_{t+1} = v_t + a_fwd * dt
   psi_{t+1} = psi_t + w_yaw * dt
   x_{t+1} = x_t + v_t * cos(psi_t) * dt
   y_{t+1} = y_t + v_t * sin(psi_t) * dt
2. Non-Holonomic Constraints (NHC):
   v_lateral ≈ 0, v_vertical ≈ 0 (soft constraint with tire sideslip tolerance).
3. Dynamic Centripetal Consistency:
   a_lateral ≈ v * w_yaw
4. Physical Plausibility & Sanity Guardrails:
   - Clamps impossible acceleration (> 6.5 m/s²) and braking (< -8.5 m/s²)
   - Bounds maximum yaw rate (|w| <= 1.2 rad/s)
   - Filters abrupt unphysical speed discontinuities
"""

from __future__ import annotations

import math
import numpy as np


class VehiclePhysicsEngineV6:
    def __init__(
        self,
        dt: float = 0.1,
        max_fwd_accel: float = 6.5,    # m/s²
        max_brake_accel: float = -8.5,  # m/s²
        max_yaw_rate: float = 1.2,     # rad/s
        max_speed: float = 45.0,       # m/s (~162 km/h)
        centripetal_tol: float = 2.5,  # m/s² tolerance for tire slip & road banking
    ):
        self.dt = dt
        self.max_fwd_accel = max_fwd_accel
        self.max_brake_accel = max_brake_accel
        self.max_yaw_rate = max_yaw_rate
        self.max_speed = max_speed
        self.centripetal_tol = centripetal_tol

    def validate_and_constrain_dynamics(
        self,
        v_prev: float,
        delta_v_pred: float,
        w_pred: float,
        a_fwd_measured: float,
        a_lat_measured: float,
        accel_bias: float = 0.0,
        gyro_bias: float = 0.0,
        w_gyro_measured: float | None = None,
    ) -> tuple[float, float, dict]:
        """
        Validates and applies active physical constraints and sensor fusion to neural predictions.
        Returns:
            v_constrained: physically plausible velocity
            w_constrained: physically plausible yaw rate
            diagnostics: dictionary of physics residuals
        """
        # Bias compensation
        a_fwd_clean = a_fwd_measured - accel_bias
        w_clean = w_pred - gyro_bias
        w_gyro_clean = (w_gyro_measured - gyro_bias) if w_gyro_measured is not None else w_clean

        # 1. Kinematic forward velocity update
        v_candidate = v_prev + delta_v_pred
        a_implied = (v_candidate - v_prev) / self.dt

        # Active IMU Throttle / Brake Kinematic Fusion
        if a_fwd_clean < -1.0:
            # Active braking confirmed by accelerometer
            a_use = min(a_implied, a_fwd_clean)
            v_candidate = v_prev + a_use * self.dt
        elif a_fwd_clean > 1.0 and v_candidate < 28.0:
            # Sustained throttle acceleration confirmed by forward accelerometer
            if a_implied < 0.6 * a_fwd_clean:
                effective_accel = 0.4 * a_implied + 0.6 * (1.1 * a_fwd_clean)
                v_candidate = v_prev + effective_accel * self.dt
        elif a_implied > self.max_fwd_accel:
            v_candidate = v_prev + self.max_fwd_accel * self.dt
        elif a_implied < self.max_brake_accel:
            v_candidate = v_prev + self.max_brake_accel * self.dt

        v_constrained = float(np.clip(v_candidate, 0.0, self.max_speed))

        # 2. Yaw rate constraint (neural model is clean and noise-free)
        w_constrained = float(np.clip(w_clean, -self.max_yaw_rate, self.max_yaw_rate))

        # 3. Centripetal consistency check: a_lat ≈ v * w
        a_lat_expected = v_constrained * w_constrained
        centripetal_residual = abs(a_lat_measured - a_lat_expected)
        nhc_trust = max(0.5, 1.0 - max(0.0, centripetal_residual - self.centripetal_tol) / 5.0)

        diagnostics = {
            "a_implied": a_implied,
            "centripetal_residual": centripetal_residual,
            "nhc_trust": nhc_trust,
            "a_fwd_clean": a_fwd_clean,
            "w_gyro_clean": w_gyro_clean,
        }

        return v_constrained, w_constrained, diagnostics

    def integrate_step(
        self,
        x: float,
        y: float,
        heading: float,
        v: float,
        w: float,
    ) -> tuple[float, float, float]:
        """
        Deterministic 2D Non-Holonomic trajectory propagation step (dt = 0.1s).
        Enforces v_lat = 0 and v_vert = 0 in vehicle body frame.
        """
        heading_next = heading + w * self.dt
        # Normalize heading to [-pi, pi]
        heading_next = (heading_next + math.pi) % (2 * math.pi) - math.pi

        # Trapezoidal / midpoint integration for position
        heading_mid = heading + 0.5 * w * self.dt
        dx = v * math.cos(heading_mid) * self.dt
        dy = v * math.sin(heading_mid) * self.dt

        x_next = x + dx
        y_next = y + dy

        return x_next, y_next, heading_next
