"""
three_regime_router.py
----------------------
Physically Separable 3-Regime Kinematic Router for PINO-DR v7.

Collapses 5 overlapping scenario categories into 3 physically separable regimes:
  1. CRUISING:              High speed, low dynamics -> Motorway specialist (S1, 4ch)
  2. TURNING:               High |w_yaw| or |a_lat|  -> Cornering / Roundabout specialist (S2, 6ch)
  3. LONGITUDINAL TRANSIENT: |a_fwd| spike           -> Hard Brake (a_fwd < 0) / Quick Accel (a_fwd > 0)

Thresholds:
  tau_yaw:   threshold on |w_yaw| (rad/s) to enter turning regime
  tau_accel: threshold on |a_fwd| (m/s^2) to enter longitudinal transient regime
  tau_speed: threshold on speed (m/s) to classify steady cruising
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


class ThreeRegimeRouter:
    """3-Regime Kinematic Router using only real-time sensor measurements."""

    def __init__(self, tau_yaw: float = 0.04, tau_accel: float = 1.0, tau_speed: float = 16.0):
        self.tau_yaw = float(tau_yaw)
        self.tau_accel = float(tau_accel)
        self.tau_speed = float(tau_speed)

    def classify(self, v_prev: float, a_fwd: float, a_lat: float, w_yaw: float) -> str:
        """Route to appropriate specialist based on live physical cues.
        
        Priority:
          1. Longitudinal Transient (|a_fwd| > tau_accel):
             - a_fwd < 0 -> hard_brake
             - a_fwd > 0 -> quick_accel
          2. Turning (|w_yaw| > tau_yaw or |a_lat| > 1.8):
             - If moderate sustained speed & lateral acceleration -> roundabout
             - Else -> sharp_turns
          3. Cruising (v_prev > tau_speed):
             - motorway
          4. Default / Urban driving:
             - sharp_turns
        """
        # Priority 1: Longitudinal transient (hard brake or hard accel)
        if abs(a_fwd) >= self.tau_accel:
            return "hard_brake" if a_fwd < 0.0 else "quick_accel"

        # Priority 2: Turning
        if abs(w_yaw) >= self.tau_yaw or abs(a_lat) >= 1.8:
            # Sub-regime: Roundabout (sustained circular cornering at medium speeds)
            if abs(a_lat) >= 1.5 and 4.0 <= v_prev <= 18.0:
                return "roundabout"
            return "sharp_turns"

        # Priority 3: Cruising (high speed, steady heading)
        if v_prev >= self.tau_speed:
            return "motorway"

        # Default fallback: urban driving (low speed, small yaw variations)
        return "sharp_turns"

    def compute_regime_weights(self, v_prev: float, a_fwd: float, a_lat: float, w_yaw: float) -> Dict[str, float]:
        """Soft regime affinities for continuous blending (optional)."""
        assigned = self.classify(v_prev, a_fwd, a_lat, w_yaw)
        # Soft one-hot default
        weights = {
            "motorway": 0.0,
            "roundabout": 0.0,
            "quick_accel": 0.0,
            "hard_brake": 0.0,
            "sharp_turns": 0.0,
        }
        weights[assigned] = 1.0
        return weights
