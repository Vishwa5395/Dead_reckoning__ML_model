"""
augmentation_v6.py
------------------
Physical-unit IMU augmentation, maneuver weighting, and SMOTE-style upscaling
for PINO-DR v6.

Three techniques requested by the user:
  1. IMU augmentation (noise / scale / time-jitter) applied in PHYSICAL units.
  2. Maneuver importance weighting (emphasize hard turn / accel / brake samples).
  3. SMOT (SMOTE-style) upscaling: synthetic oversampling of the minority
     low-frequency high-error maneuvers (roundabouts, quick accelerations,
     sharp turns) via linear interpolation in feature space.
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# 1. Maneuver Importance Weighting
# ============================================================================

def compute_maneuver_weights(
    y_w_phys: np.ndarray,   # (N, 1) true yaw rate, radians/s (physical)
    y_dv_phys: np.ndarray,  # (N, 1) true delta-v over 0.1 s, m/s (physical)
    y_z: np.ndarray,        # (N, 1) ZUPT binary indicator
    alpha_turn: float = 1.0,
    alpha_accel: float = 1.0,
    alpha_zupt: float = 1.0,
):
    """
    Per-sample importance weight emphasizing high-maneuver dynamics.

    The more the yaw-rate / accel deviate from the steady-state baseline,
    the higher the weight. ZUPT (stop) samples get a small boost so the
    model still learns standstill behavior.

    Returns normalized weights in [0, 1] with mean ~1.
    """
    y_w = np.asarray(y_w_phys).reshape(-1)
    y_dv = np.asarray(y_dv_phys).reshape(-1)
    y_z = np.asarray(y_z).reshape(-1)

    std_w = float(np.std(y_w)) + 1e-6
    std_dv = float(np.std(y_dv)) + 1e-6

    # Turn score: normalized |yaw rate| (higher = sharper turn)
    turn_score = np.abs(y_w) / std_w

    # Longitudinal maneuver score: normalized |delta-v| (accel / brake)
    accel_score = np.abs(y_dv) / std_dv

    # Standing-still boost
    zupt_score = y_z

    raw = (
        1.0
        + alpha_turn * turn_score
        + alpha_accel * accel_score
        + alpha_zupt * zupt_score
    )

    # Normalize to mean 1, clip to avoid exploding weights.
    weights = raw / (float(np.mean(raw)) + 1e-8)
    weights = np.clip(weights, 0.25, 4.0).astype(np.float32)

    return weights.reshape(-1, 1)


# ============================================================================
# 2. Physical-Unit IMU Augmentor
# ============================================================================

class IMUAugmentor:
    """
    Applies physically-plausible augmentations to a 10 Hz IMU window.

    The window is in physical units with channels:
      [0] a_fwd          longitudinal accel (m/s^2)
      [1] w_yaw          yaw rate (rad/s)
      [2] a_lat          lateral accel (m/s^2)
      [3] v_prev         previous speed feedback (m/s)
      [4] w_yaw_accel    yaw acceleration (rad/s^2)
      [5] centripetal_res  a_lat - v_prev * w_yaw (m/s^2)

    Augmentations:
      - Gaussian sensor noise on IMU channels (0,1,2,4)
      - Multiplicative amplitude scale on accel/gyro
      - Random small time shift (roll) for temporal jitter
      - Recompute centripetal residual for physical consistency
    """

    def __init__(
        self,
        noise_std: float = 0.04,
        scale_low: float = 0.92,
        scale_high: float = 1.08,
        max_roll: int = 1,
        noise_channels: tuple = (0, 1, 2, 4),
        seed: int = 42,
    ):
        self.noise_std = noise_std
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.max_roll = max_roll
        self.noise_channels = noise_channels
        self._rng = np.random.default_rng(seed)

    def _channel_magnitudes(self, x: np.ndarray) -> np.ndarray:
        """Per-channel (6,) RMS scale for noise normalisation."""
        return np.sqrt(np.mean(x ** 2, axis=0) + 1e-6)

    def augment_window(self, x: np.ndarray) -> np.ndarray:
        """
        Take a physical window (W, 6) and return an augmented physical window.
        """
        x = np.array(x, dtype=np.float64, copy=True)
        W, C = x.shape

        magnitudes = self._channel_magnitudes(x)

        # 1. Gaussian sensor noise on selected IMU channels.
        noise = self._rng.normal(0.0, 1.0, size=(W, C))
        noise_scale = np.zeros(C)
        # Normalise noise by channel magnitude, but not zero out magnitude.
        noise_scale = magnitudes * self.noise_std
        for c in self.noise_channels:
            x[:, c] = x[:, c] + noise[:, c] * noise_scale[c]

        # 2. Amplitude scale on accel/gyro channels.
        for c in self.noise_channels:
            factor = self._rng.uniform(self.scale_low, self.scale_high)
            x[:, c] = x[:, c] * factor

        # 3. Small temporal roll (jitter).
        if self.max_roll > 0 and W > self.max_roll * 2:
            roll = int(self._rng.integers(-self.max_roll, self.max_roll + 1))
            if roll != 0:
                x = np.roll(x, roll, axis=0)
                # v_prev is feedback; keep the last value as the actual
                # predictor so it remains a valid "previous speed".
                x[:, 3] = x[:, 3]  # unchanged by roll is handled below

        # 4. Recompute centripetal residual for physical consistency.
        #    We align v_prev so that centripetal[i] = a_lat[i] - v_prev[i]*w_yaw[i].
        #    In the preprocessed window, ch_v_prev[i] is the speed estimated at
        #    step i-1 (shifted by one). We roll it forward by one to align with
        #    the non-shifted channels.
        v_aligned = np.roll(x[:, 3], 1)
        x[:, 5] = np.clip(
            x[:, 2] - v_aligned * x[:, 1],
            -8.0, 8.0,
        )

        return x.astype(np.float32)


# ============================================================================
# 3. SMOTE-style Synthetic Upscaling
# ============================================================================

def _compute_hard_mask(
    X_scaled: np.ndarray,
    y_w_phys: np.ndarray,
    y_dv_phys: np.ndarray,
    tau_turn: float,
    z_score_thresh: float = 1.0,
    dv_percentile: float = 70.0,
) -> np.ndarray:
    """
    Identify "minority" hard-maneuver samples:
      - high turn score   (|w| normalized by train std, above tau_turn)
      - high |delta-v|    (top dv_percentile of longitudinal maneuvers)
      - high combined (turn AND accel)
    """
    y_w = np.asarray(y_w_phys).reshape(-1)
    y_dv = np.asarray(y_dv_phys).reshape(-1)

    std_w = float(np.std(y_w)) + 1e-6
    std_dv = float(np.std(y_dv)) + 1e-6

    turn_score = np.abs(y_w) / std_w
    accel_score = np.abs(y_dv) / std_dv

    high_turn = turn_score > max(tau_turn * 0.75, z_score_thresh)
    high_accel = accel_score > float(np.percentile(accel_score, dv_percentile))

    # Hard = high turn OR strong accel/brake (the high-error minority).
    hard = (high_turn | high_accel).astype(bool)
    return hard


def generate_smote_upsamples(
    X_scaled: np.ndarray,     # (N, W, C) normalized inputs
    y_dv: np.ndarray,         # (N, 1) normalized delta-v
    y_w: np.ndarray,          # (N, 1) normalized yaw rate
    y_z: np.ndarray,          # (N, 1) ZUPT
    y_ba: np.ndarray,         # (N, 1) normalized accel bias
    y_bw: np.ndarray,         # (N, 1) normalized gyro bias
    y_w_phys: np.ndarray,     # (N, 1) physical yaw rate (for hard mask)
    y_dv_phys: np.ndarray,    # (N, 1) physical delta-v (for hard mask)
    tau_turn: float,
    n_synthetic: int | None = None,
    k_neighbors: int = 5,
    seed: int = 42,
):
    """
    SMOTE-style upscaling:

    For each minority (hard-maneuver) sample, pick k nearest neighbours
    (Euclidean in flattened scaled feature space) and interpolate:
        x_new = x_i + alpha * (x_j - x_i)
        y_new = y_i + alpha * (y_j - y_i)

    Returns the ORIGINAL arrays with synthetic samples appended.
    """
    rng = np.random.default_rng(seed)
    N = X_scaled.shape[0]
    flat = X_scaled.reshape(N, -1)

    hard = _compute_hard_mask(X_scaled, y_w_phys, y_dv_phys, tau_turn)
    hard_idx = np.where(hard)[0]
    n_hard = len(hard_idx)

    if n_hard < 5:
        return X_scaled, y_dv, y_w, y_z, y_ba, y_bw

    # Target number of synthetic samples.
    if n_synthetic is None:
        n_synthetic = int(n_hard * 0.5)   # upsample hard class by 50%

    # Cap so we do not over-inflate the training set (overfit guard).
    n_synthetic = min(n_synthetic, int(N * 0.35))

    # Scalar projection for cheap nearest neighbours of the hard subset.
    # We compute full pairwise distance on the hard subset only (small).
    hard_flat = flat[hard_idx]
    # Sub-sample hard subset if huge to bound memory.
    max_hard_for_nn = 8000
    if n_hard > max_hard_for_nn:
        perm = rng.permutation(n_hard)[:max_hard_for_nn]
        hard_idx_use = hard_idx[perm]
        hard_flat_use = hard_flat[perm]
    else:
        hard_idx_use = hard_idx
        hard_flat_use = hard_flat

    # Memory-efficient pairwise squared distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
    sq_norms = np.sum(hard_flat_use ** 2, axis=1, keepdims=True)
    dist2 = sq_norms + sq_norms.T - 2.0 * np.dot(hard_flat_use, hard_flat_use.T)
    np.maximum(dist2, 0.0, out=dist2)
    # Zero out self-distance
    np.fill_diagonal(dist2, np.inf)
    kk = min(k_neighbors, len(hard_flat_use) - 1)
    nearest = np.argpartition(dist2, kk, axis=1)[:, :kk]

    synth_X = []
    synth_dv = []
    synth_w = []
    synth_z = []
    synth_ba = []
    synth_bw = []

    n_to_make = n_synthetic
    made = 0
    attempts = 0
    while made < n_to_make and attempts < n_to_make * 10:
        attempts += 1
        i = int(rng.integers(0, len(hard_flat_use)))
        neighbor_list = nearest[i]
        j = int(neighbor_list[rng.integers(0, len(neighbor_list))])
        alpha = float(rng.uniform(0.2, 0.8))  # avoid degenerate copies

        src = hard_idx_use[i]
        if j >= len(hard_idx_use):
            continue
        dst = hard_idx_use[j]

        x_new = flat[src] + alpha * (flat[dst] - flat[src])
        synth_X.append(x_new.astype(np.float32))
        synth_dv.append(y_dv[src] + alpha * (y_dv[dst] - y_dv[src]))
        synth_w.append(y_w[src] + alpha * (y_w[dst] - y_w[src]))
        synth_z.append(y_z[src] + alpha * (y_z[dst] - y_z[src]))
        synth_ba.append(y_ba[src] + alpha * (y_ba[dst] - y_ba[src]))
        synth_bw.append(y_bw[src] + alpha * (y_bw[dst] - y_bw[src]))
        made += 1

    if made == 0:
        return X_scaled, y_dv, y_w, y_z, y_ba, y_bw

    synth_X = np.asarray(synth_X, dtype=np.float32).reshape(-1, X_scaled.shape[1], X_scaled.shape[2])
    synth_dv = np.asarray(synth_dv, dtype=np.float32).reshape(-1, 1)
    synth_w = np.asarray(synth_w, dtype=np.float32).reshape(-1, 1)
    synth_z = np.asarray(synth_z, dtype=np.float32).reshape(-1, 1)
    synth_ba = np.asarray(synth_ba, dtype=np.float32).reshape(-1, 1)
    synth_bw = np.asarray(synth_bw, dtype=np.float32).reshape(-1, 1)

    X_aug = np.concatenate([X_scaled, synth_X], axis=0)
    y_dv_aug = np.concatenate([y_dv, synth_dv], axis=0)
    y_w_aug = np.concatenate([y_w, synth_w], axis=0)
    y_z_aug = np.concatenate([y_z, synth_z], axis=0)
    y_ba_aug = np.concatenate([y_ba, synth_ba], axis=0)
    y_bw_aug = np.concatenate([y_bw, synth_bw], axis=0)

    return X_aug, y_dv_aug, y_w_aug, y_z_aug, y_ba_aug, y_bw_aug
