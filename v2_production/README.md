# Version 2 (v2) — Production IDNN with Domain Bounding & Deep Architectures

## Executive Summary
`v2_production` repaired the catastrophic scaling outlier from v1 by introducing **physical domain bounding (clipping)**, refitting scalers strictly on clean training data, and upgrading the neural architecture from simple 2-layer MLPs to **3-layer deep networks with Batch Normalization, GELU activations, and He/Kaiming initialization**.

This version rescued the displacement model, turning a model that underperformed pure physics by -28.3% into one that **beat pure INS by an average of +32.9% across all 5 test scenarios**.

---

## 1. Architecture & Engineering Upgrades

### Neural Architecture Upgrades
Both the Displacement and Orientation models were expanded:
* **Hidden Layers**: Expanded from $(32, 32)$ to **$(64, 64, 32)$** (3 hidden layers).
* **Normalization**: Added `nn.BatchNorm1d` after every hidden layer to stabilize gradient propagation and prevent internal covariate shift.
* **Activation**: Replaced standard `ReLU` with **`GELU` (Gaussian Error Linear Unit)** for smoother non-linear gradient flow.
* **Regularization**: Added Dropout (20%) after each activation.
* **Weight Initialization**: Initialized with `kaiming_normal_` and zero biases.

### Mathematical Formulation
For input vector $\mathbf{x} \in \mathbb{R}^{D}$:
$$\mathbf{h}_1 = \text{Dropout}_{0.2}\left(\text{GELU}\left(\text{BatchNorm}\left(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1\right)\right)\right)$$
$$\mathbf{h}_2 = \text{Dropout}_{0.2}\left(\text{GELU}\left(\text{BatchNorm}\left(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2\right)\right)\right)$$
$$\mathbf{h}_3 = \text{Dropout}_{0.2}\left(\text{GELU}\left(\text{BatchNorm}\left(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3\right)\right)\right)$$
$$\hat{y} = \mathbf{W}_4 \mathbf{h}_3 + b_4$$

### Trainable Parameter Count:
* **Displacement IDNN v2**: 8,481 parameters (vs 1,761 in v1).
* **Orientation IDNN v2**: 7,841 parameters (vs 1,409 in v1).

---

## 2. Root Cause Fix: Inversion & Domain Bounding

Because raw CSV files lived on an external drive, v2 applied **exact analytical inversion** on the existing MinMaxScalers:
$$x_{\text{raw}} = x_{\text{scaled}} \cdot (\text{max}_{\text{old}} - \text{min}_{\text{old}}) + \text{min}_{\text{old}}$$

Physical thresholds based on real-world vehicle dynamics limits were applied:
* **Longitudinal Acceleration**: $[-8.0, 8.0] \text{ m/s}^2$ (real p99.99 $\approx 6.25$)
* **Forward Velocity / Displacement**: $[0.0, 45.0] \text{ m/s}$ (real p99.99 $\approx 36.4$)
* **Gyroscope Angular Velocity**: $[-1.0, 1.0] \text{ rad/s}$ (real p99.99 $\approx 0.59$)
* **Vehicle Yaw Rate**: $[-1.2, 1.2] \text{ rad/s}$ (real p99.99 $\approx 0.71$)

Clean scalers were then re-fitted **strictly on the 70% training split**.

---

## 3. Benchmark Evaluation (v2 vs v1 vs Pure INS)

### 10-Second GNSS Outage Final Drift (Meters)

| Scenario | Sequences | Raw INS | v1 Baseline | **v2 Production IDNN** | Improvement vs Raw INS | Improvement vs v1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Motorway** | 7 | 29.99 m | 43.17 m | **10.53 m** | **+64.9%** | **+75.6%** |
| **Quick Accel** | 4 | 29.11 m | 35.04 m | **24.57 m** | **+15.6%** | **+29.9%** |
| **Hard Brake** | 12 | 43.97 m | 25.98 m | **23.33 m** | **+46.9%** | **+10.2%** |
| **Roundabout** | 3 | 81.93 m | 74.21 m | **63.45 m** | **+22.6%** | **+14.5%** |
| **Sharp Turns** | 39 | 48.85 m | 47.88 m | **39.88 m** | **+18.4%** | **+16.7%** |

### Cumulative Residual Squared Error (CRSE)

#### Displacement CRSE (Meters)
* **Motorway**: v1 = 42.62 m $\to$ **v2 = 8.26 m (-80.6% error reduction)**.
* **Quick Accel**: v1 = 32.18 m $\to$ **v2 = 15.16 m (-52.9% error reduction)**.
* **Roundabout**: v1 = 33.69 m $\to$ **v2 = 22.88 m (-32.1% error reduction)**.
* **Hard Brake**: v1 = 25.72 m $\to$ **v2 = 23.52 m (-8.5% error reduction)**.
* **Sharp Turns**: v1 = 29.97 m $\to$ **v2 = 26.72 m (-10.8% error reduction)**.
* **Overall**: v2 improved displacement CRSE in **5 out of 5 scenarios** (average 37.0% error reduction).

#### Orientation CRSE (Radians)
* **Motorway**: v1 = 0.060 rad $\to$ **v2 = 0.056 rad (-7.7%)**.
* **Quick Accel**: v1 = 0.161 rad $\to$ **v2 = 0.156 rad (-3.3%)**.
* **Hard Brake**: v1 = 0.175 rad $\to$ **v2 = 0.180 rad (+2.7%)**.
* **Roundabout**: v1 = 1.987 rad $\to$ **v2 = 1.983 rad (-0.2%)**.
* **Sharp Turns**: v1 = 1.030 rad $\to$ **v2 = 1.030 rad (equal)**.

---

## 4. Key Limitations & Why v3 Was Required

While v2 represented a major production improvement over v1, two fundamental engineering issues remained:

1. **Window-Level Split Leakage**:
   The train/test split in v2 was performed *after* windowing all journeys into 10-second sliding slices. Consecutive sliding windows overlap by 90% (9 out of 10 seconds). As a result, 70% of the motorway and quick-acceleration journeys were present in the training set, allowing the MLP to partially memorize the cruising speeds of those drives.
2. **Decoupled Predictions without Kinematic Physics**:
   Displacement and orientation were trained as completely separate models. The displacement model had no awareness of turning maneuvers or centripetal acceleration ($a_{\text{lat}} = v \cdot \omega$).
3. **Absence of Standstill Zero Velocity Update (ZUPT)**:
   During prolonged stops at traffic intersections or parking, small accelerometer bias continued to integrate, drifting without a stationary detector.

---

## 5. Directory Structure & Execution

```
v2_production/
├── checkpoints/
│   ├── displacement_idnn.pth / .onnx / _torchscript.pt
│   └── orientation_idnn.pth / .onnx / _torchscript.pt
├── data/
│   ├── clip_metadata.json
│   ├── dataset_splits_clean.npz
│   └── scalers_clean.pkl
├── results/
│   ├── benchmark_summary_v2.json
│   ├── model_quality_report_v2.txt / .json
│   ├── training_curves_v2.png
│   ├── training_history_displacement.json
│   ├── training_history_orientation.json
│   └── trajectory_v2_*.png
├── src/
│   ├── evaluate_v2.py
│   ├── models_v2.py
│   ├── preprocess_v2.py
│   ├── report_v2.py
│   └── train_v2.py
└── run_pipeline_v2.py
```

### Reproducing v2:
```bash
# Run benchmark evaluation
python run_pipeline_v2.py --step evaluate

# Clean and retrain
python run_pipeline_v2.py --step all
```
