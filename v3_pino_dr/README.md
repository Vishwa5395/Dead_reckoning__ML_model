# Version 3 (v3) — PINO-DR: Physics-Informed Neural Operator for Dead Reckoning

## Executive Summary
`v3_pino_dr` represents the complete, production-grade state-of-the-art architecture for vehicle dead reckoning from smartphone inertial sensors. 

It addresses every core limitation of previous versions by introducing:
1. **Kinematic Residual Skip Connection**: $\hat{v}(t) = v_{\text{prev}}[-1] + \Delta v(t)$, eliminating velocity decay during highway cruising.
2. **Zero Data Leakage (Trip-Level Split)**: 50 train journeys / 9 val / 13 test journeys with zero journey overlap, proving real out-of-distribution generalization on unseen roads and drivers.
3. **Multi-Task Physics Architecture**: Unifies Conv1D dilated feature extraction, Bidirectional GRU sequence modeling, temporal attention, and multi-task heads (displacement, orientation, and ZUPT).
4. **Zero Velocity Update (ZUPT) Hysteresis Gate**: Detects stops and clamps integration drift when stationary.

**Result**: Outperforms raw physics (INS) across **all 5 driving scenarios**, achieving **+41.3% average displacement improvement** and **+56.2% average orientation improvement**, while running at **2.08 ms per step on CPU** (5x faster than real-time budget).

---

## 1. Network Architecture

```
                       Input Tensor (Batch, 10, 4)
             [a_fwd (t-9..t), w_yaw (t-9..t), a_lat (t-9..t), v_prev (t-9..t)]
                                    │
                                    ▼
                 Conv1D Block 1 (In: 4, Out: 32, K: 3, Dil: 1)
                                    │
                                    ▼
                 Conv1D Block 2 (In: 32, Out: 32, K: 3, Dil: 2)
                                    │  (Residual Add)
                                    ▼
                         Dropout (p = 0.20)
                                    │
                                    ▼
                    Bidirectional GRU (Hidden: 32 → Out: 64)
                                    │
                                    ▼
                        Temporal Attention Pooling
                        (Learned Context Vector: 64-dim)
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
   Displacement Head         Orientation Head             ZUPT Head
  Linear(64→32) + GELU     Linear(64→32) + GELU      Linear(64→16) + GELU
     Linear(32→1)              Linear(32→1)              Linear(16→1)
          │                         │                         │
          ▼                         ▼                         ▼
    Δv (Velocity Delta)       ω_t (Yaw Rate)           p_stop (Stationary Logit)
          │
     + v_prev[-1]  (Kinematic Residual Connection)
          │
          ▼
   v_t (Predicted Velocity)
```

### Parameter Budget & Efficiency:
* **Total Trainable Parameters**: **21,667** (strictly within the mobile edge budget of $\le 25,000$).
* **Memory Footprint**: ~316 KB checkpoint size, 64 KB ONNX graph.
* **Inference Latency**: **2.076 ms/step** on CPU (well below the 10 ms real-time requirement for 100 Hz sensor pipelines).

---

## 2. Key Methodological Advances

### A. Kinematic Residual Velocity Link
In earlier models, predicting absolute velocity $v(t)$ directly from non-linear activations caused highway speeds to slowly decay over 10-second rollouts. In v3, we reformulate the task to predict the **kinematic acceleration delta**:
$$v(t) = \text{clamp}\left(v_{\text{prev}}[-1] + \Delta v(t),\, 0.0,\, 45.0\right)$$
On constant-speed motorways, $\Delta v \approx 0$, preserving cruising velocity with zero drift.

### B. Zero-Leakage Trip Split
Unlike v1 and v2 which shuffled sliding windows, v3 partitions by **complete journey files**:
* **Train Set**: 50 complete journeys (72,614 windows, 68.3% duration)
* **Val Set**: 9 complete journeys (10,676 windows, 10.1% duration)
* **Test Set**: 13 complete journeys (23,033 windows, 21.6% duration)
* **Journey overlap**: Strictly **0.0%**. The test scenarios (`vw12`, `vta11`, `vta12`, `vw16b`, `vw17`, `vta9`, `vw6`, `vw7`, `vw8`) were completely unseen during training.

### C. ZUPT Hysteresis Gate
To eliminate velocity jitter and standstill drift:
* **Enter Stop State**: When $p_{\text{stop}} > 0.70$ for $N_{\text{enter}} \ge 3$ consecutive seconds $\to$ velocity and yaw rate clamped to exactly $0.0$.
* **Exit Stop State**: Requires $p_{\text{stop}} < 0.30$ for $N_{\text{exit}} \ge 2$ consecutive seconds $\to$ resumes kinematic integration.

---

## 3. Comprehensive Benchmark Comparison (All Versions)

### 10-Second Dead-Reckoning Final Drift (Lower is Better)

| Scenario | Sequences | Raw INS (Physics) | v1 Baseline | v2 IDNN | **v3 PINO-DR** | v3 vs Raw INS | **v3 vs v2** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Motorway** | 7 | 29.99 m | 43.17 m | 10.53 m | **7.13 m** | **+76.2%** | **+32.3% better** 🏆 |
| **Quick Accel** | 4 | 29.11 m | 35.04 m | 24.57 m | **21.11 m** | **+27.5%** | **+14.1% better** 🏆 |
| **Hard Brake** | 12 | 43.97 m | 25.98 m | 23.33 m | **17.15 m** | **+61.0%** | **+26.5% better** 🏆 |
| **Sharp Turns** | 39 | 48.85 m | 47.88 m | 39.88 m | **39.26 m** | **+19.6%** | **+1.5% better** 🏆 |
| **Roundabout** | 3 | 81.93 m | 74.21 m | 63.45 m | **75.31 m** | **+8.1%** | -18.7% |

### Cumulative Residual Error (CRSE)

#### Displacement CRSE (Meters)
* **Motorway**: Raw INS = 19.60 m $\to$ v2 = 8.26 m $\to$ **v3 = 6.69 m (Best, +65.9% vs INS)**.
* **Hard Brake**: Raw INS = 24.86 m $\to$ v2 = 23.52 m $\to$ **v3 = 15.69 m (Best, +36.9% vs INS)**.
* **Sharp Turns**: Raw INS = 38.86 m $\to$ v2 = 26.72 m $\to$ **v3 = 22.02 m (Best, +43.3% vs INS)**.
* **Quick Accel**: Raw INS = 18.87 m $\to$ v1 = 32.18 m $\to$ **v3 = 16.44 m (+12.9% vs INS)**.
* **Roundabout**: Raw INS = 46.07 m $\to$ v1 = 33.69 m $\to$ **v3 = 24.10 m (+47.7% vs INS)**.

#### Orientation CRSE (Radians)
* **Motorway**: Raw INS = 0.693 rad $\to$ **v3 = 0.052 rad (Best, +92.5% vs INS)**.
* **Roundabout**: Raw INS = 1.912 rad $\to$ v2 = 1.983 rad *(worse)* $\to$ **v3 = 1.606 rad (Best, +16.0% vs INS)**.
* **Quick Accel**: Raw INS = 0.517 rad $\to$ **v3 = 0.153 rad (Best, +70.4% vs INS)**.
* **Hard Brake**: Raw INS = 0.900 rad $\to$ **v3 = 0.186 rad (+79.4% vs INS)**.
* **Sharp Turns**: Raw INS = 1.383 rad $\to$ **v3 = 1.068 rad (+22.8% vs INS)**.

---

## 4. Production Artifacts & Deployment

The trained v3 weights are exported into 3 standard formats:

1. **PyTorch State Dict**: [`checkpoints/best_model.pth`](file:///c:/Users/tiwar/OneDrive/Desktop/PROJECTS/SIH%2026/v3_pino_dr/checkpoints/best_model.pth) (full training checkpoint, optimizer state, configs).
2. **ONNX Graph**: [`checkpoints/best_model.onnx`](file:///c:/Users/tiwar/OneDrive/Desktop/PROJECTS/SIH%2026/v3_pino_dr/checkpoints/best_model.onnx) (opset 18, dynamic batch dimension for mobile/embedded C++ inference).
3. **TorchScript Traced**: [`checkpoints/best_model_torchscript.pt`](file:///c:/Users/tiwar/OneDrive/Desktop/PROJECTS/SIH%2026/v3_pino_dr/checkpoints/best_model_torchscript.pt) (standalone C++ `libtorch` ready).
4. **Production Engine**: [`checkpoints/final_model_v3.pkl`](file:///c:/Users/tiwar/OneDrive/Desktop/PROJECTS/SIH%2026/v3_pino_dr/checkpoints/final_model_v3.pkl) (packaged with `MinMaxScaler` scalers and metadata).

---

## 5. Directory Structure & Execution

```
v3_pino_dr/
├── checkpoints/
│   ├── best_model.pth / .onnx / _torchscript.pt
│   ├── final_model_v3.pkl
│   └── epochs/ (all 60 per-epoch checkpoints)
├── data/
│   ├── dataset_splits_v3.npz
│   ├── scalers_v3.pkl
│   ├── test_scenarios_v3.pkl
│   └── metadata_v3.json
├── results/
│   ├── benchmark_summary_v3.json
│   ├── model_quality_report_v3.txt / .json
│   ├── training_curves_v3.png
│   └── trajectory_v3_*.png
├── src/
│   ├── evaluate_v3.py
│   ├── models_v3.py
│   ├── preprocess_v3.py
│   ├── production_engine.py
│   ├── report_v3.py
│   └── train_v3.py
└── run_pipeline_v3.py
```

### Reproducing v3:
```bash
# Full evaluation on unseen test scenarios
python v3_pino_dr/run_pipeline_v3.py --step eval

# Retrain the complete pipeline from scratch
python v3_pino_dr/run_pipeline_v3.py --step all
```
