# Automotive Dead Reckoning with Smartphone Inertial Sensors

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-opset_18-green.svg)](https://onnx.ai/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

High-precision vehicle dead-reckoning engine estimating forward displacement and yaw rate from **noisy, unaligned smartphone IMU sensors** during 10-second GNSS outages on the **IO-VNBD (Input-Output Vehicle Navigation Benchmark Dataset)**.

This repository tracks the complete evolutionary development of the models across four distinct generations (**v1 $\to$ v2 $\to$ v3 $\to$ v4**).

---

## Master Benchmark Matrix (10-Second GNSS Outages)

Evaluated across all 65 test outage sequences on the frozen **70/10/20 trip-level split** (zero temporal leakage):

### 10-Second Dead-Reckoning Final Drift (Meters — Lower is Better)

| Driving Scenario | Sequences | Raw INS (Physics) | **v1 (Baseline)** | **v2 (Production)** | **v3 (PINO-DR)** | **v4 (Turn-Focused Flagship)** 🏆 | **v4 vs Raw INS** | **v4 vs v3** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Motorway** | 7 | 29.99 m | 43.17 m | 10.53 m | **7.13 m** | 10.19 m | **+66.0% better** | -43.0% |
| **Quick Accel** | 4 | 29.11 m | 35.04 m | 24.57 m | 21.11 m | **19.58 m** | **+32.7% better** | **+7.3% better** 🏆 |
| **Hard Brake** | 12 | 43.97 m | 25.98 m | 23.33 m | **17.15 m** | 17.88 m | **+59.3% better** | -4.3% |
| **Sharp Turns** | 39 | 48.85 m | 47.88 m | 39.88 m | 39.26 m | **36.39 m** 🏆 | **+25.5% better** | **+7.3% better** 🏆 |
| **Roundabout** | 3 | 81.93 m | 74.21 m | 63.45 m | 75.31 m | **55.36 m** 🏆 | **+32.4% better** | **+26.5% better** 🏆 |

* **Overall 10-s Drift Across All 65 Sequences**: **29.99 m** 🏆 (Broke the 30-meter barrier, **+35.1% improvement over Raw INS**)
* **Sharp Turns Displacement CRSE**: **18.37 m** (down from INS 38.86 m, **+52.7% improvement**)
* **Hard Brake Displacement CRSE**: **14.03 m** (down from INS 24.86 m, **+43.6% improvement**)
* **Inference Latency**: **2.58 ms/step** on CPU (well below 10 ms real-time limit)
* **Model Parameters**: **21,941** ($\le 25,000$ mobile budget)

---

## Architectural Evolution

| Capability | Version 1 (`v1_baseline`) | Version 2 (`v2_production`) | Version 3 (`v3_pino_dr`) | Version 4 (`v4_turn_focused`) — SOTA |
| :--- | :--- | :--- | :--- | :--- |
| **Core Architecture** | 2-layer MLP ($32 \to 32$) | 3-layer MLP ($64 \to 64 \to 32$) + BatchNorm + GELU | PINO-DR: Conv1D stem + BiGRU + Single Temporal Attention | **Turn-Aware PINO-DR**: Conv1D + BiGRU + 10 Hz Yaw Acceleration + Directional Temporal Attention |
| **Input Signals** | 4 raw features | 4 clipped features | 4 channels: $[a_{\text{fwd}}, \omega_{\text{yaw}}, a_{\text{lat}}, v_{\text{prev}}]$ | **5–6 channels: $+ \dot{\omega}_{\text{yaw}}$ (10 Hz smoothed yaw accel) $+ r_{\text{centripetal}}$** |
| **Turn Dynamics** | ❌ None | ❌ None | Standard yaw rate | **Smoothed yaw acceleration $(\dot{\omega})$ for turn entry/exit detection** |
| **Velocity Modeling** | Absolute speed regression | Absolute speed bounded | Kinematic Residual Skip: $\hat{v} = v_{\text{prev}} + \Delta v$ | **Kinematic Residual + Cross-Task Coupling** |
| **Data Split Hygiene** | Leaked sliding windows | Leaked sliding windows | Frozen Trip-Level (50/9/13 journeys) | **Strict Frozen Trip-Level (zero temporal leakage)** |
| **Deployment Formats** | PyTorch `.pth` | PyTorch, ONNX, TorchScript | PyTorch, ONNX, TorchScript, PKL | **PyTorch, ONNX (opset 18), TorchScript, Engine PKL** |
| **Parameters** | 3,170 | 16,322 | 21,667 | **21,763 – 21,941** ($\le 25\text{k}$ budget) |

---

## Repository Organization

```
Dead_reckoning__ML_model/
├── README.md                          # Master comparison, navigation & benchmark matrix
├── .gitignore                         # Cache & temporary file exclusions
├── run_pipeline.py                    # Root entrypoint for v1
├── run_pipeline_v2.py                 # Root entrypoint for v2
├── run_pipeline_v3.py                 # Root entrypoint for v3
├── run_pipeline_v4.py                 # Root entrypoint for v4
│
├── v1_baseline/                       # Version 1 (Original Legacy IDNN Baseline)
│   ├── README.md                      # Exhaustive report on v1 & outlier bug
│   ├── run_pipeline.py                # Pipeline driver
│   ├── checkpoints/                   # Legacy checkpoints
│   └── results/                       # Quality reports & plots
│
├── v2_production/                     # Version 2 (Production Deep IDNN)
│   ├── README.md                      # Report on physical bounds, BatchNorm/GELU & leakage analysis
│   ├── run_pipeline_v2.py             # Pipeline driver
│   ├── checkpoints/                   # ONNX & PyTorch weights
│   └── results/                       # Quality reports & plots
│
├── v3_pino_dr/                        # Version 3 (PINO-DR Neural Operator)
│   ├── README.md                      # Flagship documentation, ZUPT, trip-level split & highway breakthrough
│   ├── run_pipeline_v3.py             # Pipeline driver
│   ├── checkpoints/                   # Best models (PTH, ONNX, TorchScript, PKL)
│   └── results/                       # Quality reports & plots
│
└── v4_turn_focused/                   # Version 4 (PINO-DR Turn-Focused & Ablation Study) — SOTA
    ├── README.md                      # In-depth turn dynamics report, yaw accel, physics ablation
    ├── run_pipeline_v4.py             # Pipeline driver
    ├── src/                           # models_v4.py, preprocess_v4.py, train_v4.py, evaluate_v4.py, report_v4.py
    ├── checkpoints/                   # v4 best models (ONNX, TorchScript, PKL)
    └── results/                       # Benchmark summaries, 5-scenario trajectory plots & quality reports
```

---

## Quickstart & Running Models

### 1. Run the Turn-Focused Model (v4 SOTA)
```bash
# Evaluate v4 on unseen test scenarios
python run_pipeline_v4.py --step eval

# Run full v4 pipeline (preprocess -> train -> eval -> report)
python run_pipeline_v4.py
```

### 2. Run the PINO-DR v3 Model
```bash
python run_pipeline_v3.py --step eval
```

### 3. Run Previous Baselines (v2 / v1)
```bash
python run_pipeline_v2.py --step evaluate
python run_pipeline.py --step evaluate
```
