# Automotive Dead Reckoning with Smartphone Inertial Sensors

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-opset_18-green.svg)](https://onnx.ai/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

High-precision vehicle dead-reckoning engine estimating forward displacement and yaw rate from **noisy, unaligned smartphone IMU sensors** during 10-second GNSS outages on the **IO-VNBD (Input-Output Vehicle Navigation Benchmark Dataset)**.

This repository tracks the complete evolutionary development of the models across three distinct generations (**v1 $\to$ v2 $\to$ v3**).

---

## Master Benchmark Matrix (10-Second GNSS Outages)

### 10-Second Dead-Reckoning Final Drift (Meters — Lower is Better)

| Driving Scenario | Sequences | Raw INS (Physics Baseline) | **v1 (Baseline IDNN)** | **v2 (Production IDNN)** | **v3 (PINO-DR)** | **v3 vs Raw INS** | **v3 vs v2** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Motorway** | 7 | 29.99 m | 43.17 m | 10.53 m | **7.13 m** | **+76.2% better** | **+32.3% better** 🏆 |
| **Quick Accel** | 4 | 29.11 m | 35.04 m | 24.57 m | **21.11 m** | **+27.5% better** | **+14.1% better** 🏆 |
| **Hard Brake** | 12 | 43.97 m | 25.98 m | 23.33 m | **17.15 m** | **+61.0% better** | **+26.5% better** 🏆 |
| **Sharp Turns** | 39 | 48.85 m | 47.88 m | 39.88 m | **39.26 m** | **+19.6% better** | **+1.5% better** 🏆 |
| **Roundabout** | 3 | 81.93 m | 74.21 m | 63.45 m | **75.31 m** | **+8.1% better** | -18.7% |

* **Average Displacement Improvement vs Raw INS**: **+41.33%**
* **Average Orientation Improvement vs Raw INS**: **+56.21%**
* **Inference Latency**: **2.08 ms/step** on CPU (well below 10 ms real-time limit)

---

## Architectural Evolution

| Capability | Version 1 (`v1_baseline`) | Version 2 (`v2_production`) | Version 3 (`v3_pino_dr`) — SOTA |
| :--- | :--- | :--- | :--- |
| **Core Architecture** | 2-layer MLP ($32 \to 32$) | 3-layer MLP ($64 \to 64 \to 32$) + BatchNorm + GELU | **PINO-DR**: Conv1D stem + BiGRU + Temporal Attention |
| **Velocity Modeling** | Absolute speed direct regression | Absolute speed with bounded clipping | **Kinematic Residual Skip**: $\hat{v}(t) = v_{\text{prev}}[-1] + \Delta v(t)$ |
| **Data Split Hygiene** | Window-level random split (leaked) | Window-level random split (leaked) | **Trip-Level Split**: 50 train / 9 val / 13 test (zero leakage) |
| **Data Cleaning** | ❌ None (collapsed by 1,843 m/s outlier) | ✅ Physical limits clipping on $[0, 45]$ m/s | ✅ Physical limits clipping + gravity leveling |
| **Zero-Velocity (ZUPT)**| ❌ None | ❌ None | ✅ **Multi-Task Head with Hysteresis Gating** |
| **Deployment Formats** | PyTorch `.pth` | PyTorch, ONNX, TorchScript | **PyTorch, ONNX (opset 18), TorchScript, Engine `.pkl`** |
| **Parameters** | 3,170 | 16,322 | **21,667** (within $\le 25\text{k}$ mobile budget) |
| **INS Comparison** | Worse than INS in 3/5 scenarios | Beats INS in 5/5 scenarios | **Beats INS in 5/5 scenarios with honest zero-leakage** |

---

## Repository Organization

To keep the repository clean and modular, each version is organized into a self-contained directory containing its own code, checkpoints, data splits, and detailed technical documentation:

```
Dead_reckoning__ML_model/
├── README.md                          # Master comparison, navigation & benchmark matrix
├── .gitignore                         # Cache & temporary file exclusions
│
├── v1_baseline/                       # Version 1 (Original Legacy IDNN)
│   ├── README.md                      # Complete analysis of v1 & outlier bug
│   ├── run_pipeline.py                # Pipeline driver for v1
│   ├── src/                           # v1 models and training code
│   ├── checkpoints/                   # v1 model weights
│   ├── data/                          # v1 splits
│   └── results/                       # v1 benchmark plots and reports
│
├── v2_production/                     # Version 2 (Deep IDNN with Domain Bounding)
│   ├── README.md                      # Detailed technical report on v2 & improvements
│   ├── run_pipeline_v2.py             # Pipeline driver for v2
│   ├── src/                           # v2 models, clipping & training code
│   ├── checkpoints/                   # v2 ONNX & PyTorch weights
│   ├── data/                          # v2 clean splits & scalers
│   └── results/                       # v2 benchmark plots and reports
│
├── v3_pino_dr/                        # Version 3 (PINO-DR with Kinematic Residuals) — SOTA
│   ├── README.md                      # Flagship documentation, architecture & metrics
│   ├── run_pipeline_v3.py             # Pipeline driver for v3
│   ├── src/                           # v3 PINO-DR neural operator & engine
│   ├── checkpoints/                   # v3 best models (ONNX, TorchScript, PKL) + epochs/
│   ├── data/                          # v3 trip-level splits & test scenarios
│   └── results/                       # v3 benchmark summaries, trajectory plots & reports
│
└── scripts/                           # Standalone analysis & benchmark utilities
```

---

## Quickstart & Running Models

### 1. Run the State-of-the-Art Model (v3 PINO-DR)
```bash
# Evaluate the pre-trained v3 model on unseen test scenarios
python v3_pino_dr/run_pipeline_v3.py --step eval

# Retrain v3 from scratch (60 epochs with per-epoch checkpoints)
python v3_pino_dr/run_pipeline_v3.py --step train
```

### 2. Run the Production Baseline (v2)
```bash
python v2_production/run_pipeline_v2.py --step evaluate
```

### 3. Run the Legacy Baseline (v1)
```bash
python v1_baseline/run_pipeline.py --step evaluate
```

---

## Production Deployment

For mobile apps (Android/iOS) and embedded automotive C++ systems:
* Use the **ONNX model** at [`v3_pino_dr/checkpoints/best_model.onnx`](file:///c:/Users/tiwar/OneDrive/Desktop/PROJECTS/SIH%2026/v3_pino_dr/checkpoints/best_model.onnx).
* Use the **TorchScript model** at [`v3_pino_dr/checkpoints/best_model_torchscript.pt`](file:///c:/Users/tiwar/OneDrive/Desktop/PROJECTS/SIH%2026/v3_pino_dr/checkpoints/best_model_torchscript.pt).
* Real-time rolling buffer Python engine: [`v3_pino_dr/src/production_engine.py`](file:///c:/Users/tiwar/OneDrive/Desktop/PROJECTS/SIH%2026/v3_pino_dr/src/production_engine.py).
