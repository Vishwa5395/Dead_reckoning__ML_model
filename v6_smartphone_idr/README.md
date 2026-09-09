# PINO-DR v6: Smartphone-Only Intelligent Dead Reckoning with Probabilistic Multi-Sensor Fusion

> **Version**: 6.0.0-production  
> **Target Problem**: Smartphone-Only Automotive Dead Reckoning (10 Hz Native Processing)  
> **Key Innovations**:  
> 1. 10 Hz Native Causal Temporal TCN-BiGRU Architecture ($\Delta t = 0.1\text{ s}$)  
> 2. 6-State Extended Kalman Filter (EKF) with Heteroscedastic Uncertainty Weighting  
> 3. Zero Velocity Update (ZUPT) with Hysteresis Lock  
> 4. Soft Non-Holonomic Constraints (NHC) & Centripetal Physics Consistency  
> 5. Multi-Hypothesis Probabilistic HMM Map Matching with Road Network Geometry  
> 6. Online Accelerometer & Gyroscope Bias Estimation  
> 7. Zero Leakage: Strict Frozen 50/9/13 Trip Split  
> **Mobile Budget**: 24,601 parameters ($\le 25,000$ constraint, $\sim 2.5\text{ ms}$ CPU latency)

---

## 1. System Architecture Overview

```
                         10 Hz Smartphone IMU Stream (No Wires / No OBD)
                                      │
                                      ▼
                        In-Vehicle Alignment & Leveling
                     (Dynamic Gravity Projection + Forward Covariance)
                                      │
                                      ▼
                        6-Channel 10 Hz Input Tensor (20 Steps)
               [a_fwd, w_yaw, a_lat, v_prev, w_yaw_accel, centripetal_residual]
                                      │
                                      ▼
                       Causal Conv1D Dilated Stem (d=1, d=2)
                                      │
                                      ▼
                      Bidirectional GRU (Hidden: 32/dir -> 64)
                                      │
                                      ▼
                      Directional 2-Head Temporal Attention
                          (q_fwd: Steady, q_bwd: Dynamic)
                                      │
                                      ▼
                     Predicted Cross-Task Coupling (2 -> 16 -> 2)
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
      Kinematic Delta v           Yaw Rate ω           Sensor Biases (ba, bw)
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
                        Deterministic Physics Validator
                       (Soft NHC + Centripetal Check)
                                      │
                                      ▼
                         6-State Extended Kalman Filter
                     [px, py, v, psi, b_accel, b_gyro]
                                      │
                                      ▼
                       ZUPT Standstill Hysteresis Lock
                                      │
                                      ▼
                       Probabilistic HMM Map Matcher
                    (Multi-Hypothesis Soft Road Snapping)
                                      │
                                      ▼
                 Continuous Lane-Level Trajectory Output (10 Hz)
```

---

## 2. Directory Structure

```
v6_smartphone_idr/
├── config/
│   └── v6_config.json               # Master hyperparameters & sensor limits
├── src/
│   ├── preprocess_v6.py             # 10 Hz continuous IMU pipeline & frozen split
│   ├── augmentation_v6.py           # Jitter, shocks, rumble & maneuver weighting
│   ├── models_v6.py                 # Causal Conv1D + BiGRU + Attention + Heads
│   ├── physics_engine_v6.py         # Deterministic kinematics & NHC validator
│   ├── ekf_state_estimator_v6.py    # 6-state EKF with uncertainty weighting
│   ├── map_matcher_v6.py            # HMM multi-hypothesis road matcher
│   ├── train_v6.py                  # Multi-task training engine
│   ├── evaluate_v6.py               # 65-sequence benchmark & ablation runner
│   ├── hyperparameter_search_v6.py  # Validation-only hyperparameter optimization
│   ├── failure_analysis_v6.py       # Diagnostic error attribution
│   └── export_onnx_v6.py            # ONNX opset 18 export & CPU profiler
├── data/                            # Processed splits & test scenario pickles
├── checkpoints/                     # Model weights (.pth, .onnx, .pt)
├── results/                         # JSON summaries & trajectory PNGs
└── run_pipeline_v6.py               # Master CLI driver
```

---

## 3. Quickstart & Commands

```bash
# 1. Run Complete V6 Pipeline (Preprocess -> Train -> Ablation -> Report -> Export)
python run_pipeline_v6.py --step all

# 2. Evaluate Full V6 on the 65 Test Outages
python run_pipeline_v6.py --step eval --mode H

# 3. Run the Ablation Suite (B through H)
python run_pipeline_v6.py --step ablation

# 4. Export ONNX & Profile Mobile Latency
python run_pipeline_v6.py --step export
```
