# Version 1 (v1) — Baseline Input Delay Neural Network (IDNN)

## Executive Summary
`v1_baseline` represents the original, initial implementation of the Input Delay Neural Network (IDNN) for smartphone-based automotive dead reckoning, inspired by Onyekpe et al. (2021). 

While the model achieved acceptable heading estimation compared to raw inertial sensors, **it suffered from a catastrophic data-scaling bug that caused its forward displacement estimation to collapse**, resulting in performance **worse than pure physics-based INS dead reckoning in 3 out of 5 driving scenarios**.

---

## 1. Architecture & Model Specification

The v1 system consists of two decoupled, single-step feedforward neural networks:

### Displacement Predictor (`DisplacementIDNN`)
* **Input**: 20 scalar features (10 historical forward acceleration values $[a_{t-9}, \dots, a_t]$ concatenated with 10 historical displacement/velocity values $[v_{t-9}, \dots, v_t]$).
* **Architecture**:
  $$\text{Linear}(20 \to 32) \to \text{ReLU} \to \text{Linear}(32 \to 32) \to \text{ReLU} \to \text{Linear}(32 \to 1)$$
* **Parameters**: 1,761 trainable parameters.
* **Target**: Forward displacement / vehicle speed $v_t$ over a 1-second interval.

### Orientation Rate Predictor (`OrientationIDNN`)
* **Input**: 10 scalar features (10 historical gyroscope yaw rate values $[\omega_{t-9}, \dots, \omega_t]$).
* **Architecture**:
  $$\text{Linear}(10 \to 32) \to \text{ReLU} \to \text{Linear}(32 \to 32) \to \text{ReLU} \to \text{Linear}(32 \to 1)$$
* **Parameters**: 1,409 trainable parameters.
* **Target**: Vehicle yaw rate $\omega_t$ over a 1-second interval.

---

## 2. The Fatal Outlier Bug (Root Cause Analysis)

### The 1,843.7 m/s GPS Outlier
When preprocessing the IO-VNBD dataset in v1, features and targets were scaled using `sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))`. 

During the fit pass across the dataset, a GPS sensor glitch in one journey recorded an instantaneous speed jump to **1,843.696 m/s (~6,637 km/h)**. 

### Why It Broke the Model:
1. `MinMaxScaler` maps $[0, \text{max\_observed}] \to [0, 1]$.
2. The 99.99th percentile of realistic road driving speeds in the dataset is **~36.4 m/s (~131 km/h)**, with a median of **10.8 m/s**.
3. Because the maximum was set to $1,843.7$, realistic driving speeds were compressed into the tiny numeric band:
   $$\text{Target}_{\text{scaled}} \in [0.000, 0.019]$$
4. Gradient descent under Mean Squared Error (MSE) quickly determined that predicting a near-constant normalized value of $\approx 0.0065$ (which maps back to $\approx 12\text{ m/s}$ in real units) minimized the global loss.
5. Consequently, **the model learned to output ~12 m/s regardless of whether the vehicle was stopped at a red light (0 m/s) or cruising on the highway (30 m/s)**.

---

## 3. Benchmark Evaluation (10-Second GNSS Outages)

Evaluated across the 5 official IO-VNBD test scenarios against raw Inertial Navigation System (INS) dead reckoning:

| Scenario | Sequences | Raw INS Drift (10s) | v1 Baseline Drift (10s) | v1 vs Raw INS | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Motorway** | 7 | 29.99 m | **43.17 m** | **-43.9% (Worse)** | ❌ Severe failure |
| **Roundabout** | 3 | 81.93 m | **74.21 m** | **+9.4% (Better)** | ⚠️ Marginal |
| **Quick Accel** | 4 | 29.11 m | **35.04 m** | **-20.4% (Worse)** | ❌ Severe failure |
| **Hard Brake** | 12 | 43.97 m | **25.98 m** | **+40.9% (Better)** | ⚠️ Partial |
| **Sharp Turns** | 39 | 48.85 m | **47.88 m** | **+2.0% (Better)** | ⚠️ Marginal |

### Cumulative Residual Squared Error (CRSE)

#### Displacement CRSE
* **Motorway**: INS = 19.60 m vs **v1 = 42.62 m (-117.4% worse)**. Because actual speed was ~28 m/s but v1 predicted ~12 m/s, it fell behind by 16 meters every single second!
* **Quick Accel**: INS = 18.87 m vs **v1 = 32.18 m (-70.5% worse)**. During rapid acceleration, the model could not ramp up velocity.
* **Roundabout**: INS = 46.07 m vs **v1 = 33.69 m (+26.9%)**.
* **Hard Brake**: INS = 24.86 m vs **v1 = 25.72 m (-3.5%)**.
* **Sharp Turns**: INS = 38.86 m vs **v1 = 29.97 m (+22.9%)**.

#### Orientation CRSE
* **Motorway**: INS = 0.693 rad vs **v1 = 0.060 rad (+91.3% improvement)**.
* **Roundabout**: INS = 1.912 rad vs **v1 = 1.987 rad (-3.9% worse)**.
* **Quick Accel**: INS = 0.517 rad vs **v1 = 0.161 rad (+68.8% improvement)**.
* **Hard Brake**: INS = 0.900 rad vs **v1 = 0.175 rad (+80.5% improvement)**.
* **Sharp Turns**: INS = 1.383 rad vs **v1 = 1.030 rad (+25.5% improvement)**.

---

## 4. Key Takeaways & Why v2 Was Built

1. **Orientation was promising**: Gyroscope bias modeling via neural network proved effective, outperforming pure INS heading integration in 4 of 5 scenarios.
2. **Displacement was broken**: Due to the outlier scaling collapse, v1 was unusable for real-world navigation.
3. **No Zero Velocity Update (ZUPT)**: When vehicles stopped at traffic signals or hard brakes, v1 continued predicting ~12 m/s forward motion, causing phantom displacement.
4. **Window-level leakage**: Train/test splitting after sliding windows introduced high overlap.

---

## 5. Directory Structure & Execution

```
v1_baseline/
├── checkpoints/
│   ├── displacement_idnn.pth
│   └── orientation_idnn.pth
├── data/
│   ├── dataset_splits.npz
│   ├── scalers.pkl
│   └── test_scenarios.pkl
├── results/
│   ├── benchmark_summary.json
│   ├── model_quality_report.txt
│   ├── mae_curve.png
│   └── trajectory_*.png
├── src/
│   ├── dataset_loader.py
│   ├── evaluate.py
│   ├── models.py
│   └── train.py
└── run_pipeline.py
```

### Reproducing v1:
```bash
# Run benchmark evaluation
python run_pipeline.py --step evaluate

# Retrain v1 models
python run_pipeline.py --step train
```
