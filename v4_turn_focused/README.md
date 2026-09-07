# PINO-DR v4: Physics-Informed Neural Odometry (Turn-Focused)

> **Version**: 4.0.0-production  
> **Target Problem**: Calibrating Complex Vehicle Turning Dynamics (Sharp Turns, Roundabouts, S-Curves)  
> **Key Innovations**: 10 Hz Smoothed Yaw Acceleration ($\dot{\omega}$), Centripetal Residual Coupling ($a_{\text{lat}} - v_{\text{pred}} \cdot \omega_{\text{yaw}}$), Directional 2-Head Temporal Attention, Predictive Cross-Task Coupling, Turn-Adaptive Huber Loss, Soft Centripetal Physics Regularizer.  
> **Parameter Footprint**: 21,941 parameters ($\le 25,000$ mobile budget).  

---

## 1. Motivation: The Turning Deficit in v3

While **v3 (PINO-DR)** achieved remarkable performance on straight high-speed segments (reducing Motorway drift to **7.13m**, a **+76.2%** improvement over raw INS), its dead-reckoning accuracy showed notable vulnerabilities during non-linear turning maneuvers:

1. **Sharp Turns**: 10-s drift was 39.26m (only a **+1.5%** marginal improvement over v2's 39.88m), with orientation CRSE degrading from 1.030 rad to 1.068 rad.
2. **Roundabouts**: 10-s drift regressed to 75.31m (**-18.7% worse** than v2's 63.45m), showing orientation drift accumulating rapidly under continuous centripetal acceleration.

### Root Cause Analysis
- **Missing Turn-Initiation Signals**: The model only observed raw yaw rate $\omega_{\text{yaw}}$, lacking the rate-of-change of yaw (yaw acceleration $\dot{\omega}_{\text{yaw}}$) required to detect turn entry/exit transients before large heading changes occur.
- **Uncoupled Predictions**: The displacement head ($\Delta \hat{v}$) and orientation head ($\hat{\omega}$) were predicted independently without explicit physical coupling, ignoring the fundamental kinematic constraint $a_{\text{lat}} \approx v \cdot \omega$.
- **Temporal Attention Collapsing**: A single attention query vector was forced to compromise between steady highway cruising and rapid transient maneuvers.
- **Uniform Loss Weighting**: Heading error during tight turns received the same weight as heading error on straight highway segments.

---

## 2. Mathematical & Architectural Innovations in v4

### A. 10 Hz IMU Preconditioning & Yaw Acceleration
At the raw 10 Hz smartphone IMU domain ($\Delta t = 0.1$ s), we compute smoothed yaw acceleration using a Savitzky-Golay filter (polynomial order 2, window size 5) to eliminate high-frequency differentiation noise without introducing phase delay:
$$\dot{\omega}_{\text{yaw}}[t] = \text{SavitzkyGolay}\left(\frac{\omega_t - \omega_{t-1}}{\Delta t}\right)$$
This gives the network immediate sensitivity to steering onset.

### B. Explicit Centripetal Residual Feature
To provide the small network with direct nonlinear interaction signals without having to learn sensor multiplications from scratch:
$$r_{\text{centripetal}} = a_{\text{lat}} - v_{\text{prev}} \cdot \omega_{\text{yaw}}$$
- **Zero-Leakage Guarantee**: At inference time, $v_{\text{prev}}$ is strictly the autoregressive velocity estimate from the model's previous prediction, **never** ground-truth GPS speed.

### C. Directional 2-Head Temporal Attention
Instead of an unconstrained multi-head projection that would explode parameters beyond budget:
- The BiGRU outputs 64 features (32 forward causal, 32 backward anti-causal).
- **Head 1 ($\mathbf{q}_{\text{fwd}}$)**: Attends over the 32 forward states, capturing steady-state kinematic velocity accumulation.
- **Head 2 ($\mathbf{q}_{\text{bwd}}$)**: Attends over the 32 backward states, capturing transient turn transitions and maneuvers.
- Both heads pool independently and concatenate back to 64 dimensions with **zero projection overhead** (only 64 parameters).

### D. Predicted Cross-Task Coupling
A lightweight residual MLP ($2 \to 16 \to 2$) couples the preliminary velocity update $\hat{v}^{(0)}$ and yaw rate $\hat{\omega}^{(0)}$:
$$\begin{bmatrix} \hat{v} \\ \hat{\omega} \end{bmatrix} = \begin{bmatrix} \hat{v}^{(0)} \\ \hat{\omega}^{(0)} \end{bmatrix} + g_{\text{coupling}}\left(\hat{v}^{(0)}, \hat{\omega}^{(0)}\right)$$
- Initialized with zero weights (`nn.init.zeros_`), ensuring it acts as a pure identity link at initialization.
- Uses **exclusively model predictions** during both training and inference, completely preventing train/test mismatch.

### E. Calibrated Turn-Adaptive Loss Weighting
Using training split statistics ($\sigma_\omega = 0.0723$ rad/s, $\sigma_{\dot{\omega}} = 0.1568$ rad/s²), we compute a composite turn score:
$$S_{\text{turn}} = \sqrt{ \left(\frac{\omega_{\text{yaw}}}{\sigma_\omega}\right)^2 + \left(\frac{\dot{\omega}_{\text{yaw}}}{\sigma_{\dot{\omega}}}\right)^2 }$$
When $S_{\text{turn}} > \tau_{\text{turn}} = 1.1495$ (75th percentile of turning distribution), orientation loss weight is dynamically boosted from $\alpha_{\text{base}} = 1.0$ to $\alpha_{\text{turn}} = 2.0$.

### F. Soft Centripetal Physics Regularizer
Formulated strictly in physical SI units ($\text{m/s}^2$):
$$\mathcal{L}_{\text{physics}} = \text{Huber}_{\delta=1.0}\left(a_{\text{lat, measured}}, \hat{v}_{\text{phys}} \cdot \hat{\omega}_{\text{phys}}\right)$$
with gentle soft regularization weight $\lambda = 0.05$ so sensor tilt or road banking does not corrupt the supervised objective.

---

## 3. Parameter Budget & Efficiency

| Component | Architecture | Parameters |
| :--- | :--- | :---: |
| **Conv1D Stem** | Conv1D(6, 32, k=3) + Conv1D(32, 32, k=3, d=2) + BN + GELU | 3,840 |
| **BiGRU** | BiGRU(32, 32, bidirectional=True, 1 layer) | 12,672 |
| **2-Head Directional Attention** | 2 × 32 Parameter Queries | 64 |
| **Displacement Head** | Linear(64, 32) + GELU + Linear(32, 1) | 2,145 |
| **Orientation Head** | Linear(64, 32) + GELU + Linear(32, 1) | 2,145 |
| **ZUPT Head** | Linear(64, 16) + GELU + Linear(16, 1) | 1,057 |
| **Cross-Task Coupling** | Linear(2, 16) + GELU + Linear(16, 2) | 82 |
| **Total Model Parameters** | — | **21,941** |
| **Mobile Budget Limit** | — | $\le 25,000$ (3,059 margin) |

---

## 4. Pipeline Execution & Ablations

```bash
# Full Pipeline (Preprocess -> Train -> Evaluate -> Report)
python run_pipeline_v4.py

# Or step-by-step
python run_pipeline_v4.py --step preprocess
python run_pipeline_v4.py --step train --ablation E
python run_pipeline_v4.py --step eval
python run_pipeline_v4.py --step report
```
