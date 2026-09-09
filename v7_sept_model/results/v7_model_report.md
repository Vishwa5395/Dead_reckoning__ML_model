# PINO-DR v7 (5-Supreme-Specialists System) Official Benchmark Report

## 1. Executive Summary

The **5-Supreme-Specialists System** achieves a **landmark all-time project record of 21.47 m overall dead-reckoning drift** across 10-second closed-loop GNSS outages across all 65 standard test sequences. This outperforms the previous repository baseline (29.55 m) by **+27.4%** and cuts pure INS error (46.21 m) by **+53.5%**.

### Key Breakthroughs
1. **Roundabout Target Met**: Smashed from 55.36 m (v4-D) and 75.31 m (v3) down to **21.23 m** (**+61.7% error reduction** vs v4-D and **+71.8% vs v3**). The $2\times$ target was **$\le 27.68$ m** — our system beat the target by **6.45 meters**!
   - Outage 1: **2.38 m**
   - Outage 2: **45.12 m** (down from 98.29 m / 122.31 m baseline)
   - Outage 3: **16.17 m**
2. **Sharp Turns**: Slashed from 35.42 m baseline down to **27.65 m** (**+21.9% error reduction**, beating baseline by **7.77 meters**).
   - Resolved phone mounting lateral axis inversion by anchoring turn direction to the gyroscope ($\text{sign}(\omega_{\text{yaw}})$) while scaling magnitude via centripetal acceleration ($|a_{\text{lat}}| / v$).
   - Implemented cornering speed decay matching vehicle dynamics.
   - Refined entry deceleration momentum to permit stops at red lights while allowing acceleration away into traffic.
3. **Hard Brake**: Slashed from 14.74 m baseline down to **13.80 m** (**+6.4% error reduction**, beating baseline).
   - Integrated longitudinal deceleration physics ($a_{\text{fwd}} < -0.20$ m/s²), allowing active braking to overcome neural network cruising speed anchors.
   - Slashed stop-event Outage 5 from 18.30m down to **2.81 m**.
   - 4 out of 12 outages now beat the $2\times$ target of 7.37 m (Outage 1 at **1.37 m**, Outage 2 at **3.51 m**, Outage 5 at **2.81 m**, Outage 8 at **5.36 m**).
4. **Quick Accel**: Cut from 18.50 m baseline down to **9.41 m** (**+49.1% error reduction**), within **0.16 m (16 cm) of the $2\times$ target** (9.25 m).
5. **Motorway Anchor**: Preserved at **7.13 m** (2.8% error rate over 250m travel distance).

---

## 2. Comprehensive 5-Scenario Benchmark (10-Second GNSS Outages)

| Scenario | Sequences | Best in Repo (Baseline) | $2\times$ Target | Supreme Specialist (Ours) | Improvement vs Repo | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Roundabout** | 3 | 55.36 m (v4) / 75.31 m (v3) | **27.68 m** | **21.23 m** | **+61.7% vs v4 (+71.8% vs v3)** | **MET 2X TARGET! (Beat by 6.45m)** |
| **Quick Accel** | 4 | 18.50 m | **9.25 m** | **9.41 m** | **+49.1%** | **Within 0.16 m (16 cm) of $2\times$ target!** |
| **Hard Brake** | 12 | 14.74 m | **7.37 m** | **13.80 m** | **+6.4%** | **Beats Repo Baseline (4 outages beat $2\times$)** |
| **Sharp Turns** | 39 | 35.42 m | **17.71 m** | **27.65 m** | **+21.9%** | **New Repo Best (Down by 7.77m!)** |
| **Motorway** | 7 | 7.12 m | **3.56 m** | **7.13 m** | **Matched** | Cruising Anchor (2.8% error) |
| **OVERALL MEAN**| **65** | **29.55 m** | **14.77 m** | **21.47 m** | **+27.4%** | **NEW ALL-TIME REPO RECORD** |

---

## 3. Multi-Horizon Drift Progression

| Time Horizon | Global Mean Drift (m) |
| :--- | :--- |
| **1-Second Drift** | **0.99 m** |
| **3-Second Drift** | **4.16 m** |
| **5-Second Drift** | **8.43 m** |
| **10-Second Drift** | **21.47 m** |

---

## 4. Key Physical Kinematic Insights

1. **Sharp Turns (Gyroscope Directional Anchoring & Centripetal Assistance)**:
   In smartphone datasets, device mounts rotate between horizontal and vertical, sometimes inverting the lateral accelerometer axis. Guiding turn direction using the gyroscope yaw rate sign ($\text{sign}(\omega_{\text{yaw}})$) while reinforcing angular rate via centripetal acceleration ($|a_{\text{lat}}| / v_{\text{est}}$) eliminated 180° inverted heading errors and cut Sharp Turns mean drift to **27.65 m** (beating the baseline by **7.77 m**).
2. **Hard Brake (Longitudinal Deceleration Integration)**:
   Heading error is almost zero during straight braking; 100% of the drift is speed error. Neural networks trained with autoregressive velocity feedback ($v_{\text{prev}}$) resist dropping speed quickly. Coupling physical forward acceleration step integration ($a_{\text{fwd}} < -0.20$ m/s²) with pre-outage deceleration momentum cut Hard Brake drift to **13.80 m**, with 4 outages smashing the 7.37m target.
3. **Roundabout (Rotated Phone Centripetal Vector Invariance)**:
   When a smartphone is mounted sideways in a car cradle, its forward accelerometer measures lateral centripetal acceleration ($a_{\text{fwd}} \approx +2.5$ m/s²). Deriving yaw rate from the horizontal acceleration vector norm $\|\mathbf{a}_{\text{horiz}}\| / v$ and preventing false forward acceleration dropped Roundabout drift from 55.36m down to **21.23 m**, smashing the $2\times$ target!
