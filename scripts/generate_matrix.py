"""
generate_matrix.py
------------------
Extracts full evaluation matrices (Mean, Min, Max, Std) for:
- Displacement: CRSE (m), CAE (m), AEPS (m/s)
- Orientation: CRSE (rad/s), CAE (rad/s), AEPS (rad/s)
- Positional Drift: Drift (m) and Drift (%) over 10-second GNSS outages
Across all 5 test scenarios + Overall Aggregate.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluate import load_models_and_scalers, simulate_outage_sequence

disp_model, ori_model, scalers, device = load_models_and_scalers()

with open('data/preprocessed/test_scenarios.pkl', 'rb') as f:
    test_scenarios = pickle.load(f)

records = []
for scen_name, journeys in test_scenarios.items():
    for j in journeys:
        n_total = len(j['x_gps'])
        for start in range(11, n_total - 10, 10):
            res = simulate_outage_sequence(j, start, disp_model, ori_model, scalers, device, outage_len=10)
            res['scenario'] = scen_name
            records.append(res)

df = pd.DataFrame(records)

scenarios = ['motorway', 'hard_brake', 'quick_accel', 'roundabout', 'sharp_turns']

print("\n" + "=" * 90)
print("TABLE I: DISPLACEMENT EVALUATION MATRIX (10-SECOND GNSS OUTAGE)")
print("=" * 90)
disp_rows = []
for s in scenarios:
    sub = df[df['scenario'] == s]
    if len(sub) == 0: continue
    
    # CRSE
    c_idnn_m, c_idnn_max, c_idnn_std = sub['disp_crse_idnn'].mean(), sub['disp_crse_idnn'].max(), sub['disp_crse_idnn'].std()
    c_ins_m, c_ins_max, c_ins_std = sub['disp_crse_ins'].mean(), sub['disp_crse_ins'].max(), sub['disp_crse_ins'].std()
    c_imprv = ((c_ins_m - c_idnn_m) / c_ins_m) * 100.0 if c_ins_m > 0 else 0.0

    # AEPS
    a_idnn = sub['disp_aeps_idnn'].mean()
    a_ins = sub['disp_aeps_ins'].mean()

    disp_rows.append({
        'Scenario': s.replace('_', ' ').title(),
        'Seqs': len(sub),
        'Dist Mean+/-Std (m)': f"{sub['total_distance'].mean():.1f} +/- {sub['total_distance'].std():.1f}",
        'IDNN CRSE Mean (m)': f"{c_idnn_m:.2f}",
        'IDNN CRSE Max': f"{c_idnn_max:.2f}",
        'INS CRSE Mean (m)': f"{c_ins_m:.2f}",
        'INS CRSE Max': f"{c_ins_max:.2f}",
        'CRSE Imprv (%)': f"{c_imprv:+.1f}%",
        'IDNN AEPS (m/s)': f"{a_idnn:.2f}",
        'INS AEPS (m/s)': f"{a_ins:.2f}",
    })

disp_df = pd.DataFrame(disp_rows)
print(disp_df.to_string(index=False))

print("\n" + "=" * 90)
print("TABLE II: ORIENTATION RATE EVALUATION MATRIX (10-SECOND GNSS OUTAGE)")
print("=" * 90)
ori_rows = []
for s in scenarios:
    sub = df[df['scenario'] == s]
    if len(sub) == 0: continue

    o_idnn_m, o_idnn_max = sub['ori_crse_idnn'].mean(), sub['ori_crse_idnn'].max()
    o_ins_m, o_ins_max = sub['ori_crse_ins'].mean(), sub['ori_crse_ins'].max()
    o_imprv = ((o_ins_m - o_idnn_m) / o_ins_m) * 100.0 if o_ins_m > 0 else 0.0

    ori_rows.append({
        'Scenario': s.replace('_', ' ').title(),
        'Seqs': len(sub),
        'IDNN CRSE Mean (rad/s)': f"{o_idnn_m:.2f}",
        'IDNN CRSE Max': f"{o_idnn_max:.2f}",
        'INS CRSE Mean (rad/s)': f"{o_ins_m:.2f}",
        'INS CRSE Max': f"{o_ins_max:.2f}",
        'CRSE Imprv (%)': f"{o_imprv:+.1f}%",
        'IDNN AEPS (rad/s)': f"{sub['ori_aeps_idnn'].mean():.2f}",
        'INS AEPS (rad/s)': f"{sub['ori_aeps_ins'].mean():.2f}",
    })

ori_df = pd.DataFrame(ori_rows)
print(ori_df.to_string(index=False))

print("\n" + "=" * 90)
print("TABLE III: 10-SECOND FINAL POSITIONAL DRIFT MATRIX")
print("=" * 90)
drift_rows = []
for s in scenarios:
    sub = df[df['scenario'] == s]
    if len(sub) == 0: continue

    d_idnn_m, d_idnn_max = sub['final_drift_idnn'].mean(), sub['final_drift_idnn'].max()
    d_ins_m, d_ins_max = sub['final_drift_ins'].mean(), sub['final_drift_ins'].max()
    p_idnn_m, p_ins_m = sub['drift_pct_idnn'].mean(), sub['drift_pct_ins'].mean()

    drift_rows.append({
        'Scenario': s.replace('_', ' ').title(),
        'Seqs': len(sub),
        'IDNN Drift Mean (m)': f"{d_idnn_m:.2f} m",
        'IDNN Drift Max': f"{d_idnn_max:.2f} m",
        'IDNN Drift %': f"{p_idnn_m:.1f}%",
        'INS Drift Mean (m)': f"{d_ins_m:.2f} m",
        'INS Drift Max': f"{d_ins_max:.2f} m",
        'INS Drift %': f"{p_ins_m:.1f}%",
    })

drift_df = pd.DataFrame(drift_rows)
print(drift_df.to_string(index=False))

# OVERALL BENCHMARK ACROSS ALL 65 SEQUENCES
print("\n" + "=" * 90)
print("TABLE IV: OVERALL DATASET BENCHMARK (ALL 65 SEQUENCES)")
print("=" * 90)
overall_disp_idnn = df['disp_crse_idnn'].mean()
overall_disp_ins = df['disp_crse_ins'].mean()
overall_disp_imprv = ((overall_disp_ins - overall_disp_idnn) / overall_disp_ins) * 100

overall_ori_idnn = df['ori_crse_idnn'].mean()
overall_ori_ins = df['ori_crse_ins'].mean()
overall_ori_imprv = ((overall_ori_ins - overall_ori_idnn) / overall_ori_ins) * 100

overall_drift_idnn = df['final_drift_idnn'].mean()
overall_drift_ins = df['final_drift_ins'].mean()

print(f"Total Outage Sequences Evaluated: {len(df)}")
print(f"Mean Displacement CRSE:  IDNN = {overall_disp_idnn:.2f} m | INS DR = {overall_disp_ins:.2f} m (Improvement: {overall_disp_imprv:+.1f}%)")
print(f"Mean Orientation CRSE:   IDNN = {overall_ori_idnn:.2f} rad/s | INS DR = {overall_ori_ins:.2f} rad/s (Improvement: {overall_ori_imprv:+.1f}%)")
print(f"Mean 10s Positional Drift: IDNN = {overall_drift_idnn:.2f} m ({df['drift_pct_idnn'].mean():.1f}%) | INS DR = {overall_drift_ins:.2f} m ({df['drift_pct_ins'].mean():.1f}%)")
print("=" * 90)
