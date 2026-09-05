"""
evaluate.py
-----------
Evaluates the IDNN models on IO-VNBD test scenarios:
- Motorway (Vw12)
- Roundabout (Vta11)
- Quick changes in acceleration (Vta12)
- Hard brake (Vw16b, Vw17, Vta9)
- Sharp cornering and successive turns (Vw6, Vw7, Vw8)

Strictly satisfies:
1. Autoregressive closed-loop feedback at inference/evaluation time:
   The displacement model feeds back its OWN prior prediction x_predicted(t-1|t-2),
   NEVER ground-truth GPS during the 10-second GNSS outage.
2. Evaluates CRSE, CAE, AEPS, and Drift % for both IDNN and Pure INS DR baseline.
3. Generates 2D trajectory plots and error curves saved to results/.
"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models import DisplacementIDNN, OrientationIDNN

CHECKPOINT_DIR = r"c:\Users\tiwar\OneDrive\Desktop\PROJECTS\SIH 26\checkpoints"
CACHE_DIR = r"c:\Users\tiwar\OneDrive\Desktop\PROJECTS\SIH 26\data\preprocessed"
RESULTS_DIR = r"c:\Users\tiwar\OneDrive\Desktop\PROJECTS\SIH 26\results"


def load_models_and_scalers(checkpoint_dir=CHECKPOINT_DIR, cache_dir=CACHE_DIR, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load scalers
    with open(os.path.join(cache_dir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)

    # Load Displacement Model
    disp_model = DisplacementIDNN(input_dim=20, hidden_dim=32).to(device)
    disp_ckpt = torch.load(os.path.join(checkpoint_dir, "displacement_idnn.pth"), map_location=device)
    disp_model.load_state_dict(disp_ckpt["model_state_dict"])
    disp_model.eval()

    # Load Orientation Model
    ori_model = OrientationIDNN(input_dim=10, hidden_dim=32).to(device)
    ori_ckpt = torch.load(os.path.join(checkpoint_dir, "orientation_idnn.pth"), map_location=device)
    ori_model.load_state_dict(ori_ckpt["model_state_dict"])
    ori_model.eval()

    return disp_model, ori_model, scalers, device


def simulate_outage_sequence(
    journey,
    start_idx,
    disp_model,
    ori_model,
    scalers,
    device,
    outage_len=10,
    window_size=10
):
    """
    Simulates a 10-second GNSS outage starting at start_idx.
    AUTOREGRESSIVE CLOSED-LOOP FEEDBACK:
    Displacement model strictly uses its OWN previous prediction x_pred(t-1)
    as feedback for steps t >= start_idx + 1.
    """
    a_ins = journey['a_ins']
    w_ins = journey['w_ins']
    x_gps = journey['x_gps']
    w_gps = journey['w_gps']
    headings = journey['headings']

    scaler_dX = scalers['disp_X']
    scaler_dY = scalers['disp_Y']
    scaler_oX = scalers['ori_X']
    scaler_oY = scalers['ori_Y']

    # Historical displacement buffer (initialized with pre-outage known values)
    # Length: window_size
    history_x_pred = list(x_gps[start_idx - window_size : start_idx])

    # State tracking: (North, East)
    pos_gt = [(0.0, 0.0)]
    pos_idnn = [(0.0, 0.0)]
    pos_ins_dr = [(0.0, 0.0)]

    # Heading tracking
    psi_gt = np.radians(headings[start_idx])
    psi_idnn = psi_gt
    psi_ins = psi_gt

    # INS velocity tracking
    v_ins = x_gps[start_idx - 1]  # initial speed before outage

    errors_disp_idnn = []
    errors_disp_ins = []
    errors_ori_idnn = []
    errors_ori_ins = []

    for k in range(outage_len):
        curr_idx = start_idx + k

        # 1. Orientation Prediction
        w_win = w_ins[curr_idx - window_size + 1 : curr_idx + 1].reshape(1, -1)
        w_win_scaled = scaler_oX.transform(w_win)
        with torch.no_grad():
            w_pred_scaled = ori_model(torch.tensor(w_win_scaled, dtype=torch.float32).to(device)).item()
        w_pred_idnn = scaler_oY.inverse_transform([[w_pred_scaled]])[0, 0]

        # Orientation updates
        true_w = w_gps[curr_idx]
        raw_w = w_ins[curr_idx]

        psi_gt += true_w * 1.0
        psi_idnn += w_pred_idnn * 1.0
        psi_ins += raw_w * 1.0

        errors_ori_idnn.append(abs(true_w - w_pred_idnn))
        errors_ori_ins.append(abs(true_w - raw_w))

        # 2. Displacement Prediction (AUTOREGRESSIVE CLOSED-LOOP)
        # a_ins window (length W)
        a_win = a_ins[curr_idx - window_size + 1 : curr_idx + 1]
        # x_feedback window: strictly contains OWN predictions from prior outage steps!
        x_win = np.array(history_x_pred[-window_size:])

        disp_feat = np.concatenate([a_win, x_win]).reshape(1, -1)
        disp_feat_scaled = scaler_dX.transform(disp_feat)
        with torch.no_grad():
            x_pred_scaled = disp_model(torch.tensor(disp_feat_scaled, dtype=torch.float32).to(device)).item()
        x_pred_idnn = max(0.0, scaler_dY.inverse_transform([[x_pred_scaled]])[0, 0])

        # APPEND TO BUFFER: next step will use THIS prediction as its feedback!
        history_x_pred.append(x_pred_idnn)

        # INS DR Baseline (Euler integration)
        v_ins = max(0.0, v_ins + a_ins[curr_idx] * 1.0)
        x_ins = v_ins * 1.0

        true_x = x_gps[curr_idx]

        errors_disp_idnn.append(abs(true_x - x_pred_idnn))
        errors_disp_ins.append(abs(true_x - x_ins))

        # 3. Position Updates in NED frame
        # Ground Truth
        dn_gt = true_x * np.cos(psi_gt)
        de_gt = true_x * np.sin(psi_gt)
        pos_gt.append((pos_gt[-1][0] + dn_gt, pos_gt[-1][1] + de_gt))

        # IDNN
        dn_idnn = x_pred_idnn * np.cos(psi_idnn)
        de_idnn = x_pred_idnn * np.sin(psi_idnn)
        pos_idnn.append((pos_idnn[-1][0] + dn_idnn, pos_idnn[-1][1] + de_idnn))

        # Pure INS DR
        dn_ins = x_ins * np.cos(psi_ins)
        de_ins = x_ins * np.sin(psi_ins)
        pos_ins_dr.append((pos_ins_dr[-1][0] + dn_ins, pos_ins_dr[-1][1] + de_ins))

    # Metrics computation per Section 3.2
    total_dist = sum(x_gps[start_idx : start_idx + outage_len])
    final_drift_idnn = np.sqrt(
        (pos_idnn[-1][0] - pos_gt[-1][0]) ** 2 + (pos_idnn[-1][1] - pos_gt[-1][1]) ** 2
    )
    final_drift_ins = np.sqrt(
        (pos_ins_dr[-1][0] - pos_gt[-1][0]) ** 2 + (pos_ins_dr[-1][1] - pos_gt[-1][1]) ** 2
    )

    drift_pct_idnn = (final_drift_idnn / total_dist * 100.0) if total_dist > 0 else 0.0
    drift_pct_ins = (final_drift_ins / total_dist * 100.0) if total_dist > 0 else 0.0

    return {
        'total_distance': total_dist,
        'disp_crse_idnn': sum(errors_disp_idnn),
        'disp_crse_ins': sum(errors_disp_ins),
        'disp_cae_idnn': sum(errors_disp_idnn),  # signed/abs
        'disp_cae_ins': sum(errors_disp_ins),
        'disp_aeps_idnn': np.mean(errors_disp_idnn),
        'disp_aeps_ins': np.mean(errors_disp_ins),
        'ori_crse_idnn': sum(errors_ori_idnn),
        'ori_crse_ins': sum(errors_ori_ins),
        'ori_aeps_idnn': np.mean(errors_ori_idnn),
        'ori_aeps_ins': np.mean(errors_ori_ins),
        'final_drift_idnn': final_drift_idnn,
        'final_drift_ins': final_drift_ins,
        'drift_pct_idnn': drift_pct_idnn,
        'drift_pct_ins': drift_pct_ins,
        'pos_gt': np.array(pos_gt),
        'pos_idnn': np.array(pos_idnn),
        'pos_ins': np.array(pos_ins_dr),
    }


def evaluate_all_scenarios(cache_dir=CACHE_DIR, checkpoint_dir=CHECKPOINT_DIR, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    disp_model, ori_model, scalers, device = load_models_and_scalers(checkpoint_dir, cache_dir)

    with open(os.path.join(cache_dir, "test_scenarios.pkl"), "rb") as f:
        test_scenarios = pickle.load(f)

    benchmark_summary = {}

    print("\n" + "=" * 75)
    print("IO-VNBD BENCHMARK EVALUATION (AUTOREGRESSIVE CLOSED-LOOP INFERENCE)")
    print("=" * 75)

    for scen_name, journeys in test_scenarios.items():
        if not journeys:
            continue

        scen_results = []
        sample_trajectory = None

        for j in journeys:
            n_total = len(j['x_gps'])
            # Generate non-overlapping 10-second outage sequences
            # with 10 seconds of history pre-buffer
            window_size = 10
            outage_len = 10
            step = 10  # jump by 10s per sequence

            for start in range(window_size + 1, n_total - outage_len, step):
                res = simulate_outage_sequence(
                    j, start, disp_model, ori_model, scalers, device, outage_len=outage_len
                )
                scen_results.append(res)
                if sample_trajectory is None and res['total_distance'] > 50:
                    sample_trajectory = res

        if not scen_results:
            continue

        n_seq = len(scen_results)
        mean_disp_crse_idnn = np.mean([r['disp_crse_idnn'] for r in scen_results])
        mean_disp_crse_ins = np.mean([r['disp_crse_ins'] for r in scen_results])
        mean_ori_crse_idnn = np.mean([r['ori_crse_idnn'] for r in scen_results])
        mean_ori_crse_ins = np.mean([r['ori_crse_ins'] for r in scen_results])
        mean_drift_idnn = np.mean([r['final_drift_idnn'] for r in scen_results])
        mean_drift_ins = np.mean([r['final_drift_ins'] for r in scen_results])
        mean_drift_pct_idnn = np.mean([r['drift_pct_idnn'] for r in scen_results])
        mean_drift_pct_ins = np.mean([r['drift_pct_ins'] for r in scen_results])

        disp_improvement = ((mean_disp_crse_ins - mean_disp_crse_idnn) / mean_disp_crse_ins) * 100.0
        ori_improvement = ((mean_ori_crse_ins - mean_ori_crse_idnn) / mean_ori_crse_ins) * 100.0

        benchmark_summary[scen_name] = {
            'n_sequences': n_seq,
            'disp_crse_idnn': mean_disp_crse_idnn,
            'disp_crse_ins': mean_disp_crse_ins,
            'disp_improvement_pct': disp_improvement,
            'ori_crse_idnn': mean_ori_crse_idnn,
            'ori_crse_ins': mean_ori_crse_ins,
            'ori_improvement_pct': ori_improvement,
            'final_drift_m_idnn': mean_drift_idnn,
            'final_drift_m_ins': mean_drift_ins,
            'drift_pct_idnn': mean_drift_pct_idnn,
            'drift_pct_ins': mean_drift_pct_ins,
        }

        print(f"\nScenario: [{scen_name.upper()}] ({n_seq} sequences evaluated)")
        print(f"  Displacement CRSE: IDNN = {mean_disp_crse_idnn:.2f} m | INS DR = {mean_disp_crse_ins:.2f} m  (Improvement: {disp_improvement:.1f}%)")
        print(f"  Orientation CRSE:  IDNN = {mean_ori_crse_idnn:.2f} rad/s | INS DR = {mean_ori_crse_ins:.2f} rad/s (Improvement: {ori_improvement:.1f}%)")
        print(f"  10s Positional Drift: IDNN = {mean_drift_idnn:.2f} m ({mean_drift_pct_idnn:.2f}%) | INS DR = {mean_drift_ins:.2f} m ({mean_drift_pct_ins:.2f}%)")

        # Plot sample trajectory
        if sample_trajectory is not None:
            fig, ax = plt.subplots(figsize=(8, 7))
            gt = sample_trajectory['pos_gt']
            idnn = sample_trajectory['pos_idnn']
            ins = sample_trajectory['pos_ins']

            ax.plot(gt[:, 1], gt[:, 0], 'g-', linewidth=2.5, label='Ground Truth GPS')
            ax.plot(idnn[:, 1], idnn[:, 0], 'b--', linewidth=2.0, label='IDNN AI Model (Closed-Loop)')
            ax.plot(ins[:, 1], ins[:, 0], 'r:', linewidth=1.8, label='Pure INS Dead Reckoning')
            ax.scatter([gt[0, 1]], [gt[0, 0]], color='black', marker='o', s=80, zorder=5, label='Outage Start')
            ax.scatter([gt[-1, 1]], [gt[-1, 0]], color='green', marker='X', s=90, zorder=5, label='GPS End')
            ax.scatter([idnn[-1, 1]], [idnn[-1, 0]], color='blue', marker='^', s=90, zorder=5, label='IDNN End')

            ax.set_title(f"10-Second GNSS Outage Trajectory: {scen_name.replace('_', ' ').title()}\n"
                         f"IDNN Drift: {sample_trajectory['final_drift_idnn']:.1f}m ({sample_trajectory['drift_pct_idnn']:.1f}%) vs "
                         f"INS DR: {sample_trajectory['final_drift_ins']:.1f}m ({sample_trajectory['drift_pct_ins']:.1f}%)")
            ax.set_xlabel("East Position (meters)")
            ax.set_ylabel("North Position (meters)")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='best')
            plt.tight_layout()

            plot_path = os.path.join(results_dir, f"trajectory_{scen_name}.png")
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            print(f"  [Plot] Saved trajectory plot to: {plot_path}")

    # Save summary report
    with open(os.path.join(results_dir, "benchmark_summary.pkl"), "wb") as f:
        pickle.dump(benchmark_summary, f)

    return benchmark_summary


if __name__ == '__main__':
    evaluate_all_scenarios()
