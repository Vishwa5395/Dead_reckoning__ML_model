import sys
from pathlib import Path
import pickle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

from v7_sept_model.src.moe_five_model import SupremeMoENet

WINDOW = 10
OUTAGE = 10

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA_DIR / "scalers_v7.pkl", "rb") as f:
        scalers = pickle.load(f)
    s_X, s_yd, s_yo = scalers["X"], scalers["y_disp"], scalers["y_ori"]

    with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    moe = SupremeMoENet().to(device)
    ckpt = torch.load(CKPT_DIR / "best_supreme_moe.pth", map_location=device, weights_only=False)
    moe.load_state_dict(ckpt["model_state_dict"])
    moe.eval()

    sharp_journeys = test_scenarios["sharp_turns"]
    print(f"Total sharp turn journeys: {len(sharp_journeys)}")

    drifts = []
    heading_errors = []
    speed_errors = []
    long_errors = []
    lat_errors = []
    outage_details = []

    outage_id = 0
    for j_idx, j in enumerate(sharp_journeys):
        x_gps = j["x_gps"]; w_gps = j["w_gps"]; headings = j["headings"]
        a_fwd = j["a_fwd"]; w_yaw = j["w_yaw"]; a_lat = j["a_lat"]; w_accel = j["w_yaw_accel"]

        for s in range(WINDOW + 1, len(x_gps) - OUTAGE, 10):
            outage_id += 1
            history_x = list(x_gps[s - WINDOW: s])
            pos_gt, pos_pred = [(0.0, 0.0)], [(0.0, 0.0)]
            psi_gt = np.radians(headings[s])
            psi_pred = psi_gt

            step_w_gt = []
            step_w_pred = []
            step_x_gt = []
            step_x_pred = []

            for k in range(OUTAGE):
                cur = s + k
                ch_a_fwd = np.clip(a_fwd[cur-9:cur+1], -8.0, 8.0)
                ch_w_yaw = np.clip(w_yaw[cur-9:cur+1], -1.0, 1.0)
                ch_a_lat = np.clip(a_lat[cur-9:cur+1], -8.0, 8.0)
                ch_v_prev = np.clip(np.array(history_x[-10:]), 0.0, 45.0)
                ch_w_accel = np.clip(w_accel[cur-9:cur+1], -2.0, 2.0)
                ch_centripetal = np.clip(ch_a_lat - ch_v_prev * ch_w_yaw, -8.0, 8.0)

                win_6 = np.stack([ch_a_fwd, ch_w_yaw, ch_a_lat, ch_v_prev, ch_w_accel, ch_centripetal], axis=-1)
                win_s6 = s_X.transform(win_6.reshape(1, -1)).reshape(1, 10, 6).astype(np.float32)

                with torch.no_grad():
                    t_x = torch.tensor(win_s6, dtype=torch.float32, device=device)
                    d_p, o_p, z_p, weights = moe(t_x)

                xr = float(s_yd.inverse_transform([[d_p.item()]])[0, 0])
                wr = float(s_yo.inverse_transform([[o_p.item()]])[0, 0])

                history_x.append(xr)
                psi_gt += w_gps[cur]
                psi_pred += wr
                pos_gt.append((pos_gt[-1][0] + x_gps[cur] * np.cos(psi_gt), pos_gt[-1][1] + x_gps[cur] * np.sin(psi_gt)))
                pos_pred.append((pos_pred[-1][0] + xr * np.cos(psi_pred), pos_pred[-1][1] + xr * np.sin(psi_pred)))

                step_w_gt.append(w_gps[cur])
                step_w_pred.append(wr)
                step_x_gt.append(x_gps[cur])
                step_x_pred.append(xr)

            d_final = float(np.hypot(pos_pred[-1][0] - pos_gt[-1][0], pos_pred[-1][1] - pos_gt[-1][1]))
            h_err_deg = np.degrees(np.abs(psi_pred - psi_gt))
            dist_gt = sum(step_x_gt)
            dist_pred = sum(step_x_pred)
            v_err = dist_pred - dist_gt
            total_turn_gt = np.degrees(sum(step_w_gt))
            total_turn_pred = np.degrees(sum(step_w_pred))

            drifts.append(d_final)
            heading_errors.append(h_err_deg)
            speed_errors.append(v_err)

            outage_details.append({
                "id": outage_id,
                "journey": j_idx,
                "start": s,
                "drift": d_final,
                "h_err_deg": h_err_deg,
                "turn_gt_deg": total_turn_gt,
                "turn_pred_deg": total_turn_pred,
                "dist_gt": dist_gt,
                "dist_pred": dist_pred,
                "dist_err": v_err,
            })

    print(f"\nSharp Turns Summary over {len(drifts)} outages:")
    print(f"  Mean Drift: {np.mean(drifts):.2f} m (Min: {np.min(drifts):.2f}, Max: {np.max(drifts):.2f}, Median: {np.median(drifts):.2f})")
    print(f"  Mean Heading Error: {np.mean(heading_errors):.2f} deg")
    print(f"  Mean Distance Error: {np.mean(speed_errors):.2f} m")

    # Sort outages by drift
    sorted_outages = sorted(outage_details, key=lambda x: x["drift"], reverse=True)
    print("\nTop 10 Worst Outages in Sharp Turns:")
    print(f"{'ID':4s} | {'Drift':8s} | {'Heading Err':12s} | {'Turn GT':10s} | {'Turn Pred':10s} | {'Dist GT':10s} | {'Dist Pred':10s}")
    for o in sorted_outages[:10]:
        print(f"{o['id']:4d} | {o['drift']:7.2f}m | {o['h_err_deg']:10.2f}° | {o['turn_gt_deg']:9.2f}° | {o['turn_pred_deg']:9.2f}° | {o['dist_gt']:8.2f}m | {o['dist_pred']:8.2f}m")

    print("\nTop 10 Best Outages in Sharp Turns:")
    for o in sorted_outages[-10:]:
        print(f"{o['id']:4d} | {o['drift']:7.2f}m | {o['h_err_deg']:10.2f}° | {o['turn_gt_deg']:9.2f}° | {o['turn_pred_deg']:9.2f}° | {o['dist_gt']:8.2f}m | {o['dist_pred']:8.2f}m")

if __name__ == "__main__":
    main()
