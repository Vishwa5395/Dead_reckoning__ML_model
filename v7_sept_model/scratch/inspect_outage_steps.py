import sys
from pathlib import Path
import pickle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from v7_sept_model.src.moe_five_model import SupremeMoENet

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA_DIR / "scalers_v7.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    s_X, s_yd, s_yo = scalers["X"], scalers["y_disp"], scalers["y_ori"]

    moe = SupremeMoENet().to(device)
    ckpt = torch.load(CKPT_DIR / "best_supreme_moe.pth", map_location=device, weights_only=False)
    moe.load_state_dict(ckpt["model_state_dict"])
    moe.eval()

    sharp_journeys = test_scenarios["sharp_turns"]
    
    # Check Outage 2 (Journey 0, start 21)
    j = sharp_journeys[0]
    s = 21
    WINDOW = 10
    OUTAGE = 10

    x_gps = j["x_gps"]; w_gps = j["w_gps"]; headings = j["headings"]
    a_fwd = j["a_fwd"]; w_yaw = j["w_yaw"]; a_lat = j["a_lat"]; w_accel = j["w_yaw_accel"]

    history_x = list(x_gps[s - WINDOW: s])

    print(f"\n--- STEP-BY-STEP INSPECTION OF OUTAGE 2 (Journey 0, start 21) ---")
    print(f"{'Step':4s} | {'w_gps (deg)':11s} | {'wyaw (deg/s)':12s} | {'alat (m/s2)':11s} | {'Router Weights [mot, rb, qa, hb, st]':38s} | {'wr0..wr4 (deg)'}")

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
            weights = moe.gating(t_x)
            x4 = t_x[:, :, :4]
            d0_raw, o0_raw, _ = moe.exp0(x4)
            d1_raw, o1_raw, _ = moe.exp1(t_x)
            d2_raw, o2_raw, _ = moe.exp2(t_x)
            d3_raw, o3_raw, _ = moe.exp3(x4)
            d4_raw, o4_raw, _ = moe.exp4(t_x)

            # Raw orientations
            wr0 = (o0_raw.item() * 2.0263271 - 1.0224137) * 180 / np.pi
            wr1 = (o1_raw.item() * 2.0263271 - 1.0224137) * 180 / np.pi
            wr2 = (o2_raw.item() * 2.0263271 - 1.0224137) * 180 / np.pi
            wr3 = (o3_raw.item() * 2.0263271 - 1.0224137) * 180 / np.pi
            wr4 = (o4_raw.item() * 2.0263271 - 1.0224137) * 180 / np.pi

            d_p, o_p, _, _ = moe(t_x)
            xr = float(s_yd.inverse_transform([[d_p.item()]])[0, 0])
            history_x.append(xr)

            w_blend = (o_p.item() * 2.0263271 - 1.0224137) * 180 / np.pi

        w_deg = w_gps[cur] * 180 / np.pi
        wyaw_deg = w_yaw[cur] * 180 / np.pi
        alat_v = a_lat[cur]
        w_str = "[" + ", ".join([f"{w:.2f}" for w in weights.cpu().numpy()[0]]) + "]"
        wr_str = f"[{wr0:+5.1f}, {wr1:+5.1f}, {wr2:+5.1f}, {wr3:+5.1f}, {wr4:+5.1f}] -> blend={w_blend:+5.1f}"

        print(f"k={k:02d} | {w_deg:+10.2f}° | {wyaw_deg:+11.2f}°/s | {alat_v:+10.2f} | {w_str:38s} | {wr_str}")

if __name__ == "__main__":
    main()
