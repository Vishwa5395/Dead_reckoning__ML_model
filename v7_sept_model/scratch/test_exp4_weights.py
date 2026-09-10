import sys
from pathlib import Path
import pickle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from v7_sept_model.src.moe_five_model import SupremeMoENet
from v7_sept_model.src.evaluate_five_specialists import SCENARIOS, simulate_moe_outage

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

def evaluate_moe(model, test_scenarios, scalers, device):
    drifts = {}
    for scen in SCENARIOS:
        scen_drifts = []
        for j in test_scenarios[scen]:
            x_gps = j["x_gps"]
            for s in range(11, len(x_gps) - 10, 10):
                d_moe, _, _ = simulate_moe_outage(j, s, model, scalers, device)
                scen_drifts.append(d_moe)
        drifts[scen] = float(np.mean(scen_drifts))
    
    overall = float(np.mean([d for s in SCENARIOS for d in [drifts[s]] * (7 if s=='motorway' else 12 if s=='hard_brake' else 4 if s=='quick_accel' else 3 if s=='roundabout' else 39)]))
    drifts["overall"] = overall
    return drifts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA_DIR / "scalers_v7.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(DATA_DIR / "test_scenarios_v7.pkl", "rb") as f:
        test_scenarios = pickle.load(f)

    # 1. Base current SupremeMoE
    moe = SupremeMoENet().to(device)
    ckpt = torch.load(CKPT_DIR / "best_supreme_moe.pth", map_location=device, weights_only=False)
    moe.load_state_dict(ckpt["model_state_dict"])
    moe.eval()

    res_base = evaluate_moe(moe, test_scenarios, scalers, device)
    print("\n--- Current SupremeMoE (exp4 = v4-D base) ---")
    for k, v in res_base.items():
        print(f"  {k:15s}: {v:6.2f}m")

    # 2. Test with best_supreme_sharp_turns.pth loaded into exp4
    ckpt_sharp = torch.load(CKPT_DIR / "best_supreme_sharp_turns.pth", map_location=device, weights_only=False)
    moe.exp4.load_state_dict(ckpt_sharp["model_state_dict"])
    moe.eval()

    res_sharp = evaluate_moe(moe, test_scenarios, scalers, device)
    print("\n--- SupremeMoE with exp4 = best_supreme_sharp_turns.pth ---")
    for k, v in res_sharp.items():
        print(f"  {k:15s}: {v:6.2f}m")

if __name__ == "__main__":
    main()
