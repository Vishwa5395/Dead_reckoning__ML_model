"""
export_pkl.py
-------------
Bundles both trained IDNN models (Displacement + Orientation) and their
MinMaxScaler normalization parameters into a single `final_model.pkl` file
for easy integration into a mobile/web application.

Contents of final_model.pkl:
  {
    'displacement_model': <DisplacementIDNN state_dict>,
    'orientation_model':  <OrientationIDNN state_dict>,
    'displacement_config': {'input_dim': 20, 'hidden_dim': 32, 'dropout': 0.10},
    'orientation_config':  {'input_dim': 10, 'hidden_dim': 32, 'dropout': 0.10},
    'scalers': <loaded scalers.pkl dict>,
    'metadata': { ... training info ... }
  }
"""

import os
import sys
import pickle
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import DisplacementIDNN, OrientationIDNN

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def main():
    # --- Load scalers ---
    scalers_path = os.path.join(CHECKPOINT_DIR, 'scalers.pkl')
    if not os.path.exists(scalers_path):
        print(f"[ERROR] scalers.pkl not found at {scalers_path}")
        sys.exit(1)
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    print("[OK] Loaded scalers.pkl")

    # --- Load Displacement Model ---
    disp_config = {'input_dim': 20, 'hidden_dim': 32, 'dropout': 0.10}
    disp_model = DisplacementIDNN(**disp_config)
    disp_pth = os.path.join(CHECKPOINT_DIR, 'displacement_idnn.pth')
    disp_ckpt = torch.load(disp_pth, map_location='cpu', weights_only=False)
    disp_model.load_state_dict(disp_ckpt['model_state_dict'])
    disp_model.eval()
    print(f"[OK] Loaded displacement_idnn.pth (epoch {disp_ckpt.get('epoch', '?')})")

    # --- Load Orientation Model ---
    ori_config = {'input_dim': 10, 'hidden_dim': 32, 'dropout': 0.10}
    ori_model = OrientationIDNN(**ori_config)
    ori_pth = os.path.join(CHECKPOINT_DIR, 'orientation_idnn.pth')
    ori_ckpt = torch.load(ori_pth, map_location='cpu', weights_only=False)
    ori_model.load_state_dict(ori_ckpt['model_state_dict'])
    ori_model.eval()
    print(f"[OK] Loaded orientation_idnn.pth (epoch {ori_ckpt.get('epoch', '?')})")

    # --- Bundle everything ---
    bundle = {
        'displacement_model': disp_model.state_dict(),
        'orientation_model': ori_model.state_dict(),
        'displacement_config': disp_config,
        'orientation_config': ori_config,
        'scalers': scalers,
        'metadata': {
            'architecture': 'IDNN (Input Delay Neural Network)',
            'reference': 'Onyekpe et al. 2021 - Learning to Localise Automated Vehicles',
            'dataset': 'IO-VNBD (72 synchronized journeys, 5700+ km)',
            'train_val_test_split': '70/10/20',
            'displacement_epochs': 40,
            'orientation_epochs': 60,
            'batch_size': 256,
            'optimizer': 'Adamax',
            'loss': 'MAE (L1)',
            'window_size': 10,
        }
    }

    out_path = os.path.join(CHECKPOINT_DIR, 'final_model.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n[DONE] Saved final_model.pkl ({size_kb:.1f} KB)")
    print(f"       Path: {os.path.abspath(out_path)}")

    # --- Verify by reloading ---
    with open(out_path, 'rb') as f:
        check = pickle.load(f)
    print(f"\n[VERIFY] Keys in final_model.pkl: {list(check.keys())}")
    print(f"[VERIFY] Displacement model layers: {len(check['displacement_model'])} tensors")
    print(f"[VERIFY] Orientation model layers:  {len(check['orientation_model'])} tensors")
    print(f"[VERIFY] Scalers present: {list(check['scalers'].keys()) if isinstance(check['scalers'], dict) else 'Yes'}")
    print("[VERIFY] All good — ready for app integration!")


if __name__ == '__main__':
    main()
