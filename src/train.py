"""
train.py
--------
Step-by-step training script for the Displacement and Orientation Rate IDNN models.
Trained on 70% Train split, validated on 10% Val split, tested on 20% Test split.

Table 5 specifications from Onyekpe et al. (2021):
- Batch size: 256 (strictly pinned for both models)
- Optimizer: Adamax
- Loss function: Mean Absolute Error (MAE / L1Loss)
- Displacement IDNN: lr = 0.004, epochs = 40
- Orientation IDNN: lr = 0.001, epochs = 60
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models import DisplacementIDNN, OrientationIDNN, export_to_onnx

CHECKPOINT_DIR = r"c:\Users\tiwar\OneDrive\Desktop\PROJECTS\SIH 26\checkpoints"
CACHE_DIR = r"c:\Users\tiwar\OneDrive\Desktop\PROJECTS\SIH 26\data\preprocessed"


def train_model(
    model_type="displacement",
    epochs=None,
    batch_size=256,
    learning_rate=None,
    data_dir=CACHE_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    device=None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz_path = os.path.join(data_dir, "dataset_splits.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset splits not found at {npz_path}. Run preprocessing first!")

    data = np.load(npz_path)

    if model_type == "displacement":
        X_tr = torch.tensor(data["d_X_tr"], dtype=torch.float32)
        Y_tr = torch.tensor(data["d_Y_tr"], dtype=torch.float32)
        X_va = torch.tensor(data["d_X_va"], dtype=torch.float32)
        Y_va = torch.tensor(data["d_Y_va"], dtype=torch.float32)
        X_te = torch.tensor(data["d_X_te"], dtype=torch.float32)
        Y_te = torch.tensor(data["d_Y_te"], dtype=torch.float32)

        lr = learning_rate if learning_rate is not None else 0.004
        num_epochs = epochs if epochs is not None else 40
        model = DisplacementIDNN(input_dim=20, hidden_dim=32, dropout=0.10).to(device)
        model_name = "displacement_idnn"
    elif model_type == "orientation":
        X_tr = torch.tensor(data["o_X_tr"], dtype=torch.float32)
        Y_tr = torch.tensor(data["o_Y_tr"], dtype=torch.float32)
        X_va = torch.tensor(data["o_X_va"], dtype=torch.float32)
        Y_va = torch.tensor(data["o_Y_va"], dtype=torch.float32)
        X_te = torch.tensor(data["o_X_te"], dtype=torch.float32)
        Y_te = torch.tensor(data["o_Y_te"], dtype=torch.float32)

        lr = learning_rate if learning_rate is not None else 0.001
        num_epochs = epochs if epochs is not None else 60
        model = OrientationIDNN(input_dim=10, hidden_dim=32, dropout=0.10).to(device)
        model_name = "orientation_idnn"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(X_va, Y_va), batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(TensorDataset(X_te, Y_te), batch_size=batch_size, shuffle=False, drop_last=False)

    criterion = nn.L1Loss()  # MAE Loss per paper
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)

    n_train, n_val, n_test = len(X_tr), len(X_va), len(X_te)
    total_samples = n_train + n_val + n_test

    print(f"\n{'='*75}")
    print(f"TRAINING {model_name.upper()} (70/10/20 SPLIT ON FULL DATASET)")
    print(f"Samples: Total={total_samples:,} | Train(70%)={n_train:,} | Val(10%)={n_val:,} | Test(20%)={n_test:,}")
    print(f"Batch Size: {batch_size} ({len(train_loader)} batches/epoch) | Optimizer: Adamax (lr={lr}) | Loss: MAE")
    print(f"Target Epochs: {num_epochs} | Device: {device}")
    print(f"{'='*75}")

    best_val_loss = float("inf")
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}.pth")

    t_start = time.time()
    for epoch in range(1, num_epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        train_loss /= n_train

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_x)
        val_loss /= n_val
        ep_time = time.time() - ep_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "model_type": model_type,
                },
                ckpt_path,
            )
            saved_mark = " [*Best]"
        else:
            saved_mark = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(
                f"Epoch [{epoch:2d}/{num_epochs:2d}] - Train MAE: {train_loss:.5f} | "
                f"Val MAE: {val_loss:.5f} ({ep_time*1000:.1f} ms){saved_mark}"
            )

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time:.2f}s! Best Val MAE: {best_val_loss:.5f}")

    # Evaluate on held-out 20% Test Split
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            test_loss += loss.item() * len(batch_x)
    test_loss /= n_test

    print(f">>> HELD-OUT 20% TEST SET MAE: {test_loss:.5f} <<<")
    print(f"Model checkpoint saved: {ckpt_path}")

    # Export to ONNX
    dummy_input = torch.randn(1, 20 if model_type == "displacement" else 10, device=device)
    onnx_path = os.path.join(checkpoint_dir, f"{model_name}.onnx")
    export_to_onnx(model, dummy_input, onnx_path)

    return model, best_val_loss, test_loss


def train_all():
    print(">>> Step 1/2: Training Displacement IDNN Model on 70% Train Split...")
    train_model(model_type="displacement", epochs=40, batch_size=256, learning_rate=0.004)

    print("\n>>> Step 2/2: Training Orientation Rate IDNN Model on 70% Train Split...")
    train_model(model_type="orientation", epochs=60, batch_size=256, learning_rate=0.001)


if __name__ == "__main__":
    train_all()
