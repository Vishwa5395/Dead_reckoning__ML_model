"""
hyperparameter_search_v6.py
---------------------------
Validation-only hyperparameter search for PINO-DR v6.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from v6_smartphone_idr.src.models_v6 import (
    PINODeadReckoningNetV6
)

from v6_smartphone_idr.src.train_v6 import (
    V6AugmentedDataset,
    compute_v6_loss,
)

from v6_smartphone_idr.src.augmentation_v6 import (
    compute_maneuver_weights,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def run_val_hyperparameter_sweep(
    sample_budget: int = 4,
    trial_epochs: int = 20,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print(
        "PINO-DR v6: "
        "Validation-Only Hyperparameter Search"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Load preprocessed data
    # --------------------------------------------------------

    npz_path = (
        DATA_DIR
        / "dataset_splits_v6.npz"
    )

    scaler_path = (
        DATA_DIR
        / "scalers_v6.pkl"
    )

    d = np.load(npz_path)

    with open(
        scaler_path,
        "rb",
    ) as f:
        scalers = pickle.load(f)

    X_tr = d["X_tr"]
    y_dv_tr = d["y_dv_tr"]
    y_w_tr = d["y_w_tr"]
    y_z_tr = d["y_z_tr"]
    y_ba_tr = d["y_ba_tr"]
    y_bw_tr = d["y_bw_tr"]

    X_va = d["X_va"]
    y_dv_va = d["y_dv_va"]
    y_w_va = d["y_w_va"]
    y_z_va = d["y_z_va"]
    y_ba_va = d["y_ba_va"]
    y_bw_va = d["y_bw_va"]

    # --------------------------------------------------------
    # Small subset for hyperparameter search
    # --------------------------------------------------------

    X_tr = X_tr[::40]
    y_dv_tr = y_dv_tr[::40]
    y_w_tr = y_w_tr[::40]
    y_z_tr = y_z_tr[::40]
    y_ba_tr = y_ba_tr[::40]
    y_bw_tr = y_bw_tr[::40]

    X_va = X_va[::10]
    y_dv_va = y_dv_va[::10]
    y_w_va = y_w_va[::10]
    y_z_va = y_z_va[::10]
    y_ba_va = y_ba_va[::10]
    y_bw_va = y_bw_va[::10]

    # --------------------------------------------------------
    # Compute maneuver weights in PHYSICAL units
    # --------------------------------------------------------

    y_w_tr_physical = (
        scalers["y_w"]
        .inverse_transform(y_w_tr)
    )

    y_dv_tr_physical = (
        scalers["y_dv"]
        .inverse_transform(y_dv_tr)
    )

    weights_tr = compute_maneuver_weights(
        y_w_tr_physical,
        y_dv_tr_physical,
        y_z_tr,
    )

    weights_va = np.ones(
        len(X_va),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    ds_tr = V6AugmentedDataset(
        X_tr,
        y_dv_tr,
        y_w_tr,
        y_z_tr,
        y_ba_tr,
        y_bw_tr,
        scaler_X=scalers["X"],
        sample_weights=weights_tr,
        is_train=True,
    )

    ds_va = V6AugmentedDataset(
        X_va,
        y_dv_va,
        y_w_va,
        y_z_va,
        y_ba_va,
        y_bw_va,
        scaler_X=scalers["X"],
        sample_weights=weights_va,
        is_train=False,
    )

    loader_tr = DataLoader(
        ds_tr,
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )

    loader_va = DataLoader(
        ds_va,
        batch_size=128,
        shuffle=False,
    )

    tau_turn = float(
        scalers["turn_stats"]["tau_turn"]
    )

    # --------------------------------------------------------
    # Search space
    # --------------------------------------------------------

    search_space = [
        {
            "lr": 1e-3,
            "dropout": 0.10,
            "weight_decay": 1e-4,
            "w_head": 1.5,
            "tag": "Candidate_A",
        },
        {
            "lr": 8e-4,
            "dropout": 0.15,
            "weight_decay": 1e-4,
            "w_head": 1.8,
            "tag": "Candidate_B",
        },
        {
            "lr": 5e-4,
            "dropout": 0.20,
            "weight_decay": 5e-5,
            "w_head": 2.0,
            "tag": "Candidate_C",
        },
        {
            "lr": 1.2e-3,
            "dropout": 0.15,
            "weight_decay": 2e-4,
            "w_head": 1.2,
            "tag": "Candidate_D",
        },
    ]

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    results = []

    for cfg in search_space[
        :sample_budget
    ]:

        print(
            f"\n--- {cfg['tag']} ---"
        )

        model = PINODeadReckoningNetV6(
            dropout=cfg["dropout"]
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg[
                "weight_decay"
            ],
        )

        loss_weights = {
            "velocity": 1.0,
            "heading": cfg["w_head"],
            "zupt": 0.25,
            "bias": 0.05,
            "uncertainty": 0.02,
        }

        for epoch in range(
            1,
            trial_epochs + 1,
        ):

            model.train()

            for batch in loader_tr:

                (
                    bx,
                    b_ydv,
                    b_yw,
                    b_yz,
                    b_yba,
                    b_ybw,
                    b_wgt,
                ) = [
                    x.to(device)
                    for x in batch
                ]

                optimizer.zero_grad(
                    set_to_none=True
                )

                preds = model(bx)

                loss, _ = compute_v6_loss(
                    preds,
                    (
                        b_ydv,
                        b_yw,
                        b_yz,
                        b_yba,
                        b_ybw,
                    ),
                    b_wgt,
                    tau_turn,
                    loss_weights,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_losses = []

        with torch.no_grad():

            for batch in loader_va:

                (
                    bx,
                    b_ydv,
                    b_yw,
                    b_yz,
                    b_yba,
                    b_ybw,
                    b_wgt,
                ) = [
                    x.to(device)
                    for x in batch
                ]

                preds = model(bx)

                loss, _ = compute_v6_loss(
                    preds,
                    (
                        b_ydv,
                        b_yw,
                        b_yz,
                        b_yba,
                        b_ybw,
                    ),
                    b_wgt,
                    tau_turn,
                    loss_weights,
                )

                val_losses.append(
                    float(loss.item())
                )

        mean_val_loss = float(
            np.mean(val_losses)
        )

        print(
            f"Validation Loss: "
            f"{mean_val_loss:.6f}"
        )

        results.append(
            {
                **cfg,
                "val_loss":
                    mean_val_loss,
            }
        )

    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["val_loss"]
    )

    best = results[0]

    print("\n" + "=" * 80)
    print(
        "Best validation configuration:"
    )
    print(
        json.dumps(
            best,
            indent=2,
        )
    )
    print("=" * 80)

    output_path = (
        RESULTS_DIR
        / "hyperparameter_search_summary_v6.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "best_config": best,
                "all_trials": results,
            },
            f,
            indent=2,
        )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":

    run_val_hyperparameter_sweep()