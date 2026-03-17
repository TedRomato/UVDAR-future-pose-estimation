# result_manager.py

import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml

import helpers


def save_results(name: str, cfg, artifacts, run_dir: str,
                 results_subdir: str = None, model_type: str = "3d"):
    """
    Create results dir (with val in name), save model, stats, and ONLY the
    learning-curve image (train/val MSE). Also store which dataset (run_dir)
    was used for training, and a model_info.json so the model can be
    reconstructed from saved weights.
    """
    final_val_loss = artifacts["final_val_loss"]
    val_str = f"{final_val_loss:.6f}"
    results_dir = helpers.ensure_fresh_results_dir(f"{name}_val{val_str}", subdir=results_subdir)
    print(f"Creating final results directory: {results_dir}")

    # Augment config with dataset path for traceability (optional but handy)
    cfg_with_meta = dict(cfg)
    cfg_with_meta["_dataset_path"] = os.path.abspath(run_dir)

    # Save config as YAML
    cfg_path_out = os.path.join(results_dir, "config_used.yaml")
    with open(cfg_path_out, "w") as f:
        yaml.safe_dump(cfg_with_meta, f, sort_keys=False)
    print(f"Config saved to: {cfg_path_out}")

    # Also store dataset path in its own text file (very easy to read from tools)
    dataset_path_file = os.path.join(results_dir, "dataset_path.txt")
    with open(dataset_path_file, "w") as f:
        f.write(os.path.abspath(run_dir) + "\n")
    print(f"Dataset path saved to: {dataset_path_file}")

    # Save weights
    torch.save(
        artifacts["model"].state_dict(),
        os.path.join(results_dir, "model_state_dict.pt"),
    )
    print(f"Model weights saved to: {results_dir}/model_state_dict.pt")

    # Save normalization stats as JSON
    norm_stats_json = {k: v.tolist() for k, v in artifacts["norm_stats"].items()}
    with open(os.path.join(results_dir, "normalization.json"), "w") as f:
        json.dump(norm_stats_json, f, indent=2)

    # Save model reconstruction info (type + dimensions)
    # in_dim / out_dim are inferred from the first & last Linear layers.
    state = artifacts["model"].state_dict()
    first_weight = next(v for k, v in state.items() if k.endswith(".weight"))
    last_weight  = list(v for k, v in state.items() if k.endswith(".weight"))[-1]
    model_info = {
        "model_type": model_type,
        "in_dim":     int(first_weight.shape[1]),
        "out_dim":    int(last_weight.shape[0]),
    }
    model_info_path = os.path.join(results_dir, "model_info.json")
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info saved to: {model_info_path}")

    # --- Only: learning curves (train/val MSE in normalized space) ---
    train_losses = artifacts["train_losses"]
    val_losses   = artifacts["val_losses"]
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="train loss (MSE)")
    ax.plot(epochs, val_losses,   label="val loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (normalized MSE)")
    ax.set_title("Learning curves")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    loss_fig_path = os.path.join(results_dir, "learning_curves_mse.png")
    fig.savefig(loss_fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved {loss_fig_path}")

    print(f"✅ All results saved to: {results_dir}")
    return results_dir
