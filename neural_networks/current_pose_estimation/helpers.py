import os
import numpy as np
import pandas as pd
import os
import yaml


def load_config():
    """Load YAML config from config.yaml next to this script (strict: no defaults)."""
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)

    # Strict required keys
    required = [
        "learning_rate",
        "epochs",
        "layers",
        "batch_size",
        "val_split",
        "activation",
        "optimizer",
        "weight_decay",
    ]
    for k in required:
        if k not in raw:
            raise KeyError(f"Missing required config key: '{k}'")

    # Type conversion + validation
    cfg = {
        "learning_rate": float(raw["learning_rate"]),
        "epochs": int(raw["epochs"]),
        "layers": list(raw["layers"]),
        "batch_size": int(raw["batch_size"]),
        "val_split": float(raw["val_split"]),
        "activation": str(raw["activation"]),
        "optimizer": str(raw["optimizer"]),
        "weight_decay": float(raw["weight_decay"]),
    }

    return cfg


def load_xyz(path):
    """Load CSV with time,x,y,z and add integer milliseconds (t_ms) for exact matching."""
    df = pd.read_csv(path)[["time","x","y","z"]].dropna().sort_values("time")
    df["t_ms"] = (df["time"] * 1000).round().astype(np.int64)
    df = df.drop_duplicates(subset="t_ms")
    return df

def interpolate_on_index(target_idx, df):
    """Reindex df (with t_ms,x,y,z) onto target_idx (int ms)."""
    d = (df[["t_ms","x","y","z"]].set_index("t_ms").sort_index())
    out = (d.reindex(target_idx)
             .interpolate(method="index")
             .ffill()
             .bfill())
    out.index.name = "t_ms"
    return out[["x","y","z"]]

def align_three_streams_on_uvdar(est_df, od1_df, od2_df):
    """Build ONE common timeline = unique, sorted UVDAR t_ms."""
    ref_ms = np.unique(est_df["t_ms"].values)
    od1_al = interpolate_on_index(ref_ms, od1_df)
    od2_al = interpolate_on_index(ref_ms, od2_df)
    est_on_ref = (est_df.set_index("t_ms").loc[ref_ms])[["x","y","z"]]
    t_sec = ref_ms.astype(np.float64) / 1000.0
    return (est_on_ref.values.astype(np.float32),
            od1_al.values.astype(np.float32),
            od2_al.values.astype(np.float32),
            t_sec)

def ensure_fresh_results_dir(name: str) -> str:
    """Create ./results/<name> if it does not exist; error if it does."""
    out = os.path.join(".", "results", name)
    if os.path.exists(out):
        raise FileExistsError(f"Results folder already exists: {out}")
    os.makedirs(out, exist_ok=False)
    return out
