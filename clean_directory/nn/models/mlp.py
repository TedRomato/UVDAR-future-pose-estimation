"""
models/mlp.py — MLP model builder and optimizer factory.
"""

import torch.nn as nn
import torch.optim as optim


# ------------------------------------------------------------------ #
#  Activation registry                                                #
# ------------------------------------------------------------------ #

_ACTIVATIONS = {
    "relu":       nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "tanh":       nn.Tanh,
    "gelu":       nn.GELU,
    "elu":        nn.ELU,
}


# ------------------------------------------------------------------ #
#  Model builder                                                      #
# ------------------------------------------------------------------ #

def build_model(cfg: dict, in_dim: int, out_dim: int = 3) -> nn.Sequential:
    """
    Construct a feed-forward MLP from *cfg*.

    Parameters
    ----------
    cfg : dict
        Must contain ``layers`` (list[int]) and ``activation`` (str).
    in_dim, out_dim : int
        Input / output dimensionality.
    """
    hidden_sizes = cfg["layers"]
    act_name = cfg["activation"].lower()

    if act_name not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation '{cfg['activation']}'. "
            f"Choose from: {list(_ACTIVATIONS)}"
        )

    layers: list[nn.Module] = []
    cur_dim = in_dim

    for h in hidden_sizes:
        layers.append(nn.Linear(cur_dim, h))
        act_fn = _ACTIVATIONS[act_name]
        layers.append(act_fn() if callable(act_fn) else act_fn)
        cur_dim = h

    layers.append(nn.Linear(cur_dim, out_dim))
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ #
#  Optimizer builder                                                  #
# ------------------------------------------------------------------ #

def build_optimizer(cfg: dict, model: nn.Module) -> optim.Optimizer:
    """
    Create an optimizer from *cfg*.

    Parameters
    ----------
    cfg : dict
        Must contain ``optimizer``, ``learning_rate``, and ``weight_decay``.
    """
    opt_name = cfg["optimizer"].lower()
    lr = float(cfg["learning_rate"])
    wd = float(cfg["weight_decay"])

    if opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    raise ValueError(
        f"Unsupported optimizer '{cfg['optimizer']}'. Choose from: adam, adamw, sgd"
    )
