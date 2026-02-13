import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from poke_env.environment.singles_env import SinglesEnv


def compute_action_mask(battle) -> np.ndarray:
    """Build a binary mask of legal actions (26-dim for gen9)."""
    mask = np.zeros(26, dtype=np.float32)
    for order in battle.valid_orders:
        try:
            action = int(SinglesEnv.order_to_action(order, battle, fake=True))
            if 0 <= action < 26:
                mask[action] = 1.0
        except Exception:
            pass
    if mask.sum() == 0:
        mask[:] = 1.0
    return mask


def setup_logging(log_dir: str) -> SummaryWriter:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return SummaryWriter(run_dir)


def save_checkpoint(model, optimizer, timestep, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timestep": timestep,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("timestep", 0)
