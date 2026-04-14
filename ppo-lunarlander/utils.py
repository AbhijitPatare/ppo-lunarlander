
import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any


def set_seed(seed: int = 42) -> None:
    """Seed all random generators for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[utils] Seed set to {seed}")


def get_device() -> torch.device:
    """Return best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[utils] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[utils] Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("[utils] CPU — ~2 hours for LunarLander")
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    episode: int,
    total_steps: int,
    best_reward: float,
    path: str = "checkpoints/best_model.pt",
) -> None:
    """Save model + optimizer state to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode":             episode,
        "total_steps":         total_steps,
        "best_reward":         best_reward,
    }
    torch.save(checkpoint, path)
    print(f"[utils] Saved → {path} (ep {episode}, reward {best_reward:.1f})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load checkpoint. Returns metadata dict {episode, total_steps, best_reward}."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No checkpoint at '{path}'")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"[utils] Loaded ← {path}")
    return {
        "episode":     checkpoint.get("episode",     0),
        "total_steps": checkpoint.get("total_steps", 0),
        "best_reward": checkpoint.get("best_reward", -999.0),
    }


def moving_average(data: list, window: int = 100) -> np.ndarray:
    """Smooth reward curve with a sliding window average."""
    data = np.array(data, dtype=np.float32)
    smoothed = np.convolve(data, np.ones(window) / window, mode="valid")
    pad = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def log_config(config: Dict[str, Any]) -> None:
    """Print config dict to terminal in a readable box."""
    def _flatten(d: dict, prefix: str = "") -> list:
        items = []
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(_flatten(v, prefix=f"{key}."))
            else:
                items.append((key, v))
        return items
    print("┌─────────────────────────────────────┐")
    print("│           Training Config           │")
    print("├─────────────────────────────────────┤")
    for key, value in _flatten(config):
        print(f"│ {key:<20} : {str(value):>12} │")
    print("└─────────────────────────────────────┘")

