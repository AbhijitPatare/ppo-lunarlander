
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

if __name__ == "__main__":
    print("Testing utils.py...")

    # ── Test 1: set_seed ──────────────────────────────────────
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    assert torch.allclose(a, b), "FAIL: same seed must give same tensors"
    print("Test 1 PASSED — set_seed() is deterministic")

    # ── Test 2: get_device ────────────────────────────────────
    device = get_device()
    assert isinstance(device, torch.device), "FAIL: must return torch.device"
    print(f"Test 2 PASSED — device: {device}")

    # ── Test 3: save and load checkpoint ─────────────────────
    from model import ActorCritic
    model = ActorCritic(8, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    save_checkpoint(model, optimizer, episode=10,
                    total_steps=5000, best_reward=150.0,
                    path="checkpoints/test_ckpt.pt")

    model2 = ActorCritic(8, 4)
    meta = load_checkpoint("checkpoints/test_ckpt.pt", model2)

    assert meta["episode"] == 10,      "FAIL: episode not restored"
    assert meta["total_steps"] == 5000, "FAIL: steps not restored"
    assert meta["best_reward"] == 150.0,"FAIL: reward not restored"

    # Verify weights were actually restored correctly
    for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                    model2.named_parameters()):
        assert torch.allclose(p1, p2), f"FAIL: weights differ at {n1}"
    print("Test 3 PASSED — save/load checkpoint works correctly")

    # ── Test 4: moving_average ────────────────────────────────
    rewards = [-200] * 50 + [200] * 50    # step function
    smoothed = moving_average(rewards, window=10)
    assert len(smoothed) == len(rewards),  "FAIL: length must match"
    assert smoothed[0]  < 0,              "FAIL: start should be negative"
    assert smoothed[-1] > 0,              "FAIL: end should be positive"
    print("Test 4 PASSED — moving_average() smooths correctly")

    # ── Test 5: log_config ────────────────────────────────────
    test_config = {"ppo": {"lr": 3e-4, "clip": 0.2}, "env": {"seed": 42}}
    log_config(test_config)
    print("Test 5 PASSED — log_config() prints without error")

    # Clean up test checkpoint
    import shutil
    shutil.rmtree("checkpoints", ignore_errors=True)

    print("All tests passed! utils.py is ready.")