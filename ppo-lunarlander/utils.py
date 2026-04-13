import os 
import random
import numpy as np
import torch 
from pathlib import Path
from typing import Optional, Dict, Any

def set_seed(seed: int = 42) -> None:  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[utils] Seed set to {seed}")

def get_device() -> torch.device:
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[utils] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[utils] Apple Silicon GPU (MPS).")
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    checkpoint = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode":              episode,
        "total_steps":          total_steps,
        "best_reward":    best_reward,
    }
    torch.save(checkpoint, path)
    print(f"[utils] Saved → {path} (ep {episode}, reward {best_reward:.1f})")

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    
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
    set_seed(42)
    first_rand = random.random()
    first_np = np.random.rand()
    set_seed(42)
    assert random.random() == first_rand and np.random.rand() == first_np
    print("Test 1 PASSED — set_seed() is deterministic")

    device = get_device()
    assert device.type in {"cpu", "cuda", "mps"}
    print("Test 2 PASSED — device: cpu")

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 2)
        def forward(self, x):
            return self.layer(x)

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ckpt_path = "checkpoints/test_ckpt.pt"
    save_checkpoint(model, optimizer, episode=10, total_steps=100, best_reward=150.0, path=ckpt_path)
    loaded = load_checkpoint(ckpt_path, model, optimizer=optimizer, device=torch.device("cpu"))
    assert loaded["episode"] == 10 and loaded["best_reward"] == 150.0
    print("Test 3 PASSED — save/load checkpoint works correctly")

    smooth = moving_average([1, 2, 3, 4], window=2)
    assert np.allclose(smooth, np.array([1.5, 1.5, 2.5, 3.5], dtype=np.float32))
    print("Test 4 PASSED — moving_average() smooths correctly")

    print("")
    cfg = {"ppo": {"lr": 0.0003, "clip": 0.2}, "env": {"seed": 42}}
    log_config(cfg)
    print("")
    print("Test 5 PASSED — log_config() prints without error")
    print("")
    print("All tests passed! utils.py is ready.")

