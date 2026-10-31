import torch
from typing import List, Dict, Any, Tuple
from defense.utils import flatten_params


def aggregate(
        updates: List[Dict[str, torch.Tensor]],
        method: str = "fedavg",
        num_malicious: int = 3,
        flame_eps: float = 0.4,
        trim_ratio: float = 0.3,
        device: str = "cpu"
) -> torch.Tensor:
    """
    Unified robust aggregation interface.
    Supported methods: fedavg, krum, median, trimmedmean, foolsgold
    """
    if len(updates) == 0:
        raise ValueError("No updates provided.")

    flat_updates = [flatten_params(u).to(device) for u in updates]
    stacked = torch.stack(flat_updates)  # (n, d)

    if method == "fedavg":
        return fedavg(stacked)
    elif method == "krum":
        return krum(stacked, num_malicious=num_malicious)
    elif method == "median":
        return coordinate_median(stacked)
    elif method == "trimmedmean":
        return trimmed_mean(stacked, trim_ratio=trim_ratio)
    elif method == "foolsgold":
        return foolsgold(stacked)
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")


# ======================
# Baseline Methods
# ======================

def fedavg(stacked_updates: torch.Tensor) -> torch.Tensor:
    return stacked_updates.mean(dim=0)


def krum(stacked_updates: torch.Tensor, num_malicious: int = 3) -> torch.Tensor:
    n = stacked_updates.size(0)
    if n <= num_malicious + 2:
        return fedavg(stacked_updates)

    diff = stacked_updates.unsqueeze(1) - stacked_updates.unsqueeze(0)
    distances = torch.sum(diff ** 2, dim=2)
    distances.fill_diagonal_(float('inf'))

    k = n - num_malicious - 2
    topk_vals, _ = torch.topk(distances, k, largest=False, dim=1)
    scores = topk_vals.sum(dim=1)
    selected_idx = torch.argmin(scores).item()
    return stacked_updates[selected_idx].clone()


def coordinate_median(stacked_updates: torch.Tensor) -> torch.Tensor:
    return torch.median(stacked_updates, dim=0).values


def trimmed_mean(stacked_updates: torch.Tensor, trim_ratio: float = 0.3) -> torch.Tensor:
    n = stacked_updates.size(0)
    trim_num = int(n * trim_ratio)
    if trim_num * 2 >= n:
        return coordinate_median(stacked_updates)

    sorted_vals, _ = torch.sort(stacked_updates, dim=0)
    trimmed = sorted_vals[trim_num: n - trim_num]
    return trimmed.mean(dim=0)


def foolsgold(stacked_updates: torch.Tensor) -> torch.Tensor:
    n, d = stacked_updates.shape
    if n <= 1:
        return stacked_updates[0]

    norms = torch.norm(stacked_updates, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    unit = stacked_updates / norms

    cos_sim = torch.mm(unit, unit.t())
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    max_sim = cos_sim.max(dim=1).values
    weights = 1.0 / torch.clamp(max_sim, min=1e-5)
    weights = weights / weights.sum()

    return (weights.view(-1, 1) * stacked_updates).sum(dim=0)