# signguard.py
import torch
import numpy as np
from sklearn.cluster import MeanShift


def signguard_aggregate(gradients, L=0.1, R=3.0, coord_ratio=0.1, device='cpu'):
    """
    SignGuard 聚合函数 - 返回扁平化的聚合梯度张量

    Args:
        gradients: List of gradient dictionaries
        L: Lower bound for norm thresholding (relative to median)
        R: Upper bound for norm thresholding (relative to median)
        coord_ratio: Ratio of coordinates to sample for sign statistics
        device: Device to use for computation

    Returns:
        Aggregated gradient as a flattened tensor
    """
    if len(gradients) == 0:
        return None

    # 首先将所有字典格式的梯度转换为扁平化张量
    flat_gradients = []
    param_shapes = None

    for grad_dict in gradients:
        # 获取参数形状信息（只在第一次获取）
        if param_shapes is None:
            param_names = list(grad_dict.keys())
            param_shapes = [(name, grad_dict[name].shape) for name in param_names]

        # 扁平化当前梯度
        flat_grad = torch.cat([grad_dict[name].flatten() for name in param_names])
        flat_gradients.append(flat_grad)

    # 使用扁平化梯度进行 SignGuard 聚合
    aggregated_flat = _signguard_aggregate_flat(flat_gradients, L, R, coord_ratio, device)
    return aggregated_flat


def _signguard_aggregate_flat(gradients, L, R, coord_ratio, device):
    """Aggregate flattened gradients using SignGuard"""
    n = len(gradients)
    if n == 0:
        return None

    # Convert to tensor on specified device
    grad_tensor = torch.stack([
        g.to(device) if isinstance(g, torch.Tensor) else torch.tensor(g, device=device)
        for g in gradients
    ])

    # Step 1: Norm-based filtering
    norms = torch.norm(grad_tensor, p=2, dim=1)
    median_norm = torch.median(norms)

    # Apply loose lower and strict upper bounds
    lower_bound = L * median_norm
    upper_bound = R * median_norm

    norm_mask = (norms >= lower_bound) & (norms <= upper_bound)
    valid_by_norm = torch.where(norm_mask)[0]

    if len(valid_by_norm) == 0:
        # Fallback: use all gradients
        valid_by_norm = torch.arange(n, device=device)

    # Step 2: Sign-based clustering
    d = grad_tensor.shape[1]
    num_coords = max(1, int(coord_ratio * d))
    selected_coords = torch.randperm(d, device=device)[:num_coords]

    # Compute sign statistics for selected coordinates
    signs = torch.sign(grad_tensor[:, selected_coords])
    pos_ratio = torch.mean((signs > 0).float(), dim=1)
    neg_ratio = torch.mean((signs < 0).float(), dim=1)
    zero_ratio = torch.mean((signs == 0).float(), dim=1)

    # Combine features
    features = torch.stack([pos_ratio, neg_ratio, zero_ratio], dim=1).cpu().numpy()

    # Apply MeanShift clustering
    try:
        clustering = MeanShift(bandwidth=None, bin_seeding=True)
        cluster_labels = clustering.fit_predict(features)

        # Find the largest cluster
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts)]
        sign_mask = torch.tensor(cluster_labels == largest_cluster_label, device=device)

    except:
        # Fallback: use all gradients if clustering fails
        sign_mask = torch.ones(n, dtype=torch.bool, device=device)

    # Combine both filters
    final_mask = norm_mask & sign_mask
    trusted_indices = torch.where(final_mask)[0]

    if len(trusted_indices) == 0:
        trusted_indices = torch.arange(n, device=device)

    # Aggregate with norm clipping
    trusted_grads = grad_tensor[trusted_indices]
    clipped_grads = []
    for i, grad in enumerate(trusted_grads):
        grad_norm = torch.norm(grad, p=2)
        if grad_norm > median_norm:
            clipped_grad = grad * (median_norm / grad_norm)
        else:
            clipped_grad = grad
        clipped_grads.append(clipped_grad)

    aggregated = torch.mean(torch.stack(clipped_grads), dim=0)
    return aggregated