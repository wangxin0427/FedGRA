import torch
def filter_by_directional_median(flat_updates, angle_threshold_deg=90.0):
    """
    使用方向中位数过滤异常更新。
    改进：用单位向量均值代替逐维中位数，更鲁棒。
    """
    N = flat_updates.size(0)
    if N <= 1:
        return flat_updates

    flat_updates = flat_updates.cpu()

    # 1. 归一化为单位向量
    norms = torch.norm(flat_updates, p=2, dim=1, keepdim=True) + 1e-12
    unit_updates = flat_updates / norms  # [N, D]

    # 2. 计算中心方向：使用单位向量的均值（更鲁棒）
    center_dir = torch.mean(unit_updates, dim=0)  # [D]
    center_norm = torch.norm(center_dir)
    if center_norm < 1e-8:
        # 所有方向完全相反，退化为随机保留一半
        indices = torch.randperm(N)[:N // 2]
        clean_updates = flat_updates[indices]
        print(f"⚠️ Degenerate case: kept {clean_updates.size(0)}/{N} updates")
        return clean_updates
    center_dir = center_dir / center_norm  # 归一化

    # 3. 计算夹角
    cos_theta = torch.sum(unit_updates * center_dir, dim=1)  # [N]
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)  # 避免 acos(-1/1) 数值问题
    angles_rad = torch.acos(cos_theta)

    # 4. 过滤
    threshold_rad = torch.pi * angle_threshold_deg / 180.0
    mask = angles_rad <= threshold_rad
    clean_updates = flat_updates[mask]

    print(f"✅ Directional median filtering: kept {mask.sum().item()}/{N} updates")
    return clean_updates