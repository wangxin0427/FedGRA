import torch
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
from collections import OrderedDict
import warnings

def flatten_updates(updates: list[dict]) -> torch.Tensor:
    """将 list of state_dicts 展平为 (N, d) 的二维张量"""
    flat_list = []
    for update in updates:
        # 按 key 排序确保一致性
        flat = torch.cat([v.flatten() for k, v in sorted(update.items())])
        flat_list.append(flat)
    return torch.stack(flat_list)

def unflatten_to_state_dict(flat_tensor: torch.Tensor, reference_state: dict) -> dict:
    """将一维张量还原为 state_dict"""
    new_state = OrderedDict()
    idx = 0
    for k, v in sorted(reference_state.items()):
        numel = v.numel()
        new_state[k] = flat_tensor[idx:idx+numel].view_as(v.shape)
        idx += numel
    return new_state

def sca_divide_projection(X: torch.Tensor, sub_dim: int, seed: int = 42) -> torch.Tensor:
    """
    SCA-divide: 随机子空间投影降维
    当梯度维度 d 极大时，使用随机投影将维度降至 sub_dim，保留 pairwise 距离结构
    """
    N, d = X.shape
    if d <= sub_dim:
        return X
    
    # 固定随机种子保证可复现性
    rng = np.random.RandomState(seed)
    # 生成稀疏随机投影矩阵 (Achlioptas, 2003) 或 高斯随机矩阵
    # 这里使用高斯矩阵并归一化，满足 Johnson-Lindenstrauss 引理
    R = torch.from_numpy(rng.randn(d, sub_dim)).float().to(X.device) / np.sqrt(sub_dim)
    
    # 降维: (N, d) @ (d, sub_dim) -> (N, sub_dim)
    X_proj = X @ R
    return X_proj

def lens_detection_filter(X: torch.Tensor, sigma: float, c: float = 3.0) -> np.ndarray:
    """
    Lens Detection: 自适应异常点修剪
    计算每个点到其 K 近邻的平均距离，若距离 > mu + c * std，则视为极端异常点并剔除
    这能防止恶意节点通过极大梯度破坏高斯核的相似度分布
    """
    N = X.shape[0]
    K = max(3, int(N * 0.1)) # 取 10% 的近邻
    
    # 计算 pairwise 欧氏距离
    dist_matrix = pairwise_distances(X.cpu().numpy(), metric='euclidean')
    
    # 获取 K 近邻距离 (排除自身)
    knn_dists = np.sort(dist_matrix, axis=1)[:, 1:K+1]
    local_scores = np.mean(knn_dists, axis=1)
    
    # 自适应阈值: tau = mu + c * sigma_lens
    mu = np.mean(local_scores)
    std_lens = np.std(local_scores)
    tau = mu + c * std_lens
    
    # 返回良性节点的 mask
    valid_mask = local_scores <= tau
    return valid_mask

def sca_aggregate(
    updates: list[dict], 
    sigma: float = 1.0, 
    sub_dim: int = 75000, 
    n_clusters: int = 2,
    device: str = 'cpu'
) -> dict:
    """
    Enhanced Spectral Clustering Aggregation (SCA)
    
    参数:
        updates: 客户端模型更新列表 (list of state_dicts)
        sigma: 高斯核宽度 (RBF kernel parameter)
        sub_dim: SCA-divide 目标降维维度
        n_clusters: 聚类数 (通常为 2: 良性/恶意)
        device: 计算设备 (谱聚类建议在 CPU 运行)
        
    返回:
        aggregated_state_dict: 聚合后的全局模型参数
    """
    if len(updates) <= n_clusters:
        warnings.warn("SCA: 客户端数量少于聚类数，退化为 FedAvg")
        # 退化为简单平均
        ref_state = updates[0]
        avg_state = OrderedDict()
        for k in ref_state.keys():
            avg_state[k] = torch.stack([u[k] for u in updates]).mean(dim=0)
        return avg_state

    # 1. 展平梯度
    X = flatten_updates(updates).to(device)
    N, d = X.shape
    
    # 2. SCA-divide 降维 (如果维度超过 sub_dim)
    X_proj = sca_divide_projection(X, sub_dim=sub_dim)
    
    # 3. Lens Detection 预过滤 (剔除极端离群点)
    # 注意：Lens Detection 在降维后的空间计算以提高效率
    valid_mask = lens_detection_filter(X_proj, sigma=sigma, c=3.0)
    
    # 如果过滤后剩下的节点太少，放弃过滤，使用全部节点
    if np.sum(valid_mask) <= n_clusters:
        valid_mask = np.ones(N, dtype=bool)
        
    X_valid = X_proj[valid_mask].cpu().numpy()
    indices_valid = np.where(valid_mask)[0]
    
    # 4. 构建高斯核相似度矩阵 (RBF Kernel)
    # W_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
    dist_sq = pairwise_distances(X_valid, metric='sqeuclidean')
    W = np.exp(-dist_sq / (2 * sigma**2))
    
    # 5. 谱聚类 (使用 sklearn 的高效实现)
    # 等价于计算归一化拉普拉斯矩阵的特征向量并进行 K-means
    try:
        sc = SpectralClustering(
            n_clusters=n_clusters, 
            affinity='precomputed', # 使用我们构建的 W
            assign_labels='kmeans',
            random_state=42,
            n_init=10
        )
        labels = sc.fit_predict(W)
    except Exception as e:
        warnings.warn(f"SCA: 谱聚类失败 ({e})，退化为 FedAvg")
        labels = np.zeros(len(X_valid)) # 全部归为一类

    # 6. 簇选择与聚合
    # 策略：选择包含节点数最多的簇（假设良性客户端占多数）
    unique_labels, counts = np.unique(labels, return_counts=True)
    majority_label = unique_labels[np.argmax(counts)]
    
    # 获取多数簇在原始 updates 中的索引
    majority_indices = indices_valid[labels == majority_label]
    
    # 7. 对良性簇内的更新执行 FedAvg
    ref_state = updates[0]
    agg_state = OrderedDict()
    
    for k in ref_state.keys():
        # 提取良性簇的对应参数
        benign_tensors = torch.stack([updates[i][k] for i in majority_indices])
        agg_state[k] = benign_tensors.mean(dim=0)
        
    return agg_state

# ==========================================
# 测试与 Benchmark 接入示例
# ==========================================
if __name__ == "__main__":
    import time
    
    # 模拟 50 个客户端，模型参数量约 100 万 (如简单 CNN)
    dummy_updates = []
    for i in range(50):
        state = OrderedDict({
            'layer1.weight': torch.randn(64, 3, 3, 3),
            'layer1.bias': torch.randn(64),
            'layer2.weight': torch.randn(128, 64, 3, 3),
            'layer2.bias': torch.randn(128),
        })
        # 注入 10 个恶意节点 (强噪声)
        if i < 10:
            for k in state.keys():
                state[k] = state[k] + torch.randn_like(state[k]) * 10.0
        dummy_updates.append(state)
        
    print("开始 SCA 聚合测试...")
    start_time = time.time()
    
    # 调用 SCA
    agg_model = sca_aggregate(
        dummy_updates, 
        sigma=1.0, 
        sub_dim=50000, # 测试降维
        device='cpu'
    )
    
    elapsed = time.time() - start_time
    print(f"SCA 聚合完成，耗时: {elapsed:.4f} 秒")
    print(f"聚合后 layer1.weight 均值: {agg_model['layer1.weight'].mean().item():.4f}")