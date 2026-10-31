"""
defense/flad.py
FLAD: Byzantine-Robust Federated Learning based on Gradient Feature Anomaly Detection
排除隐私保护部分，仅实现核心防御逻辑。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import DBSCAN
from defense.utils import flatten_params, unflatten_params


class FeatureExtractionModel(nn.Module):
    """全连接特征提取网络，输入梯度向量，输出一维特征值"""
    def __init__(self, input_dim, hidden_dims=[256, 64, 1]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_fem(model, server_loader, device, epochs=5, lr=1e-3):
    """
    训练特征提取模型 FEM（对应论文 Algorithm 1）
    输入: 全局模型、服务器干净数据加载器
    输出: 训练好的 FEM，以及标准特征 mu 和偏差 eps
    """
    # 收集服务器数据集上的梯度作为训练数据
    criterion = nn.CrossEntropyLoss()
    model.eval()
    grad_list = []
    for data, target in server_loader:
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # 提取梯度并展平
        grad_vec = []
        for param in model.parameters():
            if param.grad is not None:
                grad_vec.append(param.grad.view(-1))
        if grad_vec:
            grad_list.append(torch.cat(grad_vec).detach())
    if not grad_list:
        raise RuntimeError("No gradients collected from server dataset!")
    grad_tensor = torch.stack(grad_list)  # (num_samples, d)

    input_dim = grad_tensor.shape[1]
    fem = FeatureExtractionModel(input_dim).to(device)
    optimizer = optim.Adam(fem.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    target_val = torch.ones(grad_tensor.shape[0], 1).to(device)
    fem.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = fem(grad_tensor)
        loss = mse_loss(pred, target_val)
        loss.backward()
        optimizer.step()

    # 计算标准特征 mu 和偏差 eps
    fem.eval()
    with torch.no_grad():
        features = fem(grad_tensor).cpu().numpy().flatten()
    mu = float(np.mean(features))
    eps = max(float(np.max(features) - np.min(features)), 1e-6)  # 确保 eps > 0

    return fem, mu, eps


def extract_gradient_features(fem, client_updates, global_model, device):
    """
    提取客户端梯度特征（对应论文 Algorithm 2）
    client_updates: 字典 {cid: delta_dict}
    返回: 特征矩阵 (n_clients, 1)
    """
    features = []
    cid_order = []
    for cid, delta in client_updates.items():
        flat_grad = flatten_params(delta).to(device)
        with torch.no_grad():
            feat = fem(flat_grad.unsqueeze(0)).cpu().item()
        features.append(feat)
        cid_order.append(cid)
    return np.array(features).reshape(-1, 1), cid_order


def detect_malicious(features, mu, eps, alpha=0.5):
    """
    恶意客户端检测（对应论文 Algorithm 3）
    features: (n, 1) 特征向量
    mu: 标准特征
    eps: 扫描半径（DBSCAN 的 eps）
    返回: 恶意客户端在 features 中的索引列表
    """
    # DBSCAN 聚类
    clustering = DBSCAN(eps=eps, min_samples=2).fit(features)
    labels = clustering.labels_

    # 计算每个簇的相似度 S
    best_label = None
    best_score = -float('inf')
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # 噪声点不参与评分

    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        cluster_feats = features[idx]
        # 簇内平均特征作为该簇代表
        cluster_rep = np.mean(cluster_feats)
        # 余弦相似度（一维特征退化为符号一致性）
        cos_sim = cluster_rep * mu / (abs(cluster_rep) * abs(mu) + 1e-8)
        # 相对长度差
        len_diff = abs(cluster_rep - mu) / (abs(mu) + 1e-8)
        # 综合评分
        score = alpha * (cos_sim - len_diff) + (1 - alpha) * np.linalg.norm(cluster_feats)
        if score > best_score:
            best_score = score
            best_label = lbl

    # 诚实标签为 best_label，其余为恶意（包括噪声点）
    malicious_idx = [i for i, lbl in enumerate(labels) if lbl != best_label]
    return malicious_idx


def flad_aggregate(global_model, client_updates, server_loader, device, config):
    """
    FLAD 主聚合函数，供 main.py 调用
    """
    # 1. 训练 FEM
    fem, mu, eps = train_fem(
        global_model, server_loader, device,
        epochs=config.get('flad_fem_epochs', 5),
        lr=config.get('flad_fem_lr', 1e-3)
    )

    # 2. 提取特征
    features, cid_order = extract_gradient_features(fem, client_updates, global_model, device)

    # 3. 检测恶意客户端
    alpha = config.get('flad_alpha', 0.5)
    malicious_idx = detect_malicious(features, mu, eps, alpha)
    malicious_cids = {cid_order[i] for i in malicious_idx}

    # 4. 剔除恶意客户端，对诚实客户端执行 FedAvg
    honest_updates = {cid: delta for cid, delta in client_updates.items() if cid not in malicious_cids}
    if not honest_updates:
        return None  # 无诚实客户端，返回 None 跳过本轮更新

    agg_delta = {}
    keys = list(honest_updates.values())[0].keys()
    for k in keys:
        tensors = [honest_updates[cid][k] for cid in honest_updates]
        if isinstance(tensors[0], torch.Tensor) and torch.is_floating_point(tensors[0]):
            agg_delta[k] = torch.stack(tensors).mean(dim=0).to(device)
        else:
            agg_delta[k] = honest_updates[list(honest_updates.keys())[0]][k].clone()
    return agg_delta