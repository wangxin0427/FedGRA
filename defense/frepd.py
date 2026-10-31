import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Tuple
from defense.utils import flatten_params


# ============================================================
#  Surrogate Vector（随机采样降维，原文 Section 3.4）
# ============================================================
def random_sample_surrogate(flat_vector: np.ndarray,
                            sample_dim: int = 2000,
                            seed: int = 42) -> np.ndarray:
    """
    原文：随机采样模型更新向量的一部分维度作为 surrogate vector，
    避免维度灾难（curse of dimensionality）。
    """
    rng = np.random.RandomState(seed)
    dim = len(flat_vector)
    if dim <= sample_dim:
        return flat_vector
    indices = rng.choice(dim, size=sample_dim, replace=False)
    return flat_vector[indices]


# ============================================================
#  VAE 模型
# ============================================================
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def compute_reconstruction_errors(self, x):
        """计算每个样本的 L2 重建误差"""
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            return torch.norm(x - recon, dim=1)


# ============================================================
#  训练 VAE（用前 5 轮收集的 benign updates）
# ============================================================
def train_vae_on_benign_updates(flat_updates: torch.Tensor,
                                latent_dim: int = 100,
                                epochs: int = 50,
                                lr: float = 1e-3,
                                batch_size: int = 64,
                                sample_dim: int = 2000,
                                device: str = 'cpu') -> Tuple[VAE, Dict[str, Any]]:
    """
    flat_updates: shape (N, full_dim)，全维度梯度
    内部做随机采样降维后训练 VAE。
    """
    # 随机采样降维
    flat_np = flat_updates.numpy()
    sampled_list = []
    for i in range(len(flat_np)):
        sampled = random_sample_surrogate(flat_np[i], sample_dim=sample_dim, seed=i)
        sampled_list.append(sampled)
    sampled_tensor = torch.tensor(np.stack(sampled_list), dtype=torch.float32)

    input_dim = sampled_tensor.shape[1]
    print(f"[FREPD] VAE input dim (after random sampling): {input_dim}")

    vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(sampled_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, mu, logvar = vae(batch)
            recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon_loss + kl_loss) / batch.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"[FREPD] VAE epoch {epoch+1}/{epochs}, avg loss: {total_loss/len(loader):.6f}")

    # 计算训练集上的重建误差分布
    vae.eval()
    all_errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            errors = vae.compute_reconstruction_errors(batch)
            all_errors.append(errors.cpu().numpy())
    train_errors = np.concatenate(all_errors)

    return vae, {'errors': train_errors}


# ============================================================
#  KS 检验拟合分布
# ============================================================
def fit_distribution_to_errors(errors: np.ndarray,
                               distributions: List[str] = None) -> Tuple[str, Tuple]:
    """
    原文：用 Kolmogorov-Smirnov 检验选择最佳概率分布拟合 benign 重建误差。
    """
    if distributions is None:
        distributions = ['norm', 'exponweib', 'weibull_min', 'genextreme', 'beta', 'gamma', 'rayleigh', 'lognorm']

    best_dist = None
    best_pvalue = -1
    best_params = None

    for dist_name in distributions:
        try:
            dist = getattr(stats, dist_name)
            params = dist.fit(errors)
            # KS 检验：比较经验分布和拟合分布
            ks_stat, pvalue = stats.kstest(errors, lambda x: dist.cdf(x, *params))
            if pvalue > best_pvalue:
                best_pvalue = pvalue
                best_dist = dist_name
                best_params = params
        except Exception as e:
            continue

    if best_dist is None:
        best_dist = 'norm'
        best_params = stats.norm.fit(errors)

    print(f"[FREPD] Best distribution: {best_dist}, p-value: {best_pvalue:.4f}")
    return best_dist, best_params


# ============================================================
#  FREPD 聚合（原文 Algorithm 2）
# ============================================================
def frepd_aggregate(updates, vae, dist_name, dist_params, prob_threshold=0.9, sample_dim=2000, device='cpu'):
    """
    只返回保留下来的客户端索引，不做聚合。
    VAE 只当裁判，不当运动员。
    """
    # Step 1: 随机采样降维（只用于 VAE 判断）
    flat_sampled = []
    for i, u in enumerate(updates):
        flat = flatten_params(u).cpu().numpy()
        sampled = random_sample_surrogate(flat, sample_dim=sample_dim, seed=i)
        flat_sampled.append(sampled)
    stacked_np = np.stack(flat_sampled)
    stacked = torch.tensor(stacked_np, dtype=torch.float32).to(device)

    # Step 2: VAE 算重建误差
    vae.eval()
    with torch.no_grad():
        recon, _, _ = vae(stacked)
        errors = torch.norm(stacked - recon, dim=1).cpu().numpy()

    # Step 3: mean 阈值预过滤
    mean_err = np.mean(errors)
    mean_mask = errors <= mean_err
    if mean_mask.sum() == 0:
        mean_mask = np.ones_like(mean_mask, dtype=bool)
    errors_after_mean = errors[mean_mask]

    # Step 4: KS 分布过滤
    dist = getattr(stats, dist_name)
    scores = dist.cdf(errors_after_mean, *dist_params)
    threshold = np.quantile(scores, 1.0 - prob_threshold)
    keep_mask_filtered = scores >= threshold
    if keep_mask_filtered.sum() == 0:
        keep_mask_filtered = np.ones_like(keep_mask_filtered, dtype=bool)

    # 映射回原始索引
    final_mask = np.zeros(len(updates), dtype=bool)
    final_mask[mean_mask] = keep_mask_filtered

    kept_indices = np.where(final_mask)[0].tolist()
    print(f"[FREPD] Kept {len(kept_indices)}/{len(updates)} clients")
    return kept_indices