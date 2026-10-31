# defense/diffusion.py
import torch
import torch.nn as nn

class SimpleDiffusionModel(nn.Module):
    def __init__(self, latent_dim, hidden_dim=512, time_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_t, t):
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        x = torch.cat([z_t, t_emb], dim=-1)
        return self.net(x)

class DiffusionTrainer:
    def __init__(self, latent_dim, timesteps=200, lr=1e-4, device="cuda"):
        self.timesteps = timesteps
        self.device = device
        self.model = SimpleDiffusionModel(latent_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.latent_dim = latent_dim

        beta_start, beta_end = 1e-4, 0.02
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def _sample_timesteps(self, n):
        return torch.randint(0, self.timesteps, (n,), device=self.device)

    def _add_noise(self, x0, t):
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])[:, None]
        noise = torch.randn_like(x0)
        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return xt, noise

    def train_step(self, latents):
        latents = latents.detach().to(self.device)
        t = self._sample_timesteps(latents.shape[0])
        xt, noise = self._add_noise(latents, t)
        pred_noise = self.model(xt, t)
        loss = nn.functional.mse_loss(pred_noise, noise)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, latents, epochs=10, batch_size=32):
        dataset = torch.utils.data.TensorDataset(latents)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for (batch,) in loader:
                loss = self.train_step(batch)
                total_loss += loss
            avg_loss = total_loss / len(loader)
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"  Diffusion Epoch {epoch}, Loss: {avg_loss:.6f}")
        self.model.eval()

    @torch.no_grad()
    def reconstruct_error(self, latents):
        """返回每个样本的异常分数（预测噪声的 L2 范数）"""
        latents = latents.to(self.device)
        B = latents.shape[0]
        t_max = torch.full((B,), self.timesteps - 1, device=self.device, dtype=torch.long)
        noise_pred = self.model(latents, t_max.float())
        error = torch.norm(noise_pred, dim=1)  # [B]
        return error.cpu()  # 返回 CPU tensor

    @torch.no_grad()
    def multi_step_anomaly_score(self, latents, num_steps=10):
        latents = latents.detach().to(self.device)
        N = latents.shape[0]
        device = self.device

        # 存储多次重建的 x0 估计
        recon_x0_list = []

        for _ in range(num_steps):
            # 随机采样时间步 t ∈ [50, T) （避免 t 太小，噪声太少）
            t = torch.randint(50, self.timesteps, (N,), device=device)

            # 前向加噪
            sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None]
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])[:, None]
            noise = torch.randn_like(latents)
            z_t = sqrt_alpha_cumprod * latents + sqrt_one_minus_alpha_cumprod * noise

            # 模型预测噪声
            pred_noise = self.model(z_t, t.float())

            # 单步重建 x0 的估计（DDIM 近似）
            x0_pred = (z_t - sqrt_one_minus_alpha_cumprod * pred_noise) / (sqrt_alpha_cumprod + 1e-8)
            recon_x0_list.append(x0_pred.unsqueeze(0))  # [1, N, D]

        # 合并所有重建结果: [num_steps, N, D]
        all_recons = torch.cat(recon_x0_list, dim=0)

        # 计算每个样本在多次重建中的标准差（L2 norm of std across steps）
        std_across_steps = torch.std(all_recons, dim=0)  # [N, D]
        anomaly_score = torch.norm(std_across_steps, dim=1)  # [N]

        return anomaly_score.cpu()