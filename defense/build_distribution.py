import torch
from .autoencoder import AutoEncoder
from .diffusion import DiffusionTrainer


def train_autoencoder_on_cpu(ae, data, config):
    """
    在 CPU 上训练 AutoEncoder，避免 GPU OOM
    """
    device = "cpu"
    ae.to(device)
    data = data.to(device)  # 确保数据也在 CPU

    optimizer = torch.optim.Adam(ae.parameters(), lr=config['ae_learning_rate'])
    ae.train()

    batch_size = config.get('ae_batch_size', min(32, len(data)))
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(config['ae_epochs']):
        total_loss = 0.0
        for batch, in dataloader:
            optimizer.zero_grad()
            recon, _ = ae(batch)
            loss = torch.nn.functional.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == config['ae_epochs'] - 1:
            print(f"  AE Epoch {epoch}, Avg Loss: {total_loss / len(dataloader):.6f}")
    
    ae.eval()
    return ae  # 返回 CPU 上的模型


def build_benign_distribution(flat_updates, config):
    """
    构建良性更新的分布模型（AE + Diffusion）
    """
    input_dim = flat_updates.shape[1]
    latent_dim = config['ae_latent_dim']

    # Step 1: 在 CPU 上创建并训练 AE
    print("Training AutoEncoder on benign updates (on CPU)...")
    ae = AutoEncoder(input_dim, latent_dim)
    ae = train_autoencoder_on_cpu(ae, flat_updates, config)

    # Step 2: 在 CPU 上提取 latents
    print("Extracting latents on CPU...")
    with torch.no_grad():
        latents = []
        batch_size = config.get('ae_batch_size', 64)
        for i in range(0, len(flat_updates), batch_size):
            batch = flat_updates[i:i + batch_size]  # flat_updates 默认在 CPU
            _, latent_batch = ae(batch)
            latents.append(latent_batch)
        latents = torch.cat(latents, dim=0)  # [N, latent_dim] on CPU

    # Step 3: 将 latents 移到 GPU 供 Diffusion 使用
    device = config['device']
    latents = latents.to(device)

    # Step 4: 在 GPU 上训练 Diffusion 模型
    print("Training Diffusion Model on benign latents (on GPU)...")
    diffusion_trainer = DiffusionTrainer(
        latent_dim=latent_dim,
        timesteps=config['diffusion_timesteps'],
        lr=config['diffusion_learning_rate'],
        device=device
    )
    diffusion_trainer.train(
        latents,
        epochs=config['diffusion_epochs'],
        batch_size=config.get('diffusion_batch_size', 32)
    )

    # Step 5: 将 AE 也移到 GPU（后续防御阶段需要快速推理）
    ae = ae.to(device)

    return ae, diffusion_trainer