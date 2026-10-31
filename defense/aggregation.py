import torch
import numpy as np
from .utils import flatten_params, is_float_tensor


def federated_aggregation_with_defense(
    global_model, client_updates, client_sizes, total_samples,
    ae, diffusion_trainer, lambda_param, device, config
):
    """
    AE + DM 融合评分 + 动态加权聚合
    支持4组单变量消融: use_ae / use_dm / use_dynamic_weight
    Warmup过滤在外部训练循环中控制，不在此函数内
    """
    client_ids = list(client_updates.keys())
    num_clients = len(client_ids)

    # 1. Flatten
    flat_updates = torch.stack([
        flatten_params(client_updates[cid]).to(device) for cid in client_ids
    ])

    # 2. 消融开关
    use_ae = config.get("use_ae", True)
    use_dm = config.get("use_dm", True)
    use_dynamic_weight = config.get("use_dynamic_weight", True)

    # 3. 评分计算
    ae_risk = np.zeros(num_clients)
    dm_risk = np.zeros(num_clients)
    latents = None

    with torch.no_grad():
        if use_ae:
            ae.eval()
            recon_flat, latents = ae(flat_updates)
            ae_errors = torch.norm(flat_updates - recon_flat, dim=1).cpu().numpy()
            ae_risk = ae_errors / (np.median(ae_errors) + 1e-8)

        if use_dm:
            if latents is None:
                ae.eval()
                _, latents = ae(flat_updates)
            dm_errors = diffusion_trainer.multi_step_anomaly_score(
                latents, num_steps=10
            ).cpu().numpy()
            dm_risk = dm_errors / (np.median(dm_errors) + 1e-8)

    # 4. 固定融合（不消融）
    if use_ae and use_dm:
        fused_risk = 0.5 * ae_risk + 0.5 * dm_risk
    elif use_ae:
        fused_risk = ae_risk
    elif use_dm:
        fused_risk = dm_risk
    else:
        fused_risk = np.zeros(num_clients)

    # 5. Risk dict
    risk_scores = {cid: float(fused_risk[i]) for i, cid in enumerate(client_ids)}

    # 6. Dynamic vs Fixed lambda
    if use_dynamic_weight:
        risk_std = np.std(fused_risk)
        dynamic_lambda = lambda_param * (1.0 + risk_std)
        dynamic_lambda = np.clip(dynamic_lambda, 1.0, 20.0)
    else:
        dynamic_lambda = lambda_param

    # 7. 权重计算
    weights = {cid: np.exp(-dynamic_lambda * risk) for cid, risk in risk_scores.items()}
    total_w = sum(weights.values()) or 1.0

    # 8. 加权聚合
    agg_update = {}
    for k, v in client_updates[client_ids[0]].items():
        if is_float_tensor(v):
            weighted_sum = torch.zeros_like(v).to(device)
            for cid in client_ids:
                w = weights[cid] / total_w
                weighted_sum += w * client_updates[cid][k].to(device)
            agg_update[k] = weighted_sum
        else:
            agg_update[k] = v.clone()

    return agg_update, risk_scores