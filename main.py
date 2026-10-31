import os
import datetime
import torch
from client.trainer import client_train
from attacks.byzantine import apply_byzantine_attack
from defense.aggregation import federated_aggregation_with_defense
from defense.build_distribution import build_benign_distribution
from utils.config import load_config
from utils.data_split import split_dataset_noniid
from utils.misc import set_seed, test_model, test_backdoor, create_backdoor_testset
from defense.utils import flatten_params, unflatten_params
from models import create_model
from utils.data_loader import load_dataset

# 导入其他防御方法
from defense.robust_aggregators import aggregate
from defense.fltrust import FLTrustAggregator
from defense.frepd import train_vae_on_benign_updates, fit_distribution_to_errors, frepd_aggregate
from defense.signguard import signguard_aggregate
from defense.flad import flad_aggregate
from defense.sca import sca_aggregate 


def fedavg_aggregate(client_updates):
    """Simple FedAvg for warmup or fallback."""
    agg_delta = {}
    keys = list(client_updates.values())[0].keys()
    for k in keys:
        tensors = [client_updates[cid][k] for cid in client_updates]
        if isinstance(tensors[0], torch.Tensor) and torch.is_floating_point(tensors[0]):
            agg_delta[k] = torch.stack(tensors).mean(dim=0)
        else:
            agg_delta[k] = client_updates[list(client_updates.keys())[0]][k].clone()
    return agg_delta


# 定义数据层攻击类型集合
DATA_ATTACKS = {"label_flipping", "sample_poisoning", "backdoor"}


def train_client_and_get_update(cid, global_model, client_dataset, config, device,
                                is_malicious, attack_type, num_classes):
    """
    统一处理客户端训练及攻击后处理，返回模型更新 delta。
    - 数据层攻击：在 client_train 内部完成，返回 delta 后仅对后门做可选模型替换。
    - 梯度层攻击：在 client_train 返回 delta 后调用 apply_byzantine_attack 处理。
    """
    # 读取数据攻击通用参数
    data_attack_noise_std = config.get('poison_noise_std', 0.05)

    # 读取后门攻击参数
    backdoor_target = config.get('backdoor_target_label', 0)
    backdoor_ratio = config.get('backdoor_poison_ratio', 0.3)
    backdoor_trigger = config.get('backdoor_trigger_type', 'pixel')
    model_replacement_scale = config.get('model_replacement_scale', None)

    # 调用统一的 client_train（支持所有数据层攻击）
    delta, loss = client_train(
        cid, global_model, client_dataset,
        config['local_epochs'], config['batch_size'],
        config['learning_rate'], device, config['gradient_clip'],
        is_malicious=is_malicious,
        attack_type=attack_type,
        num_classes=num_classes,
        data_attack_noise_std=data_attack_noise_std,
        backdoor_target=backdoor_target,
        backdoor_ratio=backdoor_ratio,
        backdoor_trigger=backdoor_trigger
    )

    # 数据层攻击：已在训练中完成，不再对 delta 进行梯度后处理
    if is_malicious and attack_type in DATA_ATTACKS:
        # 后门攻击可选模型替换
        if attack_type == "backdoor" and model_replacement_scale is not None:
            for name in delta:
                delta[name] = delta[name] * model_replacement_scale
        return delta

    # 梯度层攻击（拜占庭类）：对 delta 施加后处理
    if is_malicious and attack_type != "no_attack":
        if attack_type == "gaussian_noise":
            noise_std = config.get("fixed_noise_std", 0.1)
            delta = apply_byzantine_attack(delta, attack_type, fixed_noise_std=noise_std)
        else:
            strength = config.get("attack_strengths", {}).get(attack_type, 1.0)
            delta = apply_byzantine_attack(delta, attack_type, strength)

    return delta


def main():
    config = load_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(config.get('seed', 42))

    # Load dataset and model
    dataset_name = config.get("dataset", "mnist")
    train_dataset, test_dataset = load_dataset(dataset_name)

    # ★ [FIX] 统一将 targets 转为 Python int，防止 tensor 作为 dict key 导致类别索引爆炸
    for ds in (train_dataset, test_dataset):
        if hasattr(ds, 'targets'):
            if isinstance(ds.targets, torch.Tensor):
                ds.targets = ds.targets.tolist()
            elif hasattr(ds.targets, 'tolist'):
                ds.targets = ds.targets.tolist()
            else:
                ds.targets = [int(t) for t in ds.targets]

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    global_model = create_model(dataset_name=dataset_name, device=device)

    # Determine num_classes
    dataset_clean = dataset_name.lower()
    num_classes = 10 if dataset_clean in ["mnist", "fashion_mnist", "cifar10"] else 100

    # Data partitioning
    client_datasets, client_sizes = split_dataset_noniid(
        train_dataset,
        config['num_clients'],
        config['noniid_alpha']
    )
    total_samples = sum(client_sizes.values())
    malicious_ids = set(range(config['malicious_clients']))  # only for simulation

    # Logging
    defense_method = config["defense_method"]
    attack_type = config.get("attack_type", "no_attack")
    exp_tag = f"{dataset_name}_{attack_type}_{defense_method}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = f"results/{exp_tag}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, f"{exp_tag}_{timestamp}.txt")

    # 如果是后门攻击，准备后门测试集
    backdoor_loader = None
    if attack_type == "backdoor":
        backdoor_loader = create_backdoor_testset(
            dataset_name=dataset_name,
            trigger_type=config.get('backdoor_trigger_type', 'pattern'),
            target_label=config.get('backdoor_target_label', 0),
            device=device
        )
        print(f"✅ Backdoor testset created, size: {len(backdoor_loader.dataset)}")

    # ==============================
    # CASE 1: OUR METHOD ("ours")
    # ==============================
    if defense_method == "ours":
        print(f"🚀 Running OUR method with attack: {attack_type}")
        collected_flattened = []

        # --- Warmup Phase (Real Attack Included) ---
        print("\n=== Warmup Phase (Real Attack Included) ===")
        benign_rounds = config['benign_phase_rounds']
        for round_num in range(benign_rounds):
            k = max(1, int(config['num_clients'] * config['selection_ratio']))
            selected = torch.randperm(config['num_clients'])[:k].tolist()
            client_updates = {}
            all_flats_this_round = []

            for cid in selected:
                # 1. 获取真实的攻击类型 (从 config 读取)
                real_attack_type = config.get("attack_type", "no_attack")
                # 2. 判断是否是恶意客户端 (根据 cid 是否在 malicious_ids 列表中)
                is_malicious = (cid in malicious_ids)

                delta = train_client_and_get_update(
                    cid,
                    global_model,
                    client_datasets[cid],
                    config,
                    device,
                    is_malicious=is_malicious,  # <--- 使用真实的判断结果
                    attack_type=real_attack_type,  # <--- 使用真实的攻击类型
                    num_classes=num_classes
                )
                client_updates[cid] = delta
                all_flats_this_round.append(flatten_params(delta).cpu())

            # Update global model (FedAvg)
            agg_delta = fedavg_aggregate(client_updates)
            with torch.no_grad():
                for name, param in global_model.named_parameters():
                    if name in agg_delta:
                        update = agg_delta[name].to(param.device)
                        update = torch.clamp(update, -0.1, 0.1)
                        param.add_(update)
            collected_flattened.extend(all_flats_this_round)
            acc, loss = test_model(global_model, test_loader, device)
            print(f"[Warmup {round_num + 1}] Acc: {acc:.2f}%, Loss: {loss:.4f}")
            with open(log_file, "a") as f:
                f.write(f"[Warm-up {round_num + 1}] Accuracy = {acc:.2f}%, Loss = {loss:.4f}\n")

        # --- Build Defense Model ---
        print("\n=== Building Defense (AE + Diffusion) ===")
        if len(collected_flattened) == 0:
            raise RuntimeError("No updates collected!")
        collected_tensor = torch.stack(collected_flattened)

        if config.get("use_warmup_filter", True):
            from utils.median_filter import filter_by_directional_median
            angle_thresh = config.get('warmup_angle_threshold', 90.0)
            before = collected_tensor.shape[0]
            collected_tensor = filter_by_directional_median(collected_tensor, angle_thresh)
            after = collected_tensor.shape[0]
            print(f"✅ Directional median filtering: kept {after}/{before} updates")

        print(f"Collected {collected_tensor.shape[0]} benign updates (dim={collected_tensor.shape[1]})")
        ae, diffusion_trainer = build_benign_distribution(collected_tensor, config)

        # --- Main Training Phase ---
        start_round = benign_rounds
        total_rounds = config['global_rounds']
        for round_num in range(start_round, total_rounds):
            k = max(1, int(config['num_clients'] * config['selection_ratio']))
            selected = torch.randperm(config['num_clients'])[:k].tolist()

            client_updates = {}
            for cid in selected:
                is_malicious = (cid in malicious_ids)
                delta = train_client_and_get_update(
                    cid, global_model, client_datasets[cid], config, device,
                    is_malicious, attack_type, num_classes
                )
                client_updates[cid] = delta

            agg_delta, risks = federated_aggregation_with_defense(
                global_model, client_updates, client_sizes, total_samples,
                ae, diffusion_trainer, config['lambda_param'], device,config
            )

            with torch.no_grad():
                for name, param in global_model.named_parameters():
                    if name in agg_delta:
                        update = torch.clamp(agg_delta[name], -0.05, 0.05)
                        param.add_(update)

            acc, loss = test_model(global_model, test_loader, device)
            current = round_num + 1

            if backdoor_loader is not None:
                asr = test_backdoor(global_model, backdoor_loader, device)
                print(f"[Round {current}] Acc: {acc:.2f}%, Loss: {loss:.4f}, ASR: {asr:.2f}%")
                with open(log_file, "a") as f:
                    f.write(f"[Round {current}] Accuracy = {acc:.2f}%, Loss = {loss:.4f}, ASR = {asr:.2f}%\n")
            else:
                print(f"[Round {current}] Acc: {acc:.2f}%, Loss: {loss:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"[Round {current}] Accuracy = {acc:.2f}%, Loss = {loss:.4f}\n")

    # ==============================
    # CASE: FLTRUST
    # ==============================
    elif defense_method == "fltrust":
            print(f"🛡️  Running FLTrust with attack: {attack_type}")

            # ---- 0. 数据集形状断言（防止配置错误导致实验无效）----
            dataset_name = config.get('dataset', 'unknown').upper()
            if hasattr(train_dataset, 'data'):
                data_shape = train_dataset.data.shape if hasattr(train_dataset.data, 'shape') else None
                print(f"[PRE-CHECK] {dataset_name} | data shape: {data_shape}")
                # CIFAR-10 应为 (50000, 32, 32, 3)，FMNIST 应为 (60000, 28, 28)
                if dataset_name == 'CIFAR10' and data_shape is not None:
                    assert data_shape[-1] == 3 and data_shape[1] == 32, \
                        f"CIFAR-10 shape mismatch! Expected (N,32,32,3), got {data_shape}"

            # ---- 1. 预留根数据集（通用分层采样，适配任意数据集）----
            # ⚠️ CIFAR-10 建议 >=500，FMNIST 可用 100
            default_root_size = 500 if dataset_name == 'CIFAR10' else 100
            root_size = config.get('fltrust_root_size', default_root_size)

            if len(train_dataset) > root_size:
                # 兼容 list / Tensor / numpy array 等所有 targets 类型
                targets = list(train_dataset.targets) \
                    if hasattr(train_dataset, 'targets') \
                    else [train_dataset[i][1] for i in range(len(train_dataset))]

                from collections import defaultdict, Counter
                import random

                class_indices = defaultdict(list)
                for idx, label in enumerate(targets):
                    class_indices[label].append(idx)

                num_classes = len(class_indices)
                samples_per_class = max(1, root_size // num_classes)

                root_indices = []
                remaining_indices = []

                # 固定种子确保可复现
                rng = random.Random(config.get('seed', 42))

                for label in sorted(class_indices.keys()):
                    indices = class_indices[label]
                    rng.shuffle(indices)
                    selected = indices[:samples_per_class]
                    root_indices.extend(selected)
                    remaining_indices.extend(indices[samples_per_class:])

                # 补齐余数
                shortfall = root_size - len(root_indices)
                if shortfall > 0 and remaining_indices:
                    extra = rng.sample(remaining_indices, min(shortfall, len(remaining_indices)))
                    root_indices.extend(extra)
                    extra_set = set(extra)
                    remaining_indices = [i for i in remaining_indices if i not in extra_set]

                # ✅ 打印验证（确认类别分布均匀）
                root_labels = [targets[i] for i in root_indices]
                dist = Counter(root_labels)
                print(f"[ROOT DIAG] {dataset_name} | size={len(root_indices)}, "
                      f"classes={len(dist)}, dist={dict(sorted(dist.items()))}")

                root_dataset = torch.utils.data.Subset(train_dataset, root_indices)
                train_dataset_for_clients = torch.utils.data.Subset(train_dataset, remaining_indices)
            else:
                root_dataset = train_dataset
                train_dataset_for_clients = train_dataset
                print(f"Warning: root_size ({root_size}) >= train set size ({len(train_dataset)})!")

            # 重新划分客户端数据（使用移除根数据集后的训练集）
            client_datasets_fl, client_sizes_fl = split_dataset_noniid(
                train_dataset_for_clients,
                config['num_clients'],
                config['noniid_alpha']
            )
            total_samples_fl = sum(client_sizes_fl.values())

            # ---- 2. 初始化 FLTrust 聚合器 ----
            fltrust = FLTrustAggregator(
                model=global_model,
                root_dataset=root_dataset,
                batch_size=config.get('fltrust_batch_size', 32),
                server_lr=config.get('fltrust_server_lr', 0.01),
                server_epochs=config.get('fltrust_server_epochs', 1),
                device=device
            )

            total_rounds = config['global_rounds']
            for round_num in range(total_rounds):
                k = max(1, int(config['num_clients'] * config['selection_ratio']))
                selected = torch.randperm(config['num_clients'])[:k].tolist()

                client_updates = {}
                for cid in selected:
                    is_malicious = (cid in malicious_ids)
                    delta = train_client_and_get_update(
                        cid, global_model, client_datasets_fl[cid], config, device,
                        is_malicious, attack_type, num_classes
                    )
                    client_updates[cid] = delta

                update_list = [client_updates[cid] for cid in selected]

                agg_delta = fltrust.aggregate(global_model, update_list)
                if agg_delta is None:
                    print(f"[FLTrust Round {round_num + 1}] Aggregation returned None, skipping.")
                    continue

                # ✅ 应用全局更新（无截断版本）
                with torch.no_grad():
                    for name, param in global_model.named_parameters():
                        if name in agg_delta:
                            update = agg_delta[name]
                            # 仅处理 NaN（避免训练中断），绝不截断 inf
                            update = torch.nan_to_num(update, nan=0.0, posinf=float('inf'), neginf=float('-inf'))
                            param.add_(update)
                        else:
                            # 仅记录警告，不执行任何更新操作
                            print(f"[WARN] Layer '{name}' missing from FLTrust agg_delta!")

                acc, loss = test_model(global_model, test_loader, device)
                print(f"[FLTrust Round {round_num + 1}] Acc: {acc:.2f}%, Loss: {loss:.4f}")

                with open(log_file, "a") as f:
                    f.write(f"[FLTrust Round {round_num + 1}] Accuracy = {acc:.2f}%, Loss = {loss:.4f}\n")
            return
# ==============================
# CASE: FREPD
# =====================  =========
    elif defense_method == "frepd":
        print(f"🛡️  Running FREPD with attack: {attack_type}")

        warmup_rounds = config.get('benign_phase_rounds', 5)
        collected_benign = []

        print("\n=== FREPD Warmup Phase (NO attack) ===")
        for round_num in range(warmup_rounds):
            k = max(1, int(config['num_clients'] * config['selection_ratio']))
            selected = torch.randperm(config['num_clients'])[:k].tolist()

            client_updates = {}
            for cid in selected:
                delta = train_client_and_get_update(
                    cid, global_model, client_datasets[cid], config, device,
                    is_malicious=False, attack_type="no_attack", num_classes=num_classes
                )
                client_updates[cid] = delta
                collected_benign.append(flatten_params(delta).cpu())

            agg_delta = fedavg_aggregate(client_updates)
            with torch.no_grad():
                for name, param in global_model.named_parameters():
                    if name in agg_delta:
                        update = agg_delta[name].to(param.device)  # 添加这一行
                        update = torch.clamp(update, -0.05, 0.05)
                        param.add_(update)

            acc, loss = test_model(global_model, test_loader, device)
            print(f"[FREPD Warmup {round_num+1}] Acc: {acc:.2f}%, Loss: {loss:.4f}")
            with open(log_file, "a") as f:
                f.write(f"[FREPD Warmup {round_num+1}] Accuracy = {acc:.2f}%, Loss = {loss:.4f}\n")

        if len(collected_benign) == 0:
            raise RuntimeError("No benign updates collected!")

        benign_tensor = torch.stack(collected_benign)
        vae, vae_info = train_vae_on_benign_updates(
            benign_tensor,
            latent_dim=config.get('vae_latent_dim', 100),
            epochs=config.get('vae_epochs', 50),
            lr=config.get('vae_lr', 1e-3),
            batch_size=config.get('vae_batch_size', 64),
            sample_dim=config.get('frepd_sample_dim', 2000),
            device=device
        )
        dist_name, dist_params = fit_distribution_to_errors(vae_info['errors'])
        print(f"Best-fitting distribution: {dist_name}, parameters: {dist_params}")

        total_rounds = config['global_rounds']
        for round_num in range(warmup_rounds, total_rounds):
            k = max(1, int(config['num_clients'] * config['selection_ratio']))
            selected = torch.randperm(config['num_clients'])[:k].tolist()

            client_updates = {}
            for cid in selected:
                is_malicious = (cid in malicious_ids)
                delta = train_client_and_get_update(
                    cid, global_model, client_datasets[cid], config, device,
                    is_malicious, attack_type, num_classes
                )
                client_updates[cid] = delta

            update_list = [client_updates[cid] for cid in selected]
            kept_indices = frepd_aggregate(
                update_list, vae, dist_name, dist_params,
                prob_threshold=config.get('frepd_prob_threshold', 0.9),
                sample_dim=config.get('frepd_sample_dim', 2000),
                device=device
            )
            if len(kept_indices) == 0:
                print("[FREPD] ⚠️ No clients kept, skipping round")
                continue
            kept_updates = [update_list[i] for i in kept_indices]
            agg_delta = {}
            keys = kept_updates[0].keys()
            for k in keys:
                tensors = [u[k].to(device) for u in kept_updates]
                agg_delta[k] = torch.stack(tensors).mean(dim=0)

            with torch.no_grad():
                for name, param in global_model.named_parameters():
                    if name in agg_delta:
                        update = torch.clamp(agg_delta[name], -0.05, 0.05)
                        param.add_(update)

            acc, loss = test_model(global_model, test_loader, device)
            current = round_num + 1
            print(f"[FREPD Round {current}] Acc: {acc:.2f}%, Loss: {loss:.4f}")
            with open(log_file, "a") as f:
                f.write(f"[FREPD Round {current}] Accuracy = {acc:.2f}%, Loss = {loss:.4f}\n")

    # ==============================
    # CASE: SIGNGUARD
    # ==============================
    elif defense_method == "signguard":
        print(f"🛡️  Running SignGuard with attack: {attack_type}")
        # SignGuard 不需要预热，直接进行每轮聚合
        total_rounds = config['global_rounds']
        for round_num in range(total_rounds):
            k = max(1, int(config['num_clients'] * config['selection_ratio']))
            selected = torch.randperm(config['num_clients'])[:k].tolist()
            client_updates = {}
            for cid in selected:
                is_malicious = (cid in malicious_ids)
                delta = train_client_and_get_update(
                    cid, global_model, client_datasets[cid], config, device,
                    is_malicious, attack_type, num_classes
                )
                client_updates[cid] = delta
            # 获取当前轮所有客户端更新的列表（顺序与 selected 一致）
            update_list = [client_updates[cid] for cid in selected]
            # SignGuard 聚合
            agg_flat = signguard_aggregate(
                update_list,
                L=config.get('signguard_L', 0.1),
                R=config.get('signguard_R', 3.0),
                coord_ratio=config.get('signguard_coord_ratio', 0.1),
                device=device
            )
            agg_delta = unflatten_params(agg_flat.to(device), global_model)
            with torch.no_grad():
                for name, param in global_model.named_parameters():
                    if name in agg_delta:
                        update = torch.clamp(agg_delta[name], -0.05, 0.05)
                        param.add_(update)
            acc, loss = test_model(global_model, test_loader, device)
            current = round_num + 1
            print(f"[SignGuard Round {current}] Acc: {acc:.2f}%, Loss: {loss:.4f}")
            with open(log_file, "a") as f:
                f.write(f"[SignGuard Round {current}] Accuracy = {acc:.2f}%, Loss = {loss:.4f}\n")
        # ==============================
        # CASE: flad
        # ==============================
    elif defense_method == "flad":
        print(f"🛡️  Running FLAD with attack: {attack_type}")

        # 预留服务器干净数据集（与 FLTrust 类似）
        root_size = config.get('flad_root_size', 100)
        if len(train_dataset) > root_size:
            root_indices = list(range(root_size))
            train_indices = list(range(root_size, len(train_dataset)))
            root_dataset = torch.utils.data.Subset(train_dataset, root_indices)
            train_dataset_for_clients = torch.utils.data.Subset(train_dataset, train_indices)
        else:
            root_dataset = train_dataset
            train_dataset_for_clients = train_dataset
            print("Warning: root dataset size is larger than training set!")

        # 重新划分客户端数据
        client_datasets_fl, client_sizes_fl = split_dataset_noniid(
            train_dataset_for_clients,
            config['num_clients'],
            config['noniid_alpha']
        )

        # 构建服务器数据加载器
        server_loader = torch.utils.data.DataLoader(
            root_dataset,
            batch_size=config.get('flad_server_batch_size', 32),
            shuffle=True
        )

        total_rounds = config['global_rounds']
        for round_num in range(total_rounds):
            k = max(1, int(config['num_clients'] * config['selection_ratio']))
            selected = torch.randperm(config['num_clients'])[:k].tolist()

            client_updates = {}
            for cid in selected:
                is_malicious = (cid in malicious_ids)
                delta = train_client_and_get_update(
                    cid, global_model, client_datasets_fl[cid], config, device,
                    is_malicious, attack_type, num_classes
                )
                client_updates[cid] = delta

            agg_delta = flad_aggregate(global_model, client_updates, server_loader, device, config)
            if agg_delta is None:
                print(f"[FLAD Round {round_num + 1}] No honest clients, skipping update.")
                continue

            with torch.no_grad():
                for name, param in global_model.named_parameters():
                    if name in agg_delta:
                        update = torch.clamp(agg_delta[name], -0.05, 0.05)
                        param.add_(update)

            acc, loss = test_model(global_model, test_loader, device)
            print(f"[FLAD Round {round_num + 1}] Acc: {acc:.2f}%, Loss: {loss:.4f}")
            with open(log_file, "a") as f:
                f.write(f"[FLAD Round {round_num + 1}] Accuracy = {acc:.2f}%, Loss = {loss:.4f}\n")
     # ==============================
      # CASE: SCA (Enhanced Spectral Clustering)
    # ==============================
    elif defense_method == "sca":
        print(f"🛡️  Running SCA with attack: {attack_type}")
        
        # SCA 核心超参数（带默认值兜底）
        sca_sigma = config.get('sca_sigma', 1.0)
        sca_sub_dim = config.get('sca_sub_dim', 75000)
        
        # SCA 是无监督谱方法，无需 Warmup，直接全量训练
        total_rounds = config['global_rounds']
        for round_num in range(total_rounds):
            k = max(1, int(config['num_clients'] * config['selection_ratio']))
            selected = torch.randperm(config['num_clients'])[:k].tolist()

            client_updates = {}
            for cid in selected:
                is_malicious = (cid in malicious_ids)
                delta = train_client_and_get_update(
                    cid, global_model, client_datasets[cid], config, device,
                    is_malicious, attack_type, num_classes
                )
                client_updates[cid] = delta

            # ★ SCA 聚合：输入 list[dict]，返回 dict
            update_list = [client_updates[cid] for cid in selected]
            agg_delta = sca_aggregate(
                updates=update_list,
                sigma=sca_sigma,
                sub_dim=sca_sub_dim,
                device='cpu'  # ★ 谱聚类强制 CPU，避免 GPU 显存溢出
            )

            # 应用全局更新（与其他防御方法保持一致的 clamp 策略）
            with torch.no_grad():
                for name, param in global_model.named_parameters():
                    if name in agg_delta:
                        update = torch.clamp(agg_delta[name].to(device), -0.05, 0.05)
                        param.add_(update)

            acc, loss = test_model(global_model, test_loader, device)
            current = round_num + 1
            
            # 后门攻击评估（与其他分支对齐）
            if backdoor_loader is not None:
                asr = test_backdoor(global_model, backdoor_loader, device)
                print(f"[SCA Round {current}] Acc: {acc:.2f}%, Loss: {loss:.4f}, ASR: {asr:.2f}%")
                with open(log_file, "a") as f:
                    f.write(f"[SCA Round {current}] Accuracy = {acc:.2f}%, Loss = {loss:.4f}, ASR = {asr:.2f}%\n")
            else:
                print(f"[SCA Round {current}] Acc: {acc:.2f}%, Loss: {loss:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"[SCA Round {current}] Accuracy = {acc:.2f}%, Loss = {loss:.4f}\n")
    # ==============================
    # CASE 2: BASELINE METHODS (FedAvg, Krum, Median, TrimmedMean)
    # ==============================
    else:
        print(f"🛡️  Running baseline: {defense_method}, attack: {attack_type}")
    
    # 【关键】预提取参数名顺序，保证所有客户端对齐
    param_names = [name for name, _ in global_model.named_parameters()]

    for round_num in range(config['global_rounds']):
        k = max(1, int(config['num_clients'] * config['selection_ratio']))
        selected = torch.randperm(config['num_clients'])[:k].tolist()

        # 1. 收集更新时，【必须】同时记录样本数
        client_updates = [] 
        for cid in selected:
            is_malicious = (cid in malicious_ids)
            delta = train_client_and_get_update(
                cid, global_model, client_datasets[cid], config, device,
                is_malicious, attack_type, num_classes
            )
            num_samples = len(client_datasets[cid])
            client_updates.append((delta, num_samples))

        # 2. 根据方法执行聚合（直接输出 dict，彻底废弃 unflatten）
        if defense_method == "fedavg":
            total_samples = sum(n for _, n in client_updates)
            agg_delta = {}
            for name in param_names:
                weighted_sum = torch.zeros_like(global_model.state_dict()[name], dtype=torch.float32, device=device)
                for delta_dict, n in client_updates:
                    if name in delta_dict:
                        weighted_sum += delta_dict[name].to(device) * (n / total_samples)
                agg_delta[name] = weighted_sum

        elif defense_method == "krum":
            f = config['malicious_clients']
            n = len(client_updates)
            if n <= 2 * f + 2:
                raise ValueError(f"Krum requires n > 2f+2, got n={n}, f={f}")
            
            # Krum 内部临时展平，对外透明
            flat_grads = []
            for delta_dict, _ in client_updates:
                flat = torch.cat([delta_dict[name].flatten().float().to(device) for name in param_names])
                flat_grads.append(flat)
            grad_tensor = torch.stack(flat_grads)

            # Krum 核心距离计算（保留你的高效矩阵运算）
            sq_norms = torch.sum(grad_tensor ** 2, dim=1)
            dist_matrix = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * torch.mm(grad_tensor, grad_tensor.t())
            dist_matrix.clamp_(min=0.0).fill_diagonal_(float('inf'))

            num_neighbors = n - f - 2
            sorted_dists, _ = torch.sort(dist_matrix, dim=1)
            krum_scores = torch.sum(sorted_dists[:, :num_neighbors], dim=1)

            # Multi-Krum 聚合（比单选更稳定）
            top_k = n - f
            _, top_indices = torch.topk(krum_scores, top_k, largest=False)
            selected_flat = torch.mean(grad_tensor[top_indices], dim=0)

            # 还原为字典格式
            agg_delta = {}
            offset = 0
            for name in param_names:
                shape = global_model.state_dict()[name].shape
                numel = global_model.state_dict()[name].numel()
                agg_delta[name] = selected_flat[offset:offset + numel].view(shape).to(device)
                offset += numel
        else:
            raise ValueError(f"Unsupported baseline: {defense_method}")

        # 3. 应用全局更新（仅做防崩溃兜底，不做数值截断）
        # 3. 应用全局更新（真正的无截断版本）
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if name in agg_delta:
                    update = agg_delta[name]
                    # 仅处理 NaN（避免训练中断），绝不截断 inf
                    update = torch.nan_to_num(update, nan=0.0, posinf=float('inf'), neginf=float('-inf'))
                    param.add_(update)

        acc, loss = test_model(global_model, test_loader, device)
        current = round_num + 1
        print(f"[Round {current}] Acc: {acc:.2f}%")
        with open(log_file, "a") as f:
            f.write(f"[Round {current}] Accuracy = {acc:.2f}%, Loss = {loss:.4f}\n")

    print(f"✅ Experiment finished. Log saved to: {log_file}")


if __name__ == "__main__":
    main()