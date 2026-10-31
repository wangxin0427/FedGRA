import copy
import torch
import torch.nn as nn


class FLTrustAggregator:
    """
    FLTrust: Byzantine-robust Federated Learning with Trust Score.
    核心思想：Server 使用干净的 Root Dataset 计算 Server Update，
    以 Server Update 为锚点计算每个客户端更新的 Trust Score (max(0, cos_sim))，
    并将所有客户端更新缩放到与 Server Update 相同模长后加权聚合。
    """

    def __init__(self, model, root_dataset, batch_size=32, server_lr=0.01,
                 server_epochs=1, device=None):
        self.server_model = copy.deepcopy(model)
        self.root_loader = torch.utils.data.DataLoader(
            root_dataset, batch_size=batch_size, shuffle=True
        )
        self.server_lr = server_lr
        self.server_epochs = server_epochs
        self.device = device or next(model.parameters()).device
        self.server_model.to(self.device)

    # ------------------------------------------------------------------ #
    #                       内部工具方法                                   #
    # ------------------------------------------------------------------ #

    def _flatten_gradients(self, state_dict, target_device=None):
        """仅展平浮点型可训练参数，跳过 BN running stats 等非梯度张量"""
        tensors = []
        for v in state_dict.values():
            if not v.is_floating_point():
                continue
            t = v.flatten()
            if target_device is not None:
                t = t.to(target_device)
            tensors.append(t)

        if not tensors:
            raise ValueError("No valid gradient tensors found in state_dict!")
        return torch.cat(tensors)

    def _unflatten_state_dict(self, flat_tensor, reference_state_dict):
        """将一维向量还原为与 reference_state_dict 结构相同的字典"""
        result = {}
        offset = 0
        for name, param in reference_state_dict.items():
            if not param.is_floating_point():
                result[name] = param.clone()
                continue
            numel = param.numel()
            result[name] = flat_tensor[offset:offset + numel].view_as(param)
            offset += numel
        return result

    # ------------------------------------------------------------------ #
    #                     Server 端训练 & 更新计算                          #
    # ------------------------------------------------------------------ #

    def _train_server_model(self, global_model_on_device):
        """在 Root Dataset 上微调 Server Model"""
        self.server_model.load_state_dict(global_model_on_device.state_dict())
        self.server_model.train()

        optimizer = torch.optim.SGD(self.server_model.parameters(), lr=self.server_lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.server_epochs):
            for data, target in self.root_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.server_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

    def _compute_server_update(self, global_model_on_device):
        """Server Update = ServerModel.params - GlobalModel.params"""
        server_update = {}
        for (name, server_param), (_, global_param) in zip(
            self.server_model.named_parameters(),
            global_model_on_device.named_parameters()
        ):
            server_update[name] = server_param.data - global_param.data
        
        # ✅ 优化：将展平操作移到循环外部，避免重复计算
        flat = self._flatten_gradients(server_update)
        return server_update

    # ------------------------------------------------------------------ #
    #                      Trust Score & 归一化                             #
    # ------------------------------------------------------------------ #

    def compute_trust_scores(self, client_updates, server_update):
        """Trust Score = max(0, cosine_similarity(client_update, server_update))"""
        server_flat = self._flatten_gradients(server_update)
        server_norm = torch.norm(server_flat)

        if server_norm < 1e-8:
            return [1.0] * len(client_updates)

        trust_scores = []
        for update in client_updates:
            client_flat = self._flatten_gradients(update, target_device=server_flat.device)
            client_norm = torch.norm(client_flat)

            if client_norm < 1e-8:
                trust_scores.append(0.0)
                continue

            cos_sim = torch.dot(client_flat, server_flat) / (client_norm * server_norm + 1e-8)
            trust_scores.append(max(0.0, cos_sim.item()))

        return trust_scores

    def normalize_updates(self, client_updates, server_update):
        """将每个客户端更新缩放到与 Server Update 相同的模长"""
        server_flat = self._flatten_gradients(server_update)
        server_norm = torch.norm(server_flat)
        normalized_updates = []

        for update in client_updates:
            client_flat = self._flatten_gradients(update, target_device=server_flat.device)
            client_norm = torch.norm(client_flat)

            if client_norm < 1e-8 or server_norm < 1e-8:
                zero_update = {k: torch.zeros_like(v) for k, v in update.items()}
                normalized_updates.append(zero_update)
                continue

            scaled_flat = client_flat * (server_norm / client_norm)
            scaled_update = self._unflatten_state_dict(scaled_flat, update)
            normalized_updates.append(scaled_update)

        return normalized_updates

    # ------------------------------------------------------------------ #
    #                           主聚合接口                                  #
    # ------------------------------------------------------------------ #

    def aggregate(self, global_model, client_updates, client_ids=None):
        """
        执行完整的 FLTrust 聚合流程。
        Args:
            global_model: 当前全局模型
            client_updates: list[dict]，每个元素是一个客户端的 delta state_dict
            client_ids: 可选，客户端 ID 列表（保留接口兼容性）
        Returns:
            aggregated_dict: 聚合后的 delta state_dict，或 None
        """
        if not client_updates:
            return None

        target_device = next(self.server_model.parameters()).device

        # 将全局模型和客户端更新移到统一设备
        global_model_on_device = copy.deepcopy(global_model).to(target_device)
        client_updates_on_device = [
            {k: v.to(target_device) for k, v in update.items()}
            for update in client_updates
        ]

        # Step 1: 在 Root Dataset 上训练 Server Model
        self._train_server_model(global_model_on_device)

        # Step 2: 计算 Server Update
        server_update = self._compute_server_update(global_model_on_device)

        # Step 3: 计算 Trust Scores
        trust_scores = self.compute_trust_scores(client_updates_on_device, server_update)

        # Step 4: 归一化客户端更新
        normalized_updates = self.normalize_updates(client_updates_on_device, server_update)

        # Step 5: Trust-weighted 聚合
        total_weight = sum(trust_scores)
        
        # 防御机制：当无客户端可信时，直接使用 Server Update 作为本轮聚合结果
        if total_weight < 1e-8:
            return server_update  
            
        weights = trust_scores
        aggregated_dict = {}
        for name in normalized_updates[0].keys():
            aggregated_dict[name] = torch.zeros_like(normalized_updates[0][name])
            for i, update in enumerate(normalized_updates):
                aggregated_dict[name] += update[name] * weights[i]
            aggregated_dict[name] /= total_weight

        return aggregated_dict