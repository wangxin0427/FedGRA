# client/trainer.py

import torch
import torch.nn.functional as F
import copy
from defense.utils import is_float_tensor, clamp_updates
import os
import sys

# 获取当前文件 (trainer.py) 所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (pythonProject2_DBFL)
root_dir = os.path.dirname(current_dir)
# 如果根目录不在系统路径中，则添加它
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 后门攻击模块（需确保 backdoor_attack.py 在项目路径中）
try:
    from attacks.backdoor_attack import BackdoorAttack
except ImportError:
    BackdoorAttack = None
    print("Warning: BackdoorAttack module not found. Backdoor attack will be disabled.")


def apply_data_attack(data, target, attack_type, num_classes=10, noise_std=0.05):
    """
    数据层攻击：标签翻转、样本投毒。
    后门攻击不在此处理，因其需要更复杂的触发器逻辑。
    """
    if attack_type == "label_flipping":
        # 标签翻转到对立类别
        return data, num_classes - 1 - target
    elif attack_type == "sample_poisoning":
        # 添加高斯噪声并截断到 [0,1]
        noise = torch.randn_like(data) * noise_std
        poisoned_data = torch.clamp(data + noise, 0.0, 1.0)
        return poisoned_data, target
    else:
        return data, target


def client_train(
    client_id,
    global_model,
    dataset,
    local_epochs,
    batch_size,
    learning_rate,
    device,
    grad_clip=1.0,
    is_malicious=False,
    attack_type="no_attack",
    num_classes=10,
    # 数据攻击参数
    data_attack_noise_std=0.05,
    # 后门攻击参数
    backdoor_target=0,
    backdoor_ratio=0.3,
    backdoor_trigger='pixel',
):
    """
    客户端本地训练。

    数据层攻击（label_flipping, sample_poisoning, backdoor）在训练循环内完成；
    拜占庭梯度攻击（gaussian_noise, sign_flipping 等）由主循环在 delta 上后处理。

    Args:
        client_id: 客户端 ID
        global_model: 全局模型
        dataset: 本地数据集
        local_epochs: 本地训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        device: 计算设备
        grad_clip: 梯度裁剪阈值
        is_malicious: 是否为恶意客户端
        attack_type: 攻击类型，可选值：
            - "no_attack"
            - "label_flipping"
            - "sample_poisoning"
            - "backdoor"
            - 其他拜占庭攻击类型（由主循环后处理，此处忽略）
        num_classes: 数据集类别数（用于标签翻转）
        data_attack_noise_std: 样本投毒的高斯噪声标准差
        backdoor_target: 后门目标类别
        backdoor_ratio: 后门投毒比例
        backdoor_trigger: 后门触发器类型 ('pixel' 或 'pattern')
    """
    model = copy.deepcopy(global_model).to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # 保存初始参数（仅浮点张量）
    original_state = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if is_float_tensor(param)
    }

    # 判断数据层攻击类型
    is_data_attack = is_malicious and attack_type in ("label_flipping", "sample_poisoning")
    is_backdoor = is_malicious and attack_type == "backdoor"

    # 如果启用后门攻击，创建 BackdoorAttack 实例
    backdoor = None
    if is_backdoor:
        if BackdoorAttack is None:
            raise RuntimeError("BackdoorAttack module not available. Please ensure backdoor_attack.py is in the path.")
        backdoor = BackdoorAttack(
            target_label=backdoor_target,
            poison_ratio=backdoor_ratio,
            trigger_type=backdoor_trigger,
            device=device
        )

    # 训练循环
    for _ in range(local_epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # ---- 数据层攻击应用 ----
            if is_data_attack:
                data, target = apply_data_attack(
                    data, target, attack_type,
                    num_classes=num_classes,
                    noise_std=data_attack_noise_std
                )
            elif is_backdoor:
                data, target = backdoor.poison_batch(data, target)
            # -----------------------

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            if grad_clip is not None:
             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

    # 计算模型更新 delta
    delta = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_state and is_float_tensor(param):
                delta[name] = (param.data - original_state[name].to(device)).cpu()

    # 裁剪更新并返回
    return clamp_updates(delta, min_val=-1.0, max_val=1.0), loss.item()