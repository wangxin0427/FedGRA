import torch
import random
import numpy as np
import torch.nn.functional as F
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100. * correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader.dataset)
    return acc, avg_loss


# utils/misc.py
def create_backdoor_testset(dataset_name, trigger_type, target_label, device):
    from torch.utils.data import DataLoader, TensorDataset
    from attacks.backdoor_attack import BackdoorAttack
    from utils.data_loader import load_dataset

    # 加载原始测试集
    _, test_dataset = load_dataset(dataset_name)
    loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    images, labels = next(iter(loader))

    # 创建后门攻击实例（仅用于生成投毒样本）
    backdoor = BackdoorAttack(
        target_label=target_label,
        poison_ratio=1.0,
        trigger_type=trigger_type,
        device=device
    )
    poisoned_images, poisoned_labels = backdoor.add_trigger(images)  # 返回 (img, label)

    # 包装成 DataLoader
    poisoned_dataset = TensorDataset(poisoned_images.cpu(), poisoned_labels.cpu())
    backdoor_loader = DataLoader(poisoned_dataset, batch_size=256, shuffle=False)
    return backdoor_loader

def test_backdoor(model, backdoor_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in backdoor_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total