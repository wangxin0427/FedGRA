import torch

def apply_data_attack(data, target, attack_type, num_classes=10):
    if attack_type == "label_flipping":
        return data, num_classes - 1 - target

    elif attack_type == "sample_poisoning":
        noise = torch.randn_like(data) * 0.05
        poisoned_data = torch.clamp(data + noise, 0.0, 1.0)
        return poisoned_data, target

    else:
        return data, target