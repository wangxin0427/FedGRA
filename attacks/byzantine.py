import torch
from defense.utils import is_float_tensor


def apply_byzantine_attack(delta, attack_type, strength=1.0, fixed_noise_std=None):
    attacked = {}
    if attack_type in ["label_flipping", "no_attack", None]:
        return delta

    if attack_type == "gaussian_noise":
        if fixed_noise_std is None:
            raise ValueError("fixed_noise_std must be provided for gaussian_noise attack.")
        for k, v in delta.items():
            if is_float_tensor(v):
                noise = torch.randn_like(v) * fixed_noise_std
                attacked[k] = v + noise
            else:
                attacked[k] = v
        return attacked
    elif attack_type == "random_weights":
        for k, v in delta.items():
            if is_float_tensor(v):
                attacked[k] = torch.randn_like(v) * (strength * v.std())
            else:
                attacked[k] = v

    elif attack_type == "scaling":
        for k, v in delta.items():
            if is_float_tensor(v):
                attacked[k] = v * strength
            else:
                attacked[k] = v

    elif attack_type == "sign_flipping":
        for k, v in delta.items():
            if is_float_tensor(v):
                attacked[k] = -v 
            else:
                attacked[k] = v

    else:
        return delta

    return attacked