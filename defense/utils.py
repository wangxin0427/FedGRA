import torch

def flatten_params(params_dict):
    """将模型参数字典展平为一维向量（仅包含浮点型张量）"""
    flattened = []
    for key in sorted(params_dict.keys()):  # ←←← 唯一修改：加 sorted
        param = params_dict[key]
        if isinstance(param, torch.Tensor) and param.dtype in [torch.float32, torch.float64]:
            flattened.append(param.view(-1))
    if len(flattened) == 0:
        raise ValueError("No float tensors found in params_dict to flatten.")
    return torch.cat(flattened)

def is_float_tensor(tensor):
    return isinstance(tensor, torch.Tensor) and tensor.dtype in [torch.float32, torch.float64]

def clamp_updates(params_dict, min_val=-1.0, max_val=1.0):
    """对模型更新进行裁剪（in-place）"""
    clamped = {}
    for k, v in params_dict.items():
        if is_float_tensor(v):
            clamped[k] = torch.clamp(v, min_val, max_val)
        else:
            clamped[k] = v
    return clamped

def unflatten_params(flat_tensor, model_template):
    """将展平的一维向量恢复为模型参数字典"""
    param_shapes = []
    for param in model_template.parameters():
        param_shapes.append(param.shape)

    params_unflat = []
    start = 0
    for shape in param_shapes:
        numel = torch.prod(torch.tensor(shape)).item()
        param_flat = flat_tensor[start:start + numel]
        params_unflat.append(param_flat.view(shape))
        start += numel

    state_dict = {}
    for (name, _), param in zip(model_template.named_parameters(), params_unflat):
        state_dict[name] = param
    return state_dict