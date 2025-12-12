import torch

def weak_ablation(activation, ablation_ratio=0.05):
    if not torch.is_tensor(activation):
        raise ValueError("activation must be a torch tensor")
    mask = torch.ones_like(activation)
    num_to_zero = max(1, int(activation.numel() * ablation_ratio))
    indices = torch.randperm(activation.numel())[:num_to_zero]
    mask.view(-1)[indices] = 0
    return activation * mask
