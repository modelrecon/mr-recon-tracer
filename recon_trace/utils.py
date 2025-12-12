import torch

def to_tensor(x, device='cpu'):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x).to(device)
