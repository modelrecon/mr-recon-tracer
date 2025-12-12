import torch
import random

def sample_activation(activation, sample_ratio=0.1):
    if not torch.is_tensor(activation):
        raise ValueError("activation must be a torch tensor")
    total = activation.numel()
    sample_size = max(1, int(total * sample_ratio))
    flat = activation.view(-1)
    indices = random.sample(range(total), sample_size)
    sampled = flat[indices]
    return sampled
