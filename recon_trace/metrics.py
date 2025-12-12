import torch

def attention_drift_score(original, ablated):
    return torch.norm(original - ablated).item()

def circuit_stability_score(activations_list):
    if len(activations_list) < 2:
        return 1.0
    diffs = [torch.norm(activations_list[i] - activations_list[i+1]).item() 
             for i in range(len(activations_list)-1)]
    return 1.0 / (1.0 + sum(diffs)/len(diffs))
