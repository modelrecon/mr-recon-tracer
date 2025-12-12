import torch
import json
from datetime import datetime

class ReconTracer:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.data = {"traces": []}

    def _hook(self, layer_name):
        def fn(module, input, output):
            self.data["traces"].append({
                "layer": layer_name.split('.')[1],
                "module": layer_name.split('.')[-1],
                "activations": output.detach().cpu().tolist()
            })
        return fn

    def add_hook(self, layer_name):
        try:
            layer = dict(self.model.named_modules())[layer_name]
            self.hooks.append(layer.register_forward_hook(self._hook(layer_name)))
        except KeyError:
            print(f"[!] Layer {layer_name} not found.")

    def run(self, input_tensor):
        self.data["model"] = {"name": str(self.model.__class__.__name__), "version": "v0.1"}
        self.data["run_id"] = datetime.now().isoformat()
        self.data["input"] = input_tensor.tolist()
        with torch.no_grad():
            output = self.model(input_tensor)
        self.data["output"] = output.tolist()
        return output

    def export_json(self, path="trace.json"):
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
