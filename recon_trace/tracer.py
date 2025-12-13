import torch
import json
from datetime import datetime
from typing import Any, Dict, Callable


class ReconTracer:
    """
    ReconTracer
    -----------
    Lightweight forward-pass tracer for transformer models.
    Captures per-layer activations in a JSON-safe reduced form.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        reducer: Callable[[torch.Tensor], Dict[str, Any]] | None = None,
    ):
        self.model = model
        self.hooks = []
        self.reducer = reducer or self._default_reducer
        self.data = {
            "meta": {},
            "traces": [],
        }

    # ---------------------------------------------------------------------
    # Core helpers
    # ---------------------------------------------------------------------

    def _extract_tensor(self, obj: Any) -> torch.Tensor | None:
        """
        Extract the first tensor from common HF-style outputs.
        """
        if torch.is_tensor(obj):
            return obj

        if isinstance(obj, (list, tuple)):
            for item in obj:
                if torch.is_tensor(item):
                    return item

        if hasattr(obj, "last_hidden_state"):
            return obj.last_hidden_state

        return None

    def _default_reducer(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Default JSON-safe activation summary.
        """
        t = tensor.detach().float().cpu()
        return {
            "shape": list(t.shape),
            "mean": t.mean().item(),
            "std": t.std().item(),
            "min": t.min().item(),
            "max": t.max().item(),
            "l2_norm": t.norm().item(),
        }

    # ---------------------------------------------------------------------
    # Hook logic
    # ---------------------------------------------------------------------

    def _hook(self, layer_name: str):
        def fn(module, inputs, output):
            tensor_out = self._extract_tensor(output)
            if tensor_out is None:
                return

            self.data["traces"].append(
                {
                    "layer": layer_name,
                    "module_type": module.__class__.__name__,
                    "reduction": self.reducer(tensor_out),
                }
            )

        return fn

    def add_hook(self, layer_name: str):
        """
        Register a forward hook on a named module.
        """
        modules = dict(self.model.named_modules())
        if layer_name not in modules:
            raise KeyError(f"Layer '{layer_name}' not found in model.")

        handle = modules[layer_name].register_forward_hook(
            self._hook(layer_name)
        )
        self.hooks.append(handle)

    def add_hooks_by_type(self, module_type: type):
        """
        Convenience: hook all modules of a given type
        (e.g., nn.Linear, nn.LayerNorm).
        """
        for name, module in self.model.named_modules():
            if isinstance(module, module_type):
                self.hooks.append(
                    module.register_forward_hook(self._hook(name))
                )

    # ---------------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------------

    def run(self, input_ids: torch.Tensor, **model_kwargs):
        self.data["meta"] = {
            "model_class": self.model.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
            "input_shape": list(input_ids.shape),
        }

        with torch.no_grad():
            output = self.model(input_ids, **model_kwargs)

        tensor_out = self._extract_tensor(output)
        if tensor_out is not None:
            self.data["meta"]["output_shape"] = list(tensor_out.shape)

        return output

    # ---------------------------------------------------------------------
    # Export / cleanup
    # ---------------------------------------------------------------------

    def export_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
