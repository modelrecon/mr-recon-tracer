import torch
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


class ReconTracer:
    """
    ReconTracer that produces an Activity Cube:
      tokens × layers × metrics
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metric_funcs: Optional[Dict[str, Callable[[torch.Tensor], float]]] = None,
        include_attention_details: bool = False,
        include_raw_tensors: bool = False,
    ):
        self.model = model
        self.hooks = []
        self.metric_funcs = metric_funcs or self._default_metrics()
        self.include_attention_details = include_attention_details
        self.include_raw_tensors = include_raw_tensors
        self.activity_cube: Dict[str, Any] = {
            "meta": {},
            "tokens": [],
            "layers": [],
            "provenance": {}
        }

    # ------------------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------------------

    def _default_metrics(self) -> Dict[str, Callable[[torch.Tensor], float]]:
        """
        Default metrics for Activity Cube cells.
        """
        return {
            "energy": lambda t: t.norm().item(),
            "mean_activation": lambda t: t.mean().item(),
            "std_activation": lambda t: t.std().item(),
        }

    def _extract_tensor(self, obj: Any) -> Optional[torch.Tensor]:
        """
        Extract the first tensor from HF-style or nested outputs.
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

    def _compute_per_token_metrics(
        self, tensor: torch.Tensor
    ) -> Dict[str, List[float]]:
        """
        Compute per-token metrics from a 3D tensor [batch, seq_len, features].
        """
        seq_len = tensor.size(1)
        results: Dict[str, List[float]] = {}
        for name, func in self.metric_funcs.items():
            token_values = []
            for i in range(seq_len):
                token_values.append(func(tensor[0, i, :]))
            results[name] = token_values
        return results

    # ------------------------------------------------------------------------------
    # Hook logic
    # ------------------------------------------------------------------------------

    def _hook(self, layer_name: str):
        def fn(module, inputs, output):
            tensor_out = self._extract_tensor(output)
            if tensor_out is None:
                return

            hidden = tensor_out.detach().cpu()
            layer_obj = {
                "layer_index": len(self.activity_cube["layers"]),
                "layer_name": layer_name,
                "core_metrics": self._compute_per_token_metrics(hidden)
            }

            if self.include_raw_tensors:
                layer_obj["raw_tensors"] = {"hidden_states": hidden.tolist()}

            self.activity_cube["layers"].append(layer_obj)

        return fn


    def add_hook(self, layer_name: str):
        """
        Register a forward hook with a named layer.
        """
        modules = dict(self.model.named_modules())
        if layer_name not in modules:
            raise KeyError(f"Layer '{layer_name}' not in model modules.")
        handle = modules[layer_name].register_forward_hook(self._hook(layer_name))
        self.hooks.append(handle)

    def add_hooks_by_type(self, module_type: type):
        """
        Register hooks on all modules of a given class.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, module_type):
                self.hooks.append(module.register_forward_hook(self._hook(name)))

    def add_hooks_all(self, filter_fn=None):
        """
        Register forward hooks for all modules that satisfy filter_fn.
        If filter_fn is None, hooks all modules that have parameters.
        """
        for name, module in self.model.named_modules():
            # Skip root module with empty name
            if name == "":
                continue

            if filter_fn is None:
                # Hook modules with trainable params
                if any(p.requires_grad for p in module.parameters()):
                    self.add_hook(name)
            else:
                # Only hook modules that satisfy the filter
                if filter_fn(name, module):
                    self.add_hook(name)


    # ------------------------------------------------------------------------------
    # Run and export
    # ------------------------------------------------------------------------------

    def run(self, input_ids: torch.Tensor, **model_kwargs):
        self.activity_cube["meta"] = {
            "model_name": self.model.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
            "input_shape": list(input_ids.shape),
        }

        self.activity_cube["tokens"] = [
            str(x) for x in input_ids[0].tolist()
        ]

        with torch.no_grad():
            output = self.model(
                input_ids,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
                **model_kwargs
            )

        # Extract attention maps if available
        all_attentions = getattr(output, "attentions", None)
        if all_attentions:
            for layer_idx, attn_tensor in enumerate(all_attentions):
                if not isinstance(attn_tensor, torch.Tensor):
                    # skip None or unexpected types
                    continue

                # record attention scores
                attn_list = attn_tensor.detach().cpu().tolist()
                self.activity_cube["layers"].append({
                    "layer_index": layer_idx,
                    "layer_name": f"attention.{layer_idx}",
                    "attention_scores": attn_list
        })


        return output


    def export_json(self, path: str):
        """
        Writes the Activity Cube as JSON.
        """
        with open(path, "w") as f:
            json.dump(self.activity_cube, f, indent=2)

    def clear_hooks(self):
        """
        Removes all registered hooks.
        """
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
