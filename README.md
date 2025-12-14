# mr-recon-tracer

**The Machine Mind Detective Tool!**

mr-recon-tracer is a lightweight instrumentation framework for transformer models that captures internal activation and routing information in a structured form called the **ActivityCube**. It uses forward hooks to collect per-layer activity, produces rich multi-dimensional data, and exports it as structured JSON suitable for analysis, visualization, or safety auditing workflows.

This tool is part of the ModelRecon ecosystem and provides the data foundation for safety metrics, activation visualizations, and model behavior analysis.

---

## Features

- Automatic instrumentation of transformer layers with `add_hooks_all()`
- Collects detailed per-token activation metrics across model depth
- Optional export of raw tensor slices
- Captures or structures attention maps when available
- Produces ActivityCube datasets in JSON for multi-granular visualization
- Designed for PyTorch transformer models (Hugging Face compatible)

---

## Key Concepts that I think is going to be useful in for visualization and data stuff

### **ActivityCube**

The **ActivityCube** is a structured representation of internal model activation patterns, I think it will be useful to represent all activation debug data this way. It captures how the model processes a prompt across:

- **Tokens** — the input sequence positions
- **Layers** — the depth of the model
- **Metrics** — numerical summaries of activity (e.g., energy, mean, std)

This cube enables consistent analysis and visualization of model behavior across datasets and models.

The output JSON conforms to the following conceptual schema:

```jsonc
{
  "meta": {
    "model_name": "GPT2LMHeadModel",
    "timestamp": "2025-12-14T08:13:22.396674",
    "input_shape": [1, 2]
  },
  "tokens": ["Hello", "world"],
  "layers": [
    {
      "layer_index": 0,
      "layer_name": "transformer.h.0.attn",
      "core_metrics": {
        "energy": [0.0012, 0.0011],
        "mean_activation": [0.00076, 0.00075],
        "std_activation": [0.00043, 0.00044]
      },
      "attention_scores": [ … ]          // optional, if returned
    },
    // additional layers...
  ],
  "provenance": {
    "commit_hash": "…",
    "package_versions": {
      "torch": "2.1.0",
      "transformers": "4.35.0"
    }
  }
}
