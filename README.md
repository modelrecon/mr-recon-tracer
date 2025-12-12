# mr-recon-tracer
The Machine Mind Detective Tool!

recon-tracer is a light-touch circuit tracer for transformers. It hooks layers, samples activations, does weak ablations, and saves full circuit data as JSON. Ideal for creating the model data that I will use for afety checkup.

**Features:**
- Forward hooks for activations and attention
- Weak ablation of neurons
- Activation sampling
- Full JSON trace output (Circuit Tracer types)
- Compatible with PyTorch transformers

**Quick Start:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from recon_trace.tracer import ReconTracer

model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tracer = ReconTracer(model) #I know this looks like circuit tracer - but I will improve it
tracer.add_hook("transformer.h.0.mlp")
inputs = tokenizer("Hello world", return_tensors="pt")["input_ids"]

tracer.run(inputs)
tracer.export_json("trace_output.json")
