import torch
from recon_trace.tracer import ReconTracer

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4,4)
    def forward(self, x):
        return self.linear(x)

def test_tracer_basic():
    model = DummyModel()
    tracer = ReconTracer(model)
    tracer.add_hook("linear")
    x = torch.randn(1,4)
    out = tracer.run(x)
    assert "traces" in tracer.data
    assert len(tracer.data["traces"]) > 0
    print("Basic tracer test passed")
