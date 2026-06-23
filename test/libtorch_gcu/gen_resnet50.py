import torch
from torchvision.models import resnet50

output_path = "resnet50_traced.pt"

model = resnet50()
model.eval()
example_input = torch.rand(1, 3, 224, 224)

traced_model = torch.jit.trace(model, example_input)
traced_model.save(output_path)