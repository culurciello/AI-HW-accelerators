import torch
import torchvision.models as models

# Load the smallest standard ResNet
model = models.resnet18(weights='DEFAULT')

# Check the total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")

# Save the model
torch.save(model.state_dict(), "resnet18_small.pt")
