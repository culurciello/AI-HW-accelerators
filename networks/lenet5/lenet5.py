import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # C1: Input 1x32x32 (standard for LeNet), Output 6x28x28
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        # S2: Avg Pool, Output 6x14x14
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: Output 16x10x10
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        # S4: Avg Pool, Output 16x5x5
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C5: Flattened 16*5*5 = 400 inputs -> 120 outputs
        self.c5 = nn.Linear(16 * 5 * 5, 120)
        # F6: 120 -> 84
        self.f6 = nn.Linear(120, 84)
        # Output: 84 -> num_classes (e.g., 10)
        self.output = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.tanh(self.c1(x)) # Original LeNet used Tanh
        x = self.s2(x)
        x = torch.tanh(self.c3(x))
        x = self.s4(x)
        x = x.view(-1, 16 * 5 * 5) # Flatten
        x = torch.tanh(self.c5(x))
        x = torch.tanh(self.f6(x))
        x = self.output(x)
        return x

def export_weights_py(path: str | Path, seed: int = 0, num_classes: int = 10) -> None:
    torch.manual_seed(seed)
    model = LeNet5(num_classes=num_classes)
    weights = {
        "C1_W": model.c1.weight.detach().tolist(),
        "C1_B": model.c1.bias.detach().tolist(),
        "C3_W": model.c3.weight.detach().tolist(),
        "C3_B": model.c3.bias.detach().tolist(),
        "C5_W": model.c5.weight.detach().tolist(),
        "C5_B": model.c5.bias.detach().tolist(),
        "F6_W": model.f6.weight.detach().tolist(),
        "F6_B": model.f6.bias.detach().tolist(),
        "OUT_W": model.output.weight.detach().tolist(),
        "OUT_B": model.output.bias.detach().tolist(),
    }
    path = Path(path)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# Auto-generated weights from LeNet5\n")
        for name, values in weights.items():
            handle.write(f"{name} = {values}\n\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    # 1. Instantiate the model
    model = LeNet5(num_classes=10)

    # 2. Save the model in .pt format
    PATH = Path(__file__).with_name("lenet5_model.pt")
    torch.save(model.state_dict(), PATH)

    print(f"LeNet-5 model structure created and saved to {PATH}")
    export_weights_py(Path(__file__).with_name("lenet5_weights.py"), seed=0, num_classes=10)
