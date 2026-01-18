import torch
import torchvision.models as models
from pathlib import Path

def export_weights_py(path: str | Path, weights="DEFAULT") -> None:
    model = models.resnet18(weights=weights)
    state_dict = model.state_dict()
    path = Path(path)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# Auto-generated ResNet-18 state_dict\n")
        handle.write("STATE_DICT = {\n")
        for key, tensor in state_dict.items():
            handle.write(f"    {key!r}: {tensor.tolist()},\n")
        handle.write("}\n")


if __name__ == "__main__":
    # Load the smallest standard ResNet
    model = models.resnet18(weights="DEFAULT")

    # Check the total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    # Save the model
    PATH = Path(__file__).with_name("resnet18_small.pt")
    torch.save(model.state_dict(), PATH)
    print(f"Model saved successfully to {PATH}")

    export_weights_py(Path(__file__).with_name("resnet18_weights.py"), weights="DEFAULT")
