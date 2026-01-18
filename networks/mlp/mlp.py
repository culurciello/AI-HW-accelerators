import torch
import torch.nn as nn
from pathlib import Path

class mlp_c1(nn.Module):
    def __init__(self):
        super(mlp_c1, self).__init__()
        
        # Layer 1:
        self.layer1 = nn.Linear(10, 32, bias=False) 
        
        # Layer 2:
        self.layer2 = nn.Linear(32, 32, bias=False)
        
        # Layer 3:
        self.layer3 = nn.Linear(32, 2, bias=False)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def export_weights_py(path: str | Path, seed: int = 0) -> None:
    torch.manual_seed(seed)
    model = mlp_c1()
    weights = {
        "LAYER1_W": model.layer1.weight.detach().tolist(),
        "LAYER2_W": model.layer2.weight.detach().tolist(),
        "LAYER3_W": model.layer3.weight.detach().tolist(),
    }
    path = Path(path)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# Auto-generated weights from mlp_c1\n")
        for name, values in weights.items():
            handle.write(f"{name} = {values}\n\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    # Instantiate the model
    model = mlp_c1()
    print(model)

    # Verify weight counts
    for i, layer in enumerate([model.layer1, model.layer2, model.layer3]):
        print(f"Layer {i+1} weight count: {layer.weight.numel()}")

    # save model
    # Define the file path
    PATH = Path(__file__).with_name("mlp_c1.pt")

    # Save the model's learned parameters
    torch.save(model.state_dict(), PATH)

    print(f"Model saved successfully to {PATH}")

    export_weights_py(Path(__file__).with_name("mlp_weights.py"), seed=0)


# # reload later:

# # 1. Re-create the model structure
# loaded_model = mlp_c1()

# # 2. Load the saved weights
# loaded_model.load_state_dict(torch.load(PATH))

# # 3. Set to evaluation mode if you are doing inference
# loaded_model.eval()
