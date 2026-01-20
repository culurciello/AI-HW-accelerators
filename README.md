# AI-HW-accelerators

![](ai-hw.png)

A frekishly funimplementation of neural networks building blocks and full networks in SystemVerilog. 
The repo pairs PyTorch models with fixed-point (Q8.8) SV implementations and Verilator-based tests for modules and entire networks.

## Status

| network | status |
| mlp1 | PASS |
| lenet5 | PASS |
| resnet18 | PASS |
| yolov8n | TBD |


## Layout
- `networks/`: PyTorch reference models and weight export helpers.
- `modules/`: SV modules that mirror PyTorch layers (Q8.8).
- `models/`: SV top-level network wiring.
- `tests/`: Python tests that generate inputs/weights, run Verilator, and compare HW vs SW.

## Precision
- Default fixed-point format: Q8.8 (`WIDTH=16`, `FRAC=8`).
- Python reference uses float math but quantizes after each layer to match SV.

## Requirements
- Python 3.10+
- PyTorch
- Verilator (5.x recommended)

## Quick start
Export weights and run tests:

```bash
# MLP
python networks/mlp/mlp.py
python tests/test_mlp_c1_real.py # tests/test_mlp_c1_real.py uses the exact PyTorch model weights exported from networks/mlp/mlp.py.

# LeNet-5
python networks/lenet5/lenet5.py
python tests/test_lenet5_real.py

# ResNet-18 (pretrained weights may require download)
python networks/resnet18/resnet18.py
python tests/test_resnet18_real.py
python tests/test_resnet18_layers.py
```

Fast ResNet-18 layer test:
```bash
RESNET18_FAST=1 RESNET18_INPUT=32 VERILATOR_THREADS=16 python tests/test_resnet18_layers.py
```

Module-level tests:

```bash
python tests/test_linear.py
python tests/test_relu.py
python tests/test_conv2d.py
python tests/test_avgpool2d.py
python tests/test_maxpool2d.py
python tests/test_batchnorm2d.py
python tests/test_add_vec.py
python tests/test_tanh.py
```

## Notes
- Verilator runs are multi-threaded by default (`--threads 16`). You can pass a
  different value by modifying `tests/common.py` or updating individual test calls.
- Tests print both Q8.8 integers and float equivalents for easier inspection.

## Networks
- `mlp_c1`: 10 -> 32 -> 32 -> 2 with ReLU between layers.
- `lenet5`: classic LeNet-5 (Conv/Tanh/AvgPool/FC) with biases.
- `resnet18`: standard ResNet-18 (Conv/BN/ReLU/MaxPool/AvgPool/FC).
