# SV Modules

Reusable SystemVerilog blocks for fixed-point neural networks (Q8.8).

## Modules
- `linear.sv`: fully-connected layer with optional bias (`WEIGHTS_FILE`, `BIAS_FILE`).
- `relu.sv`: ReLU activation on packed vectors.
- `conv2d.sv`: 2D convolution (channel-major flat vector: C,H,W).
- `avgpool2d.sv`: average pooling (KxK, stride).
- `maxpool2d.sv`: max pooling (KxK, stride, padding).
- `batchnorm2d.sv`: per-channel batchnorm using precomputed scale/bias.
- `add_vec.sv`: elementwise add for residual paths.
- `tanh.sv`: tanh activation using a generated lookup table.

## Interfaces
All modules use packed vectors of signed Q8.8 values:
- Inputs are flattened in channel-major order.
- Outputs are flattened the same way.

### Memory files
Weights and biases are loaded with `$readmemh`.
- Weights are written in row-major order (PyTorch `weight.flatten()`).
- Bias is a flat vector (one per output channel or output neuron).

## Parameters
Most modules expose:
- `WIDTH`: bit width (default 16)
- `FRAC`: fractional bits (default 8)
- `precision`: descriptive string (`"Q8.8"`)

## Testing
Module tests live in `tests/` and use Verilator to compare HW vs SW:
```bash
python tests/test_linear.py
python tests/test_relu.py
python tests/test_conv2d.py
python tests/test_avgpool2d.py
python tests/test_tanh.py
```
