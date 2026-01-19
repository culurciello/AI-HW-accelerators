from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os

import torch
import torch.nn.functional as F

from common import (
    FRAC,
    REPO_ROOT,
    WIDTH,
    q88_from_float_tensor,
    read_mem,
    run_verilator,
    write_mem,
)

SCALE = 1 << FRAC
THREADS = int(os.environ.get("VERILATOR_THREADS", "16"))
INPUT_SIZE = int(os.environ.get("RESNET18_INPUT", "32"))


def _load_module(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dump_values(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write(" ".join(str(v) for v in values))
        handle.write("\n")


def _progress(idx: int, total: int, name: str) -> None:
    print(f"[{idx}/{total}] {name}", flush=True)


def conv2d_q88(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    stride: int,
    padding: int,
) -> torch.Tensor:
    x_f = x_q.to(torch.float32) / SCALE
    w_f = w_q.to(torch.float32) / SCALE
    y_f = F.conv2d(x_f, w_f, bias=None, stride=stride, padding=padding)
    return q88_from_float_tensor(y_f)


def bn_q88(x_q: torch.Tensor, scale_q: torch.Tensor, bias_q: torch.Tensor) -> torch.Tensor:
    x_f = x_q.to(torch.float32) / SCALE
    scale_f = scale_q.to(torch.float32) / SCALE
    bias_f = bias_q.to(torch.float32) / SCALE
    y_f = x_f * scale_f.view(1, -1, 1, 1) + bias_f.view(1, -1, 1, 1)
    return q88_from_float_tensor(y_f)


def relu_q88(x_q: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x_q, min=0)


def add_q88(a_q: torch.Tensor, b_q: torch.Tensor) -> torch.Tensor:
    return (a_q + b_q).to(torch.int16)


def maxpool2d_q88(x_q: torch.Tensor, kernel: int, stride: int, padding: int) -> torch.Tensor:
    x_f = x_q.to(torch.float32) / SCALE
    y_f = F.max_pool2d(x_f, kernel_size=kernel, stride=stride, padding=padding)
    return q88_from_float_tensor(y_f)


def avgpool2d_q88_int(x_q: torch.Tensor, kernel: int, stride: int) -> torch.Tensor:
    denom = kernel * kernel
    n, ch, in_h, in_w = x_q.shape
    out_h = (in_h - kernel) // stride + 1
    out_w = (in_w - kernel) // stride + 1
    out = torch.zeros((n, ch, out_h, out_w), dtype=torch.int16)
    x_int = x_q.to(torch.int32)
    for c in range(ch):
        for oh in range(out_h):
            for ow in range(out_w):
                patch = x_int[
                    0,
                    c,
                    oh * stride : oh * stride + kernel,
                    ow * stride : ow * stride + kernel,
                ]
                acc = int(patch.sum().item())
                if acc < 0:
                    acc = -((-acc + denom // 2) // denom)
                else:
                    acc = (acc + denom // 2) // denom
                out[0, c, oh, ow] = acc
    return out


def linear_q88(x_q: torch.Tensor, w_q: torch.Tensor, b_q: torch.Tensor) -> torch.Tensor:
    x_f = x_q.to(torch.float32) / SCALE
    w_f = w_q.to(torch.float32) / SCALE
    b_f = b_q.to(torch.float32) / SCALE
    y_f = x_f @ w_f.t() + b_f
    return q88_from_float_tensor(y_f)


def _bn_scale_bias(weight: torch.Tensor, bias: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, eps: float):
    scale = weight / torch.sqrt(var + eps)
    bias_out = bias - mean * scale
    return scale, bias_out


def _build_tb_stem(
    tb_path: Path,
    input_mem: Path,
    conv_w_mem: Path,
    bn_scale_mem: Path,
    bn_bias_mem: Path,
    output_mem: Path,
    in_h: int,
    in_w: int,
) -> None:
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int FRAC = {FRAC};
  localparam int IN_H = {in_h};
  localparam int IN_W = {in_w};
  localparam int OUT_H = (IN_H + 2*3 - 7) / 2 + 1;
  localparam int OUT_W = (IN_W + 2*3 - 7) / 2 + 1;
  localparam int MP_H = (OUT_H + 2*1 - 3) / 2 + 1;
  localparam int MP_W = (OUT_W + 2*1 - 3) / 2 + 1;

  logic signed [3*IN_H*IN_W*WIDTH-1:0] in_vec;
  logic signed [64*MP_H*MP_W*WIDTH-1:0] out_vec;
  logic signed [WIDTH-1:0] in_mem [0:3*IN_H*IN_W-1];
  logic signed [64*OUT_H*OUT_W*WIDTH-1:0] conv_out;
  logic signed [64*OUT_H*OUT_W*WIDTH-1:0] bn_out;
  logic signed [64*OUT_H*OUT_W*WIDTH-1:0] relu_out;

  integer i;
  integer fd;

  initial begin
    $readmemh("{input_mem.as_posix()}", in_mem);
    for (i = 0; i < 3*IN_H*IN_W; i = i + 1) begin
      in_vec[i*WIDTH +: WIDTH] = in_mem[i];
    end
    #1;
    fd = $fopen("{output_mem.as_posix()}", "w");
    for (i = 0; i < 64*MP_H*MP_W; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  conv2d #(
    .IN_CH(3),
    .OUT_CH(64),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(7),
    .STRIDE(2),
    .PADDING(3),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .WEIGHTS_FILE("{conv_w_mem.as_posix()}")
  ) conv1 (
    .in_vec(in_vec),
    .out_vec(conv_out)
  );

  batchnorm2d #(
    .CH(64),
    .IN_H(OUT_H),
    .IN_W(OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .SCALE_FILE("{bn_scale_mem.as_posix()}"),
    .BIAS_FILE("{bn_bias_mem.as_posix()}")
  ) bn1 (
    .in_vec(conv_out),
    .out_vec(bn_out)
  );

  relu #(
    .DIM(64*OUT_H*OUT_W),
    .WIDTH(WIDTH)
  ) relu1 (
    .in_vec(bn_out),
    .out_vec(relu_out)
  );

  maxpool2d #(
    .CH(64),
    .IN_H(OUT_H),
    .IN_W(OUT_W),
    .K(3),
    .STRIDE(2),
    .PADDING(1),
    .WIDTH(WIDTH)
  ) mp (
    .in_vec(relu_out),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def _build_tb_block(
    tb_path: Path,
    input_mem: Path,
    conv1_w_mem: Path,
    bn1_scale_mem: Path,
    bn1_bias_mem: Path,
    conv2_w_mem: Path,
    bn2_scale_mem: Path,
    bn2_bias_mem: Path,
    output_mem: Path,
    in_ch: int,
    out_ch: int,
    in_h: int,
    in_w: int,
    stride: int,
    ds_conv_mem: Path | None,
    ds_bn_scale_mem: Path | None,
    ds_bn_bias_mem: Path | None,
) -> None:
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int FRAC = {FRAC};
  localparam int IN_CH = {in_ch};
  localparam int OUT_CH = {out_ch};
  localparam int IN_H = {in_h};
  localparam int IN_W = {in_w};
  localparam int STRIDE = {stride};
  localparam int OUT_H = (IN_H + 2*1 - 3) / STRIDE + 1;
  localparam int OUT_W = (IN_W + 2*1 - 3) / STRIDE + 1;

  logic signed [IN_CH*IN_H*IN_W*WIDTH-1:0] in_vec;
  logic signed [OUT_CH*OUT_H*OUT_W*WIDTH-1:0] out_vec;
  logic signed [WIDTH-1:0] in_mem [0:IN_CH*IN_H*IN_W-1];
  logic signed [OUT_CH*OUT_H*OUT_W*WIDTH-1:0] conv1_out;
  logic signed [OUT_CH*OUT_H*OUT_W*WIDTH-1:0] bn1_out;
  logic signed [OUT_CH*OUT_H*OUT_W*WIDTH-1:0] relu1_out;
  logic signed [OUT_CH*OUT_H*OUT_W*WIDTH-1:0] conv2_out;
  logic signed [OUT_CH*OUT_H*OUT_W*WIDTH-1:0] bn2_out;
  logic signed [OUT_CH*OUT_H*OUT_W*WIDTH-1:0] skip_out;
  logic signed [OUT_CH*OUT_H*OUT_W*WIDTH-1:0] ds_conv_out;
  logic signed [OUT_CH*OUT_H*OUT_W*WIDTH-1:0] ds_bn_out;
  logic signed [OUT_CH*OUT_H*OUT_W*WIDTH-1:0] add_out;

  integer i;
  integer fd;

  initial begin
    $readmemh("{input_mem.as_posix()}", in_mem);
    for (i = 0; i < IN_CH*IN_H*IN_W; i = i + 1) begin
      in_vec[i*WIDTH +: WIDTH] = in_mem[i];
    end
    #1;
    fd = $fopen("{output_mem.as_posix()}", "w");
    for (i = 0; i < OUT_CH*OUT_H*OUT_W; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  conv2d #(
    .IN_CH(IN_CH),
    .OUT_CH(OUT_CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(3),
    .STRIDE(STRIDE),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .WEIGHTS_FILE("{conv1_w_mem.as_posix()}")
  ) conv1 (
    .in_vec(in_vec),
    .out_vec(conv1_out)
  );

  batchnorm2d #(
    .CH(OUT_CH),
    .IN_H(OUT_H),
    .IN_W(OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .SCALE_FILE("{bn1_scale_mem.as_posix()}"),
    .BIAS_FILE("{bn1_bias_mem.as_posix()}")
  ) bn1 (
    .in_vec(conv1_out),
    .out_vec(bn1_out)
  );

  relu #(
    .DIM(OUT_CH*OUT_H*OUT_W),
    .WIDTH(WIDTH)
  ) relu1 (
    .in_vec(bn1_out),
    .out_vec(relu1_out)
  );

  conv2d #(
    .IN_CH(OUT_CH),
    .OUT_CH(OUT_CH),
    .IN_H(OUT_H),
    .IN_W(OUT_W),
    .K(3),
    .STRIDE(1),
    .PADDING(1),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .WEIGHTS_FILE("{conv2_w_mem.as_posix()}")
  ) conv2 (
    .in_vec(relu1_out),
    .out_vec(conv2_out)
  );

  batchnorm2d #(
    .CH(OUT_CH),
    .IN_H(OUT_H),
    .IN_W(OUT_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .SCALE_FILE("{bn2_scale_mem.as_posix()}"),
    .BIAS_FILE("{bn2_bias_mem.as_posix()}")
  ) bn2 (
    .in_vec(conv2_out),
    .out_vec(bn2_out)
  );

  generate
    if ({1 if ds_conv_mem else 0}) begin : gen_downsample
      conv2d #(
        .IN_CH(IN_CH),
        .OUT_CH(OUT_CH),
        .IN_H(IN_H),
        .IN_W(IN_W),
        .K(1),
        .STRIDE(STRIDE),
        .PADDING(0),
        .WIDTH(WIDTH),
        .FRAC(FRAC),
        .WEIGHTS_FILE("{ds_conv_mem.as_posix() if ds_conv_mem else ''}")
      ) ds_conv (
        .in_vec(in_vec),
        .out_vec(ds_conv_out)
      );

      batchnorm2d #(
        .CH(OUT_CH),
        .IN_H(OUT_H),
        .IN_W(OUT_W),
        .WIDTH(WIDTH),
        .FRAC(FRAC),
        .SCALE_FILE("{ds_bn_scale_mem.as_posix() if ds_bn_scale_mem else ''}"),
        .BIAS_FILE("{ds_bn_bias_mem.as_posix() if ds_bn_bias_mem else ''}")
      ) ds_bn (
        .in_vec(ds_conv_out),
        .out_vec(ds_bn_out)
      );
      assign skip_out = ds_bn_out;
    end else begin : gen_skip
      assign skip_out = in_vec;
    end
  endgenerate

  add_vec #(
    .DIM(OUT_CH*OUT_H*OUT_W),
    .WIDTH(WIDTH)
  ) add (
    .a_vec(bn2_out),
    .b_vec(skip_out),
    .out_vec(add_out)
  );

  relu #(
    .DIM(OUT_CH*OUT_H*OUT_W),
    .WIDTH(WIDTH)
  ) relu2 (
    .in_vec(add_out),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def _build_tb_avgpool_fc(
    tb_path: Path,
    input_mem: Path,
    fc_w_mem: Path,
    fc_b_mem: Path,
    output_mem: Path,
    ch: int,
    in_h: int,
    in_w: int,
) -> None:
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int FRAC = {FRAC};
  localparam int CH = {ch};
  localparam int IN_H = {in_h};
  localparam int IN_W = {in_w};
  localparam int OUT_H = (IN_H - IN_H) / 1 + 1;
  localparam int OUT_W = (IN_W - IN_W) / 1 + 1;

  logic signed [CH*IN_H*IN_W*WIDTH-1:0] in_vec;
  logic signed [CH*OUT_H*OUT_W*WIDTH-1:0] pool_out;
  logic signed [1000*WIDTH-1:0] out_vec;
  logic signed [WIDTH-1:0] in_mem [0:CH*IN_H*IN_W-1];

  integer i;
  integer fd;

  initial begin
    $readmemh("{input_mem.as_posix()}", in_mem);
    for (i = 0; i < CH*IN_H*IN_W; i = i + 1) begin
      in_vec[i*WIDTH +: WIDTH] = in_mem[i];
    end
    #1;
    fd = $fopen("{output_mem.as_posix()}", "w");
    for (i = 0; i < 1000; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  avgpool2d #(
    .CH(CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(IN_H),
    .STRIDE(1),
    .WIDTH(WIDTH)
  ) gap (
    .in_vec(in_vec),
    .out_vec(pool_out)
  );

  linear #(
    .IN_DIM(CH),
    .OUT_DIM(1000),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .WEIGHTS_FILE("{fc_w_mem.as_posix()}"),
    .BIAS_FILE("{fc_b_mem.as_posix()}")
  ) fc (
    .in_vec(pool_out),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def test_resnet18_layers() -> None:
    torch.manual_seed(22)
    resnet_path = REPO_ROOT / "networks" / "resnet18" / "resnet18.py"
    weights_path = REPO_ROOT / "networks" / "resnet18" / "resnet18_weights.py"
    resnet_mod = _load_module(resnet_path, "resnet18_module")
    if not weights_path.exists():
        resnet_mod.export_weights_py(weights_path, weights=None)

    weights_mod = _load_module(weights_path, "resnet18_weights")
    state = weights_mod.STATE_DICT

    def t(name: str) -> torch.Tensor:
        return torch.tensor(state[name], dtype=torch.float32)

    x_f = torch.rand((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32) * 2.0 - 1.0
    x_q = q88_from_float_tensor(x_f)

    conv1_w_q = q88_from_float_tensor(t("conv1.weight"))
    bn1_scale, bn1_bias = _bn_scale_bias(
        t("bn1.weight"),
        t("bn1.bias"),
        t("bn1.running_mean"),
        t("bn1.running_var"),
        1e-5,
    )
    bn1_scale_q = q88_from_float_tensor(bn1_scale)
    bn1_bias_q = q88_from_float_tensor(bn1_bias)

    def block_params(prefix: str):
        scale1, bias1 = _bn_scale_bias(
            t(f"{prefix}.bn1.weight"),
            t(f"{prefix}.bn1.bias"),
            t(f"{prefix}.bn1.running_mean"),
            t(f"{prefix}.bn1.running_var"),
            1e-5,
        )
        scale2, bias2 = _bn_scale_bias(
            t(f"{prefix}.bn2.weight"),
            t(f"{prefix}.bn2.bias"),
            t(f"{prefix}.bn2.running_mean"),
            t(f"{prefix}.bn2.running_var"),
            1e-5,
        )
        return {
            "conv1_w": q88_from_float_tensor(t(f"{prefix}.conv1.weight")),
            "bn1_scale": q88_from_float_tensor(scale1),
            "bn1_bias": q88_from_float_tensor(bias1),
            "conv2_w": q88_from_float_tensor(t(f"{prefix}.conv2.weight")),
            "bn2_scale": q88_from_float_tensor(scale2),
            "bn2_bias": q88_from_float_tensor(bias2),
        }

    def downsample_params(prefix: str):
        if f"{prefix}.0.weight" not in state:
            return None
        scale, bias = _bn_scale_bias(
            t(f"{prefix}.1.weight"),
            t(f"{prefix}.1.bias"),
            t(f"{prefix}.1.running_mean"),
            t(f"{prefix}.1.running_var"),
            1e-5,
        )
        return {
            "conv_w": q88_from_float_tensor(t(f"{prefix}.0.weight")),
            "bn_scale": q88_from_float_tensor(scale),
            "bn_bias": q88_from_float_tensor(bias),
        }

    blocks = [
        ("layer1.0", 64, 64, 1, None),
        ("layer1.1", 64, 64, 1, None),
        ("layer2.0", 64, 128, 2, downsample_params("layer2.0.downsample")),
        ("layer2.1", 128, 128, 1, None),
        ("layer3.0", 128, 256, 2, downsample_params("layer3.0.downsample")),
        ("layer3.1", 256, 256, 1, None),
        ("layer4.0", 256, 512, 2, downsample_params("layer4.0.downsample")),
        ("layer4.1", 512, 512, 1, None),
    ]

    build_dir = REPO_ROOT / "tests" / "build" / "resnet18_layers"
    build_dir.mkdir(parents=True, exist_ok=True)

    total_layers = 2 + len(blocks) + 1
    layer_idx = 1

    _progress(layer_idx, total_layers, "stem")
    layer_idx += 1
    stem_out = conv2d_q88(x_q, conv1_w_q, stride=2, padding=3)
    stem_out = bn_q88(stem_out, bn1_scale_q, bn1_bias_q)
    stem_out = relu_q88(stem_out)
    stem_out = maxpool2d_q88(stem_out, kernel=3, stride=2, padding=1)

    input_mem = build_dir / "stem_input.mem"
    conv_w_mem = build_dir / "stem_conv_w.mem"
    bn_scale_mem = build_dir / "stem_bn_scale.mem"
    bn_bias_mem = build_dir / "stem_bn_bias.mem"
    output_mem = build_dir / "stem_output.mem"
    tb_path = build_dir / "tb_stem.sv"

    write_mem(input_mem, x_q.squeeze(0).flatten().tolist())
    write_mem(conv_w_mem, conv1_w_q.flatten().tolist())
    write_mem(bn_scale_mem, bn1_scale_q.flatten().tolist())
    write_mem(bn_bias_mem, bn1_bias_q.flatten().tolist())
    _build_tb_stem(tb_path, input_mem, conv_w_mem, bn_scale_mem, bn_bias_mem, output_mem, INPUT_SIZE, INPUT_SIZE)
    run_verilator(
        tb_path,
        [
            REPO_ROOT / "modules" / "conv2d.sv",
            REPO_ROOT / "modules" / "batchnorm2d.sv",
            REPO_ROOT / "modules" / "relu.sv",
            REPO_ROOT / "modules" / "maxpool2d.sv",
        ],
        threads=THREADS,
    )
    hw_out = read_mem(output_mem)
    sw_out = stem_out.squeeze(0).flatten().tolist()
    _dump_values(build_dir / "stem_sw.txt", sw_out)
    _dump_values(build_dir / "stem_hw.txt", hw_out)
    print(f"stem SW (q) sample: {sw_out[:10]}")
    print(f"stem HW (q) sample: {hw_out[:10]}")
    if hw_out != sw_out:
        raise AssertionError("stem mismatch")

    x_q = stem_out
    h = stem_out.shape[2]
    w = stem_out.shape[3]

    for name, in_ch, out_ch, stride, ds in blocks:
        _progress(layer_idx, total_layers, name)
        layer_idx += 1
        params = block_params(name)

        input_x = x_q
        identity = x_q
        x_q = conv2d_q88(x_q, params["conv1_w"], stride=stride, padding=1)
        x_q = bn_q88(x_q, params["bn1_scale"], params["bn1_bias"])
        x_q = relu_q88(x_q)
        x_q = conv2d_q88(x_q, params["conv2_w"], stride=1, padding=1)
        x_q = bn_q88(x_q, params["bn2_scale"], params["bn2_bias"])
        if ds is not None:
            identity = conv2d_q88(identity, ds["conv_w"], stride=stride, padding=0)
            identity = bn_q88(identity, ds["bn_scale"], ds["bn_bias"])
        x_q = add_q88(x_q, identity)
        x_q = relu_q88(x_q)

        input_mem = build_dir / f"{name}_input.mem"
        conv1_w_mem = build_dir / f"{name}_conv1.mem"
        conv2_w_mem = build_dir / f"{name}_conv2.mem"
        bn1_scale_mem = build_dir / f"{name}_bn1_scale.mem"
        bn1_bias_mem = build_dir / f"{name}_bn1_bias.mem"
        bn2_scale_mem = build_dir / f"{name}_bn2_scale.mem"
        bn2_bias_mem = build_dir / f"{name}_bn2_bias.mem"
        output_mem = build_dir / f"{name}_output.mem"
        tb_path = build_dir / f"tb_{name.replace('.', '_')}.sv"

        write_mem(input_mem, input_x.squeeze(0).flatten().tolist())
        write_mem(conv1_w_mem, params["conv1_w"].flatten().tolist())
        write_mem(conv2_w_mem, params["conv2_w"].flatten().tolist())
        write_mem(bn1_scale_mem, params["bn1_scale"].flatten().tolist())
        write_mem(bn1_bias_mem, params["bn1_bias"].flatten().tolist())
        write_mem(bn2_scale_mem, params["bn2_scale"].flatten().tolist())
        write_mem(bn2_bias_mem, params["bn2_bias"].flatten().tolist())

        ds_conv_mem = ds_bn_scale_mem = ds_bn_bias_mem = None
        if ds is not None:
            ds_conv_mem = build_dir / f"{name}_ds_conv.mem"
            ds_bn_scale_mem = build_dir / f"{name}_ds_bn_scale.mem"
            ds_bn_bias_mem = build_dir / f"{name}_ds_bn_bias.mem"
            write_mem(ds_conv_mem, ds["conv_w"].flatten().tolist())
            write_mem(ds_bn_scale_mem, ds["bn_scale"].flatten().tolist())
            write_mem(ds_bn_bias_mem, ds["bn_bias"].flatten().tolist())

        _build_tb_block(
            tb_path,
            input_mem,
            conv1_w_mem,
            bn1_scale_mem,
            bn1_bias_mem,
            conv2_w_mem,
            bn2_scale_mem,
            bn2_bias_mem,
            output_mem,
            in_ch,
            out_ch,
            h,
            w,
            stride,
            ds_conv_mem,
            ds_bn_scale_mem,
            ds_bn_bias_mem,
        )

        run_verilator(
            tb_path,
            [
                REPO_ROOT / "modules" / "conv2d.sv",
                REPO_ROOT / "modules" / "batchnorm2d.sv",
                REPO_ROOT / "modules" / "relu.sv",
                REPO_ROOT / "modules" / "add_vec.sv",
            ],
            threads=THREADS,
        )
        hw_out = read_mem(output_mem)
        sw_out = x_q.squeeze(0).flatten().tolist()
        _dump_values(build_dir / f"{name}_sw.txt", sw_out)
        _dump_values(build_dir / f"{name}_hw.txt", hw_out)
        print(f"{name} SW (q) sample: {sw_out[:10]}")
        print(f"{name} HW (q) sample: {hw_out[:10]}")
        if hw_out != sw_out:
            raise AssertionError(f"{name} mismatch")

        h = x_q.shape[2]
        w = x_q.shape[3]

    _progress(layer_idx, total_layers, "avgpool+fc")
    layer_idx += 1
    pooled = avgpool2d_q88_int(x_q, kernel=h, stride=1)
    pooled_flat = pooled.view(1, -1)
    fc_w_q = q88_from_float_tensor(t("fc.weight"))
    fc_b_q = q88_from_float_tensor(t("fc.bias"))
    out_q = linear_q88(pooled_flat, fc_w_q, fc_b_q).squeeze(0)

    input_mem = build_dir / "gap_input.mem"
    fc_w_mem = build_dir / "fc_w.mem"
    fc_b_mem = build_dir / "fc_b.mem"
    output_mem = build_dir / "fc_output.mem"
    tb_path = build_dir / "tb_gap_fc.sv"
    write_mem(input_mem, x_q.squeeze(0).flatten().tolist())
    write_mem(fc_w_mem, fc_w_q.flatten().tolist())
    write_mem(fc_b_mem, fc_b_q.flatten().tolist())
    _build_tb_avgpool_fc(tb_path, input_mem, fc_w_mem, fc_b_mem, output_mem, 512, h, w)
    run_verilator(
        tb_path,
        [
            REPO_ROOT / "modules" / "avgpool2d.sv",
            REPO_ROOT / "modules" / "linear.sv",
        ],
        threads=THREADS,
    )
    hw_out = read_mem(output_mem)
    sw_out = out_q.tolist()
    _dump_values(build_dir / "fc_sw.txt", sw_out)
    _dump_values(build_dir / "fc_hw.txt", hw_out)
    print(f"fc SW (q) sample: {sw_out[:10]}")
    print(f"fc HW (q) sample: {hw_out[:10]}")
    if hw_out != sw_out:
        raise AssertionError("avgpool+fc mismatch")

    print("PASS")


if __name__ == "__main__":
    test_resnet18_layers()
