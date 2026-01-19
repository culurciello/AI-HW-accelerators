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
    build_verilator,
    q88_from_float_tensor,
    read_mem,
    run_verilator,
    run_verilator_exe,
    write_mem,
)

SCALE = 1 << FRAC


def _load_module(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def conv2d_q88(x_q: torch.Tensor, w_q: torch.Tensor, b_q: torch.Tensor | None) -> torch.Tensor:
    x_f = x_q.to(torch.float32) / SCALE
    w_f = w_q.to(torch.float32) / SCALE
    b_f = b_q.to(torch.float32) / SCALE if b_q is not None else None
    y_f = F.conv2d(x_f, w_f, bias=b_f)
    return q88_from_float_tensor(y_f)


def avgpool2d_q88(x_q: torch.Tensor) -> torch.Tensor:
    denom = 4
    n, ch, in_h, in_w = x_q.shape
    out_h = in_h // 2
    out_w = in_w // 2
    out = torch.zeros((n, ch, out_h, out_w), dtype=torch.int16)
    x_int = x_q.to(torch.int32)
    for c in range(ch):
        for oh in range(out_h):
            for ow in range(out_w):
                patch = x_int[0, c, oh * 2 : oh * 2 + 2, ow * 2 : ow * 2 + 2]
                acc = int(patch.sum().item())
                if acc < 0:
                    acc = -((-acc + denom // 2) // denom)
                else:
                    acc = (acc + denom // 2) // denom
                out[0, c, oh, ow] = acc
    return out


def relu_q88(x_q: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x_q, min=0)


def linear_q88_with_bias(x_q: torch.Tensor, w_q: torch.Tensor, b_q: torch.Tensor) -> torch.Tensor:
    x_f = x_q.to(torch.float32) / SCALE
    w_f = w_q.to(torch.float32) / SCALE
    b_f = b_q.to(torch.float32) / SCALE
    y_f = x_f @ w_f.t() + b_f
    return q88_from_float_tensor(y_f)


def _build_tb(
    tb_path: Path,
    input_mem: Path,
    c1_w: Path,
    c1_b: Path,
    c3_w: Path,
    c3_b: Path,
    c5_w: Path,
    c5_b: Path,
    f6_w: Path,
    f6_b: Path,
    out_w: Path,
    out_b: Path,
    output_mem: Path,
) -> None:
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int FRAC = {FRAC};
  localparam int IN_SIZE = 32*32;
  localparam int OUT_SIZE = 10;

  logic signed [IN_SIZE*WIDTH-1:0] in_vec;
  logic signed [OUT_SIZE*WIDTH-1:0] out_vec;
  logic signed [WIDTH-1:0] in_mem [0:IN_SIZE-1];

  integer i;
  integer fd;

  initial begin
    $readmemh("{input_mem.as_posix()}", in_mem);
    for (i = 0; i < IN_SIZE; i = i + 1) begin
      in_vec[i*WIDTH +: WIDTH] = in_mem[i];
    end
    #1;
    fd = $fopen("{output_mem.as_posix()}", "w");
    for (i = 0; i < OUT_SIZE; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  lenet5 #(
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .C1_WEIGHTS_FILE("{c1_w.as_posix()}"),
    .C1_BIAS_FILE("{c1_b.as_posix()}"),
    .C3_WEIGHTS_FILE("{c3_w.as_posix()}"),
    .C3_BIAS_FILE("{c3_b.as_posix()}"),
    .C5_WEIGHTS_FILE("{c5_w.as_posix()}"),
    .C5_BIAS_FILE("{c5_b.as_posix()}"),
    .F6_WEIGHTS_FILE("{f6_w.as_posix()}"),
    .F6_BIAS_FILE("{f6_b.as_posix()}"),
    .OUT_WEIGHTS_FILE("{out_w.as_posix()}"),
    .OUT_BIAS_FILE("{out_b.as_posix()}")
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def test_lenet5_real() -> None:
    torch.manual_seed(4)
    lenet_path = REPO_ROOT / "networks" / "lenet5" / "lenet5.py"
    weights_path = REPO_ROOT / "networks" / "lenet5" / "lenet5_weights.py"
    lenet_mod = _load_module(lenet_path, "lenet5_module")
    if not weights_path.exists():
        lenet_mod.export_weights_py(weights_path, seed=0, num_classes=10)

    weights_mod = _load_module(weights_path, "lenet5_weights")

    c1_w_f = torch.tensor(weights_mod.C1_W, dtype=torch.float32)
    c1_b_f = torch.tensor(weights_mod.C1_B, dtype=torch.float32)
    c3_w_f = torch.tensor(weights_mod.C3_W, dtype=torch.float32)
    c3_b_f = torch.tensor(weights_mod.C3_B, dtype=torch.float32)
    c5_w_f = torch.tensor(weights_mod.C5_W, dtype=torch.float32)
    c5_b_f = torch.tensor(weights_mod.C5_B, dtype=torch.float32)
    f6_w_f = torch.tensor(weights_mod.F6_W, dtype=torch.float32)
    f6_b_f = torch.tensor(weights_mod.F6_B, dtype=torch.float32)
    out_w_f = torch.tensor(weights_mod.OUT_W, dtype=torch.float32)
    out_b_f = torch.tensor(weights_mod.OUT_B, dtype=torch.float32)

    x_f = torch.rand((1, 1, 32, 32), dtype=torch.float32) * 2.0 - 1.0

    c1_w = q88_from_float_tensor(c1_w_f)
    c1_b = q88_from_float_tensor(c1_b_f)
    c3_w = q88_from_float_tensor(c3_w_f)
    c3_b = q88_from_float_tensor(c3_b_f)
    c5_w = q88_from_float_tensor(c5_w_f)
    c5_b = q88_from_float_tensor(c5_b_f)
    f6_w = q88_from_float_tensor(f6_w_f)
    f6_b = q88_from_float_tensor(f6_b_f)
    out_w = q88_from_float_tensor(out_w_f)
    out_b = q88_from_float_tensor(out_b_f)
    x_q = q88_from_float_tensor(x_f)

    x_q = relu_q88(conv2d_q88(x_q, c1_w, c1_b))
    x_q = avgpool2d_q88(x_q)
    x_q = relu_q88(conv2d_q88(x_q, c3_w, c3_b))
    x_q = avgpool2d_q88(x_q)
    x_q = x_q.view(1, -1)
    x_q = relu_q88(linear_q88_with_bias(x_q, c5_w, c5_b))
    x_q = relu_q88(linear_q88_with_bias(x_q, f6_w, f6_b))
    out_q = linear_q88_with_bias(x_q, out_w, out_b).squeeze(0)

    build_dir = REPO_ROOT / "tests" / "build" / "lenet5_real"
    input_mem = build_dir / "input.mem"
    c1_w_mem = build_dir / "c1_weights.mem"
    c1_b_mem = build_dir / "c1_bias.mem"
    c3_w_mem = build_dir / "c3_weights.mem"
    c3_b_mem = build_dir / "c3_bias.mem"
    c5_w_mem = build_dir / "c5_weights.mem"
    c5_b_mem = build_dir / "c5_bias.mem"
    f6_w_mem = build_dir / "f6_weights.mem"
    f6_b_mem = build_dir / "f6_bias.mem"
    out_w_mem = build_dir / "out_weights.mem"
    out_b_mem = build_dir / "out_bias.mem"
    tb_path = build_dir / "tb_lenet5_real.sv"
    output_mem = build_dir / "output.mem"

    write_mem(input_mem, x_q.flatten().tolist())
    write_mem(c1_w_mem, c1_w.flatten().tolist())
    write_mem(c1_b_mem, c1_b.flatten().tolist())
    write_mem(c3_w_mem, c3_w.flatten().tolist())
    write_mem(c3_b_mem, c3_b.flatten().tolist())
    write_mem(c5_w_mem, c5_w.flatten().tolist())
    write_mem(c5_b_mem, c5_b.flatten().tolist())
    write_mem(f6_w_mem, f6_w.flatten().tolist())
    write_mem(f6_b_mem, f6_b.flatten().tolist())
    write_mem(out_w_mem, out_w.flatten().tolist())
    write_mem(out_b_mem, out_b.flatten().tolist())

    sv_top = REPO_ROOT / "models" / "lenet5.sv"
    if not sv_top.exists():
        raise AssertionError("Missing SV top module: models/lenet5.sv")

    _build_tb(
        tb_path,
        input_mem,
        c1_w_mem,
        c1_b_mem,
        c3_w_mem,
        c3_b_mem,
        c5_w_mem,
        c5_b_mem,
        f6_w_mem,
        f6_b_mem,
        out_w_mem,
        out_b_mem,
        output_mem,
    )

    sv_sources = [
        REPO_ROOT / "modules" / "conv2d.sv",
        REPO_ROOT / "modules" / "avgpool2d.sv",
        REPO_ROOT / "modules" / "relu.sv",
        REPO_ROOT / "modules" / "linear.sv",
        sv_top,
    ]
    threads = int(os.environ.get("VERILATOR_THREADS", "16"))
    reuse = os.environ.get("VERILATOR_REUSE", "1") == "1"
    build_dir = tb_path.parent / "obj_dir"
    exe = build_dir / "Vtb"
    if not (reuse and exe.exists()):
        exe = build_verilator(tb_path, sv_sources, threads=threads, clean=not reuse)
    run_verilator_exe(exe, tb_path.parent)
    hw_out = read_mem(output_mem)
    sw_out = out_q.tolist()
    sw_float = [val / (1 << FRAC) for val in sw_out]
    hw_float = [val / (1 << FRAC) for val in hw_out]
    print(f"SW (q): {sw_out}")
    print(f"HW (q): {hw_out}")
    print(f"SW (f): {sw_float}")
    print(f"HW (f): {hw_float}")
    if hw_out != sw_out:
        raise AssertionError(f"Mismatch:\nHW: {hw_out}\nSW: {sw_out}")
    print("PASS")


if __name__ == "__main__":
    test_lenet5_real()
