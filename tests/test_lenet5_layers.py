from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path

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
THREADS = int(os.environ.get("VERILATOR_THREADS", "16"))


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


def avgpool2d_q88_int(x_q: torch.Tensor) -> torch.Tensor:
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


def linear_q88_with_bias(x_q: torch.Tensor, w_q: torch.Tensor, b_q: torch.Tensor) -> torch.Tensor:
    x_f = x_q.to(torch.float32) / SCALE
    w_f = w_q.to(torch.float32) / SCALE
    b_f = b_q.to(torch.float32) / SCALE
    y_f = x_f @ w_f.t() + b_f
    return q88_from_float_tensor(y_f)


def _progress(idx: int, total: int, name: str) -> None:
    print(f"[{idx}/{total}] {name}", flush=True)


def _dump_values(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write(" ".join(str(v) for v in values))
        handle.write("\n")


def _build_tb_conv2d(
    tb_path: Path,
    input_mem: Path,
    weights_mem: Path,
    bias_mem: Path,
    output_mem: Path,
    in_ch: int,
    out_ch: int,
    in_h: int,
    in_w: int,
    k: int,
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
  localparam int K = {k};
  localparam int OUT_H = IN_H - K + 1;
  localparam int OUT_W = IN_W - K + 1;

  logic signed [IN_CH*IN_H*IN_W*WIDTH-1:0] in_vec;
  logic signed [OUT_CH*OUT_H*OUT_W*WIDTH-1:0] out_vec;
  logic signed [WIDTH-1:0] in_mem [0:IN_CH*IN_H*IN_W-1];

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
    .K(K),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .WEIGHTS_FILE("{weights_mem.as_posix()}"),
    .BIAS_FILE("{bias_mem.as_posix()}")
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def _build_tb_avgpool(
    tb_path: Path,
    input_mem: Path,
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
  localparam int CH = {ch};
  localparam int IN_H = {in_h};
  localparam int IN_W = {in_w};
  localparam int OUT_H = IN_H / 2;
  localparam int OUT_W = IN_W / 2;

  logic signed [CH*IN_H*IN_W*WIDTH-1:0] in_vec;
  logic signed [CH*OUT_H*OUT_W*WIDTH-1:0] out_vec;
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
    for (i = 0; i < CH*OUT_H*OUT_W; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  avgpool2d #(
    .CH(CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(2),
    .STRIDE(2),
    .WIDTH(WIDTH)
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def _build_tb_relu(
    tb_path: Path,
    input_mem: Path,
    output_mem: Path,
    dim: int,
) -> None:
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int FRAC = {FRAC};
  localparam int DIM = {dim};

  logic signed [DIM*WIDTH-1:0] in_vec;
  logic signed [DIM*WIDTH-1:0] out_vec;
  logic signed [WIDTH-1:0] in_mem [0:DIM-1];

  integer i;
  integer fd;

  initial begin
    $readmemh("{input_mem.as_posix()}", in_mem);
    for (i = 0; i < DIM; i = i + 1) begin
      in_vec[i*WIDTH +: WIDTH] = in_mem[i];
    end
    #1;
    fd = $fopen("{output_mem.as_posix()}", "w");
    for (i = 0; i < DIM; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  relu #(
    .DIM(DIM),
    .WIDTH(WIDTH),
    .precision("Q8.8")
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def _build_tb_linear(
    tb_path: Path,
    input_mem: Path,
    weights_mem: Path,
    bias_mem: Path,
    output_mem: Path,
    in_dim: int,
    out_dim: int,
) -> None:
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int FRAC = {FRAC};
  localparam int IN_DIM = {in_dim};
  localparam int OUT_DIM = {out_dim};

  logic signed [IN_DIM*WIDTH-1:0] in_vec;
  logic signed [OUT_DIM*WIDTH-1:0] out_vec;
  logic signed [WIDTH-1:0] in_mem [0:IN_DIM-1];

  integer i;
  integer fd;

  initial begin
    $readmemh("{input_mem.as_posix()}", in_mem);
    for (i = 0; i < IN_DIM; i = i + 1) begin
      in_vec[i*WIDTH +: WIDTH] = in_mem[i];
    end
    #1;
    fd = $fopen("{output_mem.as_posix()}", "w");
    for (i = 0; i < OUT_DIM; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  linear #(
    .IN_DIM(IN_DIM),
    .OUT_DIM(OUT_DIM),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .WEIGHTS_FILE("{weights_mem.as_posix()}"),
    .BIAS_FILE("{bias_mem.as_posix()}")
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def test_lenet5_layers() -> None:
    torch.manual_seed(21)
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

    x_f = torch.rand((1, 1, 32, 32), dtype=torch.float32) * 2.0 - 1.0
    x_q = q88_from_float_tensor(x_f)

    build_dir = REPO_ROOT / "tests" / "build" / "lenet5_layers"
    build_dir.mkdir(parents=True, exist_ok=True)

    total_layers = 11
    layer_idx = 1

    def run_conv(
        name: str,
        in_q: torch.Tensor,
        w_q: torch.Tensor,
        b_q: torch.Tensor,
        in_ch: int,
        out_ch: int,
        in_h: int,
        in_w: int,
    ):
        nonlocal layer_idx
        _progress(layer_idx, total_layers, name)
        layer_idx += 1
        out_q = conv2d_q88(in_q, w_q, b_q)
        out_h = in_h - 5 + 1
        out_w = in_w - 5 + 1

        input_mem = build_dir / f"{name}_input.mem"
        weights_mem = build_dir / f"{name}_weights.mem"
        bias_mem = build_dir / f"{name}_bias.mem"
        output_mem = build_dir / f"{name}_output.mem"
        tb_path = build_dir / f"tb_{name}.sv"
        write_mem(input_mem, in_q.squeeze(0).flatten().tolist())
        _build_tb_conv2d(
            tb_path,
            input_mem,
            weights_mem,
            bias_mem,
            output_mem,
            in_ch,
            1,
            in_h,
            in_w,
            5,
        )
        exe = build_verilator(
            tb_path,
            [REPO_ROOT / "modules" / "conv2d.sv"],
            threads=THREADS,
            clean=True,
        )

        hw_full = torch.zeros((out_ch, out_h, out_w), dtype=torch.int16)
        for oc in range(out_ch):
            write_mem(weights_mem, w_q[oc : oc + 1].flatten().tolist())
            write_mem(bias_mem, b_q[oc : oc + 1].flatten().tolist())
            run_verilator_exe(exe, tb_path.parent)
            hw_out = read_mem(output_mem)
            hw_full[oc] = torch.tensor(hw_out, dtype=torch.int16).view(out_h, out_w)

        sw_out = out_q.squeeze(0)
        sw_vals = sw_out.flatten().tolist()
        hw_vals = hw_full.flatten().tolist()
        _dump_values(build_dir / f"{name}_sw.txt", sw_vals)
        _dump_values(build_dir / f"{name}_hw.txt", hw_vals)
        print(f"{name} SW (q) sample: {sw_vals[:10]}")
        print(f"{name} HW (q) sample: {hw_vals[:10]}")
        if hw_full.tolist() != sw_out.tolist():
            raise AssertionError(f"{name} mismatch")
        return out_q

    def run_relu(name: str, in_q: torch.Tensor):
        nonlocal layer_idx
        _progress(layer_idx, total_layers, name)
        layer_idx += 1
        out_q = torch.clamp(in_q, min=0)
        dim = in_q.numel()
        input_mem = build_dir / f"{name}_input.mem"
        output_mem = build_dir / f"{name}_output.mem"
        tb_path = build_dir / f"tb_{name}.sv"
        write_mem(input_mem, in_q.view(-1).tolist())
        _build_tb_relu(tb_path, input_mem, output_mem, dim)
        run_verilator(
            tb_path,
            [REPO_ROOT / "modules" / "relu.sv"],
            threads=THREADS,
        )
        hw_out = read_mem(output_mem)
        sw_out = out_q.view(-1).tolist()
        _dump_values(build_dir / f"{name}_sw.txt", sw_out)
        _dump_values(build_dir / f"{name}_hw.txt", hw_out)
        print(f"{name} SW (q) sample: {sw_out[:10]}")
        print(f"{name} HW (q) sample: {hw_out[:10]}")
        if hw_out != sw_out:
            raise AssertionError(f"{name} mismatch")
        return out_q

    def run_avgpool(name: str, in_q: torch.Tensor, ch: int, in_h: int, in_w: int):
        nonlocal layer_idx
        _progress(layer_idx, total_layers, name)
        layer_idx += 1
        out_q = avgpool2d_q88_int(in_q)
        input_mem = build_dir / f"{name}_input.mem"
        output_mem = build_dir / f"{name}_output.mem"
        tb_path = build_dir / f"tb_{name}.sv"
        write_mem(input_mem, in_q.squeeze(0).flatten().tolist())
        _build_tb_avgpool(tb_path, input_mem, output_mem, ch, in_h, in_w)
        run_verilator(
            tb_path,
            [REPO_ROOT / "modules" / "avgpool2d.sv"],
            threads=THREADS,
        )
        hw_out = read_mem(output_mem)
        sw_out = out_q.squeeze(0).flatten().tolist()
        _dump_values(build_dir / f"{name}_sw.txt", sw_out)
        _dump_values(build_dir / f"{name}_hw.txt", hw_out)
        print(f"{name} SW (q) sample: {sw_out[:10]}")
        print(f"{name} HW (q) sample: {hw_out[:10]}")
        if hw_out != sw_out:
            raise AssertionError(f"{name} mismatch")
        return out_q

    def run_linear(name: str, in_q: torch.Tensor, w_q: torch.Tensor, b_q: torch.Tensor):
        nonlocal layer_idx
        _progress(layer_idx, total_layers, name)
        layer_idx += 1
        out_q = linear_q88_with_bias(in_q, w_q, b_q)
        input_mem = build_dir / f"{name}_input.mem"
        weights_mem = build_dir / f"{name}_weights.mem"
        bias_mem = build_dir / f"{name}_bias.mem"
        output_mem = build_dir / f"{name}_output.mem"
        tb_path = build_dir / f"tb_{name}.sv"
        write_mem(input_mem, in_q.squeeze(0).tolist())
        write_mem(weights_mem, w_q.flatten().tolist())
        write_mem(bias_mem, b_q.flatten().tolist())
        _build_tb_linear(tb_path, input_mem, weights_mem, bias_mem, output_mem, w_q.shape[1], w_q.shape[0])
        run_verilator(
            tb_path,
            [REPO_ROOT / "modules" / "linear.sv"],
            threads=THREADS,
        )
        hw_out = read_mem(output_mem)
        sw_out = out_q.squeeze(0).tolist()
        _dump_values(build_dir / f"{name}_sw.txt", sw_out)
        _dump_values(build_dir / f"{name}_hw.txt", hw_out)
        print(f"{name} SW (q) sample: {sw_out[:10]}")
        print(f"{name} HW (q) sample: {hw_out[:10]}")
        if hw_out != sw_out:
            raise AssertionError(f"{name} mismatch")
        return out_q

    x_q = run_conv("c1", x_q, c1_w, c1_b, 1, 6, 32, 32)
    x_q = run_relu("relu1", x_q)
    x_q = run_avgpool("s2", x_q, 6, 28, 28)
    x_q = run_conv("c3", x_q, c3_w, c3_b, 6, 16, 14, 14)
    x_q = run_relu("relu2", x_q)
    x_q = run_avgpool("s4", x_q, 16, 10, 10)
    x_q = x_q.view(1, -1)
    x_q = run_linear("c5", x_q, c5_w, c5_b)
    x_q = run_relu("relu3", x_q)
    x_q = run_linear("f6", x_q, f6_w, f6_b)
    x_q = run_relu("relu4", x_q)
    _ = run_linear("out", x_q, out_w, out_b)

    print("PASS")


if __name__ == "__main__":
    test_lenet5_layers()
