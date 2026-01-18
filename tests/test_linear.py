from __future__ import annotations

from pathlib import Path

import torch

from common import (
    FRAC,
    REPO_ROOT,
    WIDTH,
    linear_q88_torch,
    q88_from_float_tensor,
    read_mem,
    run_verilator,
    write_mem,
)


def _build_tb(
    tb_path: Path,
    input_mem: Path,
    weights_mem: Path,
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
  integer fd;

  integer i;

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
    .WEIGHTS_FILE("{weights_mem.as_posix()}")
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def test_linear() -> None:
    torch.manual_seed(0)
    in_dim = 10
    out_dim = 32

    weights_f = torch.rand((out_dim, in_dim), dtype=torch.float32) * 2.0 - 1.0
    inputs_f = torch.rand((1, in_dim), dtype=torch.float32) * 2.0 - 1.0
    weights_q = q88_from_float_tensor(weights_f)
    inputs_q = q88_from_float_tensor(inputs_f)

    expected_q = linear_q88_torch(inputs_q, weights_q).squeeze(0)

    build_dir = REPO_ROOT / "tests" / "build" / "linear"
    input_mem = build_dir / "input.mem"
    weights_mem = build_dir / "weights.mem"
    tb_path = build_dir / "tb_linear.sv"
    output_mem = build_dir / "output.mem"

    write_mem(input_mem, inputs_q.squeeze(0).tolist())
    write_mem(weights_mem, weights_q.flatten().tolist())

    _build_tb(
        tb_path,
        input_mem,
        weights_mem,
        output_mem,
        in_dim,
        out_dim,
    )

    sv_sources = [
        REPO_ROOT / "modules" / "linear.sv",
    ]
    run_verilator(tb_path, sv_sources)
    hw_out = read_mem(output_mem)
    sw_out = expected_q.tolist()
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
    test_linear()
