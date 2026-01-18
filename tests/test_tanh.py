from __future__ import annotations

from pathlib import Path

import torch

from common import (
    FRAC,
    REPO_ROOT,
    WIDTH,
    q88_from_float_tensor,
    read_mem,
    run_verilator,
    write_mem,
)


def _build_tb(
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

  tanh #(
    .DIM(DIM),
    .WIDTH(WIDTH),
    .FRAC(FRAC)
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def test_tanh() -> None:
    torch.manual_seed(12)
    dim = 16

    x_f = torch.rand((dim,), dtype=torch.float32) * 2.0 - 1.0
    x_q = q88_from_float_tensor(x_f)

    x_ref = x_q.to(torch.float32) / (1 << FRAC)
    y_f = torch.tanh(x_ref)
    y_q = q88_from_float_tensor(y_f)

    build_dir = REPO_ROOT / "tests" / "build" / "tanh"
    input_mem = build_dir / "input.mem"
    tb_path = build_dir / "tb_tanh.sv"
    output_mem = build_dir / "output.mem"

    write_mem(input_mem, x_q.tolist())

    _build_tb(tb_path, input_mem, output_mem, dim)

    sv_sources = [
        REPO_ROOT / "modules" / "tanh.sv",
    ]
    run_verilator(tb_path, sv_sources)
    hw_out = read_mem(output_mem)
    sw_out = y_q.tolist()
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
    test_tanh()
