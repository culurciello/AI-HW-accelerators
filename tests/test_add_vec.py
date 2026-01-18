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
    a_mem: Path,
    b_mem: Path,
    output_mem: Path,
    dim: int,
) -> None:
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int DIM = {dim};

  logic signed [DIM*WIDTH-1:0] a_vec;
  logic signed [DIM*WIDTH-1:0] b_vec;
  logic signed [DIM*WIDTH-1:0] out_vec;
  logic signed [WIDTH-1:0] a_mem [0:DIM-1];
  logic signed [WIDTH-1:0] b_mem [0:DIM-1];

  integer i;
  integer fd;

  initial begin
    $readmemh("{a_mem.as_posix()}", a_mem);
    $readmemh("{b_mem.as_posix()}", b_mem);
    for (i = 0; i < DIM; i = i + 1) begin
      a_vec[i*WIDTH +: WIDTH] = a_mem[i];
      b_vec[i*WIDTH +: WIDTH] = b_mem[i];
    end
    #1;
    fd = $fopen("{output_mem.as_posix()}", "w");
    for (i = 0; i < DIM; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  add_vec #(
    .DIM(DIM),
    .WIDTH(WIDTH)
  ) dut (
    .a_vec(a_vec),
    .b_vec(b_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def test_add_vec() -> None:
    torch.manual_seed(15)
    dim = 16

    a_f = torch.rand((dim,), dtype=torch.float32) * 2.0 - 1.0
    b_f = torch.rand((dim,), dtype=torch.float32) * 2.0 - 1.0
    a_q = q88_from_float_tensor(a_f)
    b_q = q88_from_float_tensor(b_f)
    y_q = (a_q + b_q).to(torch.int16)

    build_dir = REPO_ROOT / "tests" / "build" / "add_vec"
    a_mem = build_dir / "a.mem"
    b_mem = build_dir / "b.mem"
    tb_path = build_dir / "tb_add_vec.sv"
    output_mem = build_dir / "output.mem"

    write_mem(a_mem, a_q.tolist())
    write_mem(b_mem, b_q.tolist())

    _build_tb(tb_path, a_mem, b_mem, output_mem, dim)

    sv_sources = [
        REPO_ROOT / "modules" / "add_vec.sv",
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
    test_add_vec()
