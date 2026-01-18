from __future__ import annotations

from pathlib import Path

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


def _build_tb(
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


def test_conv2d() -> None:
    torch.manual_seed(10)
    in_ch = 1
    out_ch = 2
    in_h = 6
    in_w = 6
    k = 3

    x_f = torch.rand((1, in_ch, in_h, in_w), dtype=torch.float32) * 2.0 - 1.0
    w_f = torch.rand((out_ch, in_ch, k, k), dtype=torch.float32) * 2.0 - 1.0
    b_f = torch.rand((out_ch,), dtype=torch.float32) * 2.0 - 1.0

    x_q = q88_from_float_tensor(x_f)
    w_q = q88_from_float_tensor(w_f)
    b_q = q88_from_float_tensor(b_f)

    x_ref = x_q.to(torch.float32) / (1 << FRAC)
    w_ref = w_q.to(torch.float32) / (1 << FRAC)
    b_ref = b_q.to(torch.float32) / (1 << FRAC)
    y_f = F.conv2d(x_ref, w_ref, bias=b_ref)
    y_q = q88_from_float_tensor(y_f).squeeze(0)

    build_dir = REPO_ROOT / "tests" / "build" / "conv2d"
    input_mem = build_dir / "input.mem"
    weights_mem = build_dir / "weights.mem"
    bias_mem = build_dir / "bias.mem"
    tb_path = build_dir / "tb_conv2d.sv"
    output_mem = build_dir / "output.mem"

    write_mem(input_mem, x_q.squeeze(0).flatten().tolist())
    write_mem(weights_mem, w_q.flatten().tolist())
    write_mem(bias_mem, b_q.flatten().tolist())

    _build_tb(
        tb_path,
        input_mem,
        weights_mem,
        bias_mem,
        output_mem,
        in_ch,
        out_ch,
        in_h,
        in_w,
        k,
    )

    sv_sources = [
        REPO_ROOT / "modules" / "conv2d.sv",
    ]
    run_verilator(tb_path, sv_sources)
    hw_out = read_mem(output_mem)
    sw_out = y_q.flatten().tolist()
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
    test_conv2d()
