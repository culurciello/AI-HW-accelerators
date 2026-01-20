# tests/test_mlp_c1_real.py uses the exact PyTorch model weights exported from networks/mlp/mlp.py.

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
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


def _load_module(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_tb(
    tb_path: Path,
    input_mem: Path,
    l1_weights: Path,
    l2_weights: Path,
    l3_weights: Path,
    output_mem: Path,
) -> None:
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int FRAC = {FRAC};

  logic signed [10*WIDTH-1:0] in_vec;
  logic signed [2*WIDTH-1:0] out_vec;
  logic signed [WIDTH-1:0] in_mem [0:9];

  integer i;
  integer fd;

  initial begin
    $readmemh("{input_mem.as_posix()}", in_mem);
    for (i = 0; i < 10; i = i + 1) begin
      in_vec[i*WIDTH +: WIDTH] = in_mem[i];
    end
    #1;
    fd = $fopen("{output_mem.as_posix()}", "w");
    for (i = 0; i < 2; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  mlp_c1 #(
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .L1_WEIGHTS_FILE("{l1_weights.as_posix()}"),
    .L2_WEIGHTS_FILE("{l2_weights.as_posix()}"),
    .L3_WEIGHTS_FILE("{l3_weights.as_posix()}")
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def test_mlp_c1_real() -> None:
    torch.manual_seed(3)
    in_dim = 10
    h_dim = 32
    out_dim = 2

    mlp_path = REPO_ROOT / "networks" / "mlp" / "mlp.py"
    pt_path = REPO_ROOT / "networks" / "mlp" / "mlp_c1.pt"
    mlp_mod = _load_module(mlp_path, "mlp_module")
    if pt_path.exists():
        state = torch.load(pt_path, map_location="cpu")
        model = mlp_mod.mlp_c1()
        model.load_state_dict(state)
    else:
        torch.manual_seed(0)
        model = mlp_mod.mlp_c1()
    l1_w_f = model.layer1.weight.detach().to(torch.float32)
    l2_w_f = model.layer2.weight.detach().to(torch.float32)
    l3_w_f = model.layer3.weight.detach().to(torch.float32)
    x_f = torch.rand((1, in_dim), dtype=torch.float32) * 2.0 - 1.0

    l1_w = q88_from_float_tensor(l1_w_f)
    l2_w = q88_from_float_tensor(l2_w_f)
    l3_w = q88_from_float_tensor(l3_w_f)
    x_q = q88_from_float_tensor(x_f)

    l1_q = linear_q88_torch(x_q, l1_w)
    l1_q = torch.clamp(l1_q, min=0)
    l2_q = linear_q88_torch(l1_q, l2_w)
    l2_q = torch.clamp(l2_q, min=0)
    out_q = linear_q88_torch(l2_q, l3_w).squeeze(0)

    build_dir = REPO_ROOT / "tests" / "build" / "mlp_c1_real"
    input_mem = build_dir / "input.mem"
    l1_mem = build_dir / "l1_weights.mem"
    l2_mem = build_dir / "l2_weights.mem"
    l3_mem = build_dir / "l3_weights.mem"
    tb_path = build_dir / "tb_mlp_c1_real.sv"
    output_mem = build_dir / "output.mem"

    write_mem(input_mem, x_q.squeeze(0).tolist())
    write_mem(l1_mem, l1_w.flatten().tolist())
    write_mem(l2_mem, l2_w.flatten().tolist())
    write_mem(l3_mem, l3_w.flatten().tolist())

    _build_tb(tb_path, input_mem, l1_mem, l2_mem, l3_mem, output_mem)

    sv_sources = [
        REPO_ROOT / "modules" / "linear.sv",
        REPO_ROOT / "modules" / "relu.sv",
        REPO_ROOT / "models" / "mlp_c1.sv",
    ]
    run_verilator(tb_path, sv_sources)
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
    test_mlp_c1_real()
