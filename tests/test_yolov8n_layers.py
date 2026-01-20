from __future__ import annotations

from pathlib import Path
import os

import torch

from common import FRAC, REPO_ROOT, WIDTH, q88_from_float_tensor, read_mem, run_verilator, write_mem
from yolov8n_utils import (
    c2f_q88,
    concat2d_q88,
    detect_q88,
    flatten_chw,
    load_yolov8n,
    sppf_q88,
    upsample2d_q88,
    write_conv_bias_files,
    write_conv_bn_files,
    yolo_conv_q88,
)

THREADS = int(os.environ.get("VERILATOR_THREADS", "16"))
FAST_MODE = os.environ.get("YOLOV8N_FAST", "0") == "1"
INPUT_SIZE = int(os.environ.get("YOLOV8N_INPUT", "64" if FAST_MODE else "640"))


def _progress(idx: int, total: int, name: str) -> None:
    print(f"[{idx}/{total}] {name}", flush=True)


def _print_samples(name: str, sw_out: list[int], hw_out: list[int]) -> None:
    print(f"{name} SW (q) sample: {sw_out[:10]}")
    print(f"{name} HW (q) sample: {hw_out[:10]}")


def _build_tb_yolo_conv(
    tb_path: Path,
    input_mem: Path,
    weights_mem: Path,
    bn_scale_mem: Path,
    bn_bias_mem: Path,
    output_mem: Path,
    in_ch: int,
    out_ch: int,
    in_h: int,
    in_w: int,
    k: int,
    stride: int,
    padding: int,
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
  localparam int STRIDE = {stride};
  localparam int PADDING = {padding};
  localparam int OUT_H = (IN_H + 2*PADDING - K) / STRIDE + 1;
  localparam int OUT_W = (IN_W + 2*PADDING - K) / STRIDE + 1;

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

  yolo_conv #(
    .IN_CH(IN_CH),
    .OUT_CH(OUT_CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .K(K),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .WEIGHTS_FILE("{weights_mem.as_posix()}"),
    .BN_SCALE_FILE("{bn_scale_mem.as_posix()}"),
    .BN_BIAS_FILE("{bn_bias_mem.as_posix()}")
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def _build_tb_c2f(
    tb_path: Path,
    input_mem: Path,
    output_mem: Path,
    in_ch: int,
    out_ch: int,
    mid_ch: int,
    in_h: int,
    in_w: int,
    n: int,
    cv1_w: Path,
    cv1_s: Path,
    cv1_b: Path,
    cv2_w: Path,
    cv2_s: Path,
    cv2_b: Path,
    bn0_cv1_w: Path,
    bn0_cv1_s: Path,
    bn0_cv1_b: Path,
    bn0_cv2_w: Path,
    bn0_cv2_s: Path,
    bn0_cv2_b: Path,
    bn1_cv1_w: str,
    bn1_cv1_s: str,
    bn1_cv1_b: str,
    bn1_cv2_w: str,
    bn1_cv2_s: str,
    bn1_cv2_b: str,
) -> None:
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int FRAC = {FRAC};
  localparam int IN_CH = {in_ch};
  localparam int OUT_CH = {out_ch};
  localparam int MID_CH = {mid_ch};
  localparam int IN_H = {in_h};
  localparam int IN_W = {in_w};
  localparam int N = {n};

  logic signed [IN_CH*IN_H*IN_W*WIDTH-1:0] in_vec;
  logic signed [OUT_CH*IN_H*IN_W*WIDTH-1:0] out_vec;
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
    for (i = 0; i < OUT_CH*IN_H*IN_W; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  c2f #(
    .IN_CH(IN_CH),
    .OUT_CH(OUT_CH),
    .MID_CH(MID_CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .N(N),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .CV1_WEIGHTS_FILE("{cv1_w.as_posix()}"),
    .CV1_BN_SCALE_FILE("{cv1_s.as_posix()}"),
    .CV1_BN_BIAS_FILE("{cv1_b.as_posix()}"),
    .CV2_WEIGHTS_FILE("{cv2_w.as_posix()}"),
    .CV2_BN_SCALE_FILE("{cv2_s.as_posix()}"),
    .CV2_BN_BIAS_FILE("{cv2_b.as_posix()}"),
    .BN0_CV1_WEIGHTS_FILE("{bn0_cv1_w.as_posix()}"),
    .BN0_CV1_BN_SCALE_FILE("{bn0_cv1_s.as_posix()}"),
    .BN0_CV1_BN_BIAS_FILE("{bn0_cv1_b.as_posix()}"),
    .BN0_CV2_WEIGHTS_FILE("{bn0_cv2_w.as_posix()}"),
    .BN0_CV2_BN_SCALE_FILE("{bn0_cv2_s.as_posix()}"),
    .BN0_CV2_BN_BIAS_FILE("{bn0_cv2_b.as_posix()}"),
    .BN1_CV1_WEIGHTS_FILE("{bn1_cv1_w}"),
    .BN1_CV1_BN_SCALE_FILE("{bn1_cv1_s}"),
    .BN1_CV1_BN_BIAS_FILE("{bn1_cv1_b}"),
    .BN1_CV2_WEIGHTS_FILE("{bn1_cv2_w}"),
    .BN1_CV2_BN_SCALE_FILE("{bn1_cv2_s}"),
    .BN1_CV2_BN_BIAS_FILE("{bn1_cv2_b}")
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def _build_tb_sppf(
    tb_path: Path,
    input_mem: Path,
    output_mem: Path,
    in_ch: int,
    out_ch: int,
    mid_ch: int,
    in_h: int,
    in_w: int,
    cv1_w: Path,
    cv1_s: Path,
    cv1_b: Path,
    cv2_w: Path,
    cv2_s: Path,
    cv2_b: Path,
) -> None:
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int FRAC = {FRAC};
  localparam int IN_CH = {in_ch};
  localparam int OUT_CH = {out_ch};
  localparam int MID_CH = {mid_ch};
  localparam int IN_H = {in_h};
  localparam int IN_W = {in_w};

  logic signed [IN_CH*IN_H*IN_W*WIDTH-1:0] in_vec;
  logic signed [OUT_CH*IN_H*IN_W*WIDTH-1:0] out_vec;
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
    for (i = 0; i < OUT_CH*IN_H*IN_W; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  sppf #(
    .IN_CH(IN_CH),
    .OUT_CH(OUT_CH),
    .MID_CH(MID_CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .CV1_WEIGHTS_FILE("{cv1_w.as_posix()}"),
    .CV1_BN_SCALE_FILE("{cv1_s.as_posix()}"),
    .CV1_BN_BIAS_FILE("{cv1_b.as_posix()}"),
    .CV2_WEIGHTS_FILE("{cv2_w.as_posix()}"),
    .CV2_BN_SCALE_FILE("{cv2_s.as_posix()}"),
    .CV2_BN_BIAS_FILE("{cv2_b.as_posix()}")
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def _build_tb_concat(
    tb_path: Path,
    a_mem: Path,
    b_mem: Path,
    output_mem: Path,
    a_ch: int,
    b_ch: int,
    h: int,
    w: int,
) -> None:
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int A_CH = {a_ch};
  localparam int B_CH = {b_ch};
  localparam int H = {h};
  localparam int W = {w};

  logic signed [A_CH*H*W*WIDTH-1:0] a_vec;
  logic signed [B_CH*H*W*WIDTH-1:0] b_vec;
  logic signed [(A_CH+B_CH)*H*W*WIDTH-1:0] out_vec;
  logic signed [WIDTH-1:0] a_mem [0:A_CH*H*W-1];
  logic signed [WIDTH-1:0] b_mem [0:B_CH*H*W-1];

  integer i;
  integer fd;

  initial begin
    $readmemh("{a_mem.as_posix()}", a_mem);
    $readmemh("{b_mem.as_posix()}", b_mem);
    for (i = 0; i < A_CH*H*W; i = i + 1) begin
      a_vec[i*WIDTH +: WIDTH] = a_mem[i];
    end
    for (i = 0; i < B_CH*H*W; i = i + 1) begin
      b_vec[i*WIDTH +: WIDTH] = b_mem[i];
    end
    #1;
    fd = $fopen("{output_mem.as_posix()}", "w");
    for (i = 0; i < (A_CH+B_CH)*H*W; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  concat2d #(
    .A_CH(A_CH),
    .B_CH(B_CH),
    .IN_H(H),
    .IN_W(W),
    .WIDTH(WIDTH)
  ) dut (
    .a_vec(a_vec),
    .b_vec(b_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def _build_tb_upsample(
    tb_path: Path,
    input_mem: Path,
    output_mem: Path,
    ch: int,
    in_h: int,
    in_w: int,
    scale: int,
) -> None:
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int CH = {ch};
  localparam int IN_H = {in_h};
  localparam int IN_W = {in_w};
  localparam int SCALE = {scale};
  localparam int OUT_H = IN_H * SCALE;
  localparam int OUT_W = IN_W * SCALE;

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

  upsample2d #(
    .CH(CH),
    .IN_H(IN_H),
    .IN_W(IN_W),
    .SCALE(SCALE),
    .WIDTH(WIDTH)
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def _build_tb_detect(
    tb_path: Path,
    input1_mem: Path,
    input2_mem: Path,
    input3_mem: Path,
    output_mem: Path,
    in_ch1: int,
    in_h1: int,
    in_w1: int,
    in_ch2: int,
    in_h2: int,
    in_w2: int,
    in_ch3: int,
    in_h3: int,
    in_w3: int,
    params: dict[str, str],
) -> None:
    assignments = ",\n    ".join(f".{name}(\"{path}\")" for name, path in params.items())
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int FRAC = {FRAC};
  localparam int REG_CH = 64;
  localparam int CLS_CH = 80;
  localparam int IN_CH1 = {in_ch1};
  localparam int IN_H1 = {in_h1};
  localparam int IN_W1 = {in_w1};
  localparam int IN_CH2 = {in_ch2};
  localparam int IN_H2 = {in_h2};
  localparam int IN_W2 = {in_w2};
  localparam int IN_CH3 = {in_ch3};
  localparam int IN_H3 = {in_h3};
  localparam int IN_W3 = {in_w3};
  localparam int OUT_CH = REG_CH + CLS_CH;
  localparam int OUT_DIM = (IN_H1*IN_W1 + IN_H2*IN_W2 + IN_H3*IN_W3);

  logic signed [IN_CH1*IN_H1*IN_W1*WIDTH-1:0] in1;
  logic signed [IN_CH2*IN_H2*IN_W2*WIDTH-1:0] in2;
  logic signed [IN_CH3*IN_H3*IN_W3*WIDTH-1:0] in3;
  logic signed [OUT_CH*OUT_DIM*WIDTH-1:0] out_vec;
  logic signed [WIDTH-1:0] in1_mem [0:IN_CH1*IN_H1*IN_W1-1];
  logic signed [WIDTH-1:0] in2_mem [0:IN_CH2*IN_H2*IN_W2-1];
  logic signed [WIDTH-1:0] in3_mem [0:IN_CH3*IN_H3*IN_W3-1];

  integer i;
  integer fd;

  initial begin
    $readmemh("{input1_mem.as_posix()}", in1_mem);
    $readmemh("{input2_mem.as_posix()}", in2_mem);
    $readmemh("{input3_mem.as_posix()}", in3_mem);
    for (i = 0; i < IN_CH1*IN_H1*IN_W1; i = i + 1) begin
      in1[i*WIDTH +: WIDTH] = in1_mem[i];
    end
    for (i = 0; i < IN_CH2*IN_H2*IN_W2; i = i + 1) begin
      in2[i*WIDTH +: WIDTH] = in2_mem[i];
    end
    for (i = 0; i < IN_CH3*IN_H3*IN_W3; i = i + 1) begin
      in3[i*WIDTH +: WIDTH] = in3_mem[i];
    end
    #1;
    fd = $fopen("{output_mem.as_posix()}", "w");
    for (i = 0; i < OUT_CH*OUT_DIM; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  detect #(
    .IN_CH1(IN_CH1),
    .IN_H1(IN_H1),
    .IN_W1(IN_W1),
    .IN_CH2(IN_CH2),
    .IN_H2(IN_H2),
    .IN_W2(IN_W2),
    .IN_CH3(IN_CH3),
    .IN_H3(IN_H3),
    .IN_W3(IN_W3),
    .REG_CH(REG_CH),
    .CLS_CH(CLS_CH),
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    {assignments}
  ) dut (
    .in1(in1),
    .in2(in2),
    .in3(in3),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def test_yolov8n_layers() -> None:
    if INPUT_SIZE % 32 != 0:
        raise AssertionError("YOLOV8N_INPUT must be divisible by 32")
    model_path = REPO_ROOT / "networks" / "yolov8n" / "yolov8n.pt"
    model = load_yolov8n(model_path)
    layers = model.model.model
    total = len(layers)

    torch.manual_seed(30)
    x_f = torch.rand((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32) * 2.0 - 1.0
    x_q = q88_from_float_tensor(x_f)

    build_dir = REPO_ROOT / "tests" / "build" / "yolov8n_layers"
    build_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[torch.Tensor | list[torch.Tensor]] = []
    x = x_q

    for idx, layer in enumerate(layers, start=1):
        name = layer.__class__.__name__
        f = getattr(layer, "f", -1)
        if isinstance(f, int):
            inp = x if f == -1 else outputs[f]
        else:
            inp = [x if j == -1 else outputs[j] for j in f]

        _progress(idx, total, name)
        print("running SW pytorch version...", flush=True)

        if name == "Conv":
            out_q = yolo_conv_q88(inp, layer)
        elif name == "C2f":
            out_q = c2f_q88(inp, layer)
        elif name == "SPPF":
            out_q = sppf_q88(inp, layer)
        elif name == "Upsample":
            out_q = upsample2d_q88(inp, scale=2)
        elif name == "Concat":
            if not isinstance(inp, list) or len(inp) != 2:
                raise AssertionError("Concat expects two inputs")
            out_q = concat2d_q88(inp)
        elif name == "Detect":
            if not isinstance(inp, list) or len(inp) != 3:
                raise AssertionError("Detect expects three inputs")
            out_q = detect_q88(inp, layer)
        else:
            raise AssertionError(f"Unsupported layer type: {name}")

        print("running SV hardware version...", flush=True)
        layer_dir = build_dir / f"{idx:02d}_{name.lower()}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        if name == "Conv":
            input_mem = layer_dir / "input.mem"
            output_mem = layer_dir / "output.mem"
            weights_mem, bn_scale_mem, bn_bias_mem = write_conv_bn_files(layer_dir, "conv", layer)
            write_mem(input_mem, flatten_chw(inp))
            tb_path = layer_dir / "tb_conv.sv"
            in_ch = inp.shape[1]
            in_h = inp.shape[2]
            in_w = inp.shape[3]
            out_ch = out_q.shape[1]
            k = layer.conv.kernel_size[0]
            stride = layer.conv.stride[0]
            padding = layer.conv.padding[0]
            _build_tb_yolo_conv(
                tb_path,
                input_mem,
                weights_mem,
                bn_scale_mem,
                bn_bias_mem,
                output_mem,
                in_ch,
                out_ch,
                in_h,
                in_w,
                k,
                stride,
                padding,
            )
            run_verilator(
                tb_path,
                [
                    REPO_ROOT / "modules" / "conv2d.sv",
                    REPO_ROOT / "modules" / "batchnorm2d.sv",
                    REPO_ROOT / "modules" / "silu.sv",
                    REPO_ROOT / "modules" / "yolo_conv.sv",
                ],
                threads=THREADS,
            )
            hw_out = read_mem(output_mem)
            sw_out = flatten_chw(out_q)
        elif name == "C2f":
            input_mem = layer_dir / "input.mem"
            output_mem = layer_dir / "output.mem"
            write_mem(input_mem, flatten_chw(inp))
            cv1_w, cv1_s, cv1_b = write_conv_bn_files(layer_dir, "cv1", layer.cv1)
            cv2_w, cv2_s, cv2_b = write_conv_bn_files(layer_dir, "cv2", layer.cv2)
            bn0_cv1_w, bn0_cv1_s, bn0_cv1_b = write_conv_bn_files(layer_dir, "bn0_cv1", layer.m[0].cv1)
            bn0_cv2_w, bn0_cv2_s, bn0_cv2_b = write_conv_bn_files(layer_dir, "bn0_cv2", layer.m[0].cv2)
            if len(layer.m) > 1:
                bn1_cv1_w, bn1_cv1_s, bn1_cv1_b = write_conv_bn_files(layer_dir, "bn1_cv1", layer.m[1].cv1)
                bn1_cv2_w, bn1_cv2_s, bn1_cv2_b = write_conv_bn_files(layer_dir, "bn1_cv2", layer.m[1].cv2)
                bn1_vals = (
                    bn1_cv1_w.as_posix(),
                    bn1_cv1_s.as_posix(),
                    bn1_cv1_b.as_posix(),
                    bn1_cv2_w.as_posix(),
                    bn1_cv2_s.as_posix(),
                    bn1_cv2_b.as_posix(),
                )
            else:
                bn1_vals = ("", "", "", "", "", "")
            tb_path = layer_dir / "tb_c2f.sv"
            in_ch = inp.shape[1]
            in_h = inp.shape[2]
            in_w = inp.shape[3]
            out_ch = out_q.shape[1]
            mid_ch = layer.cv1.conv.out_channels // 2
            _build_tb_c2f(
                tb_path,
                input_mem,
                output_mem,
                in_ch,
                out_ch,
                mid_ch,
                in_h,
                in_w,
                len(layer.m),
                cv1_w,
                cv1_s,
                cv1_b,
                cv2_w,
                cv2_s,
                cv2_b,
                bn0_cv1_w,
                bn0_cv1_s,
                bn0_cv1_b,
                bn0_cv2_w,
                bn0_cv2_s,
                bn0_cv2_b,
                *bn1_vals,
            )
            run_verilator(
                tb_path,
                [
                    REPO_ROOT / "modules" / "conv2d.sv",
                    REPO_ROOT / "modules" / "batchnorm2d.sv",
                    REPO_ROOT / "modules" / "silu.sv",
                    REPO_ROOT / "modules" / "add_vec.sv",
                    REPO_ROOT / "modules" / "yolo_conv.sv",
                    REPO_ROOT / "modules" / "bottleneck.sv",
                    REPO_ROOT / "modules" / "c2f.sv",
                ],
                threads=THREADS,
            )
            hw_out = read_mem(output_mem)
            sw_out = flatten_chw(out_q)
        elif name == "SPPF":
            input_mem = layer_dir / "input.mem"
            output_mem = layer_dir / "output.mem"
            write_mem(input_mem, flatten_chw(inp))
            cv1_w, cv1_s, cv1_b = write_conv_bn_files(layer_dir, "cv1", layer.cv1)
            cv2_w, cv2_s, cv2_b = write_conv_bn_files(layer_dir, "cv2", layer.cv2)
            tb_path = layer_dir / "tb_sppf.sv"
            in_ch = inp.shape[1]
            in_h = inp.shape[2]
            in_w = inp.shape[3]
            out_ch = out_q.shape[1]
            mid_ch = layer.cv1.conv.out_channels
            _build_tb_sppf(
                tb_path,
                input_mem,
                output_mem,
                in_ch,
                out_ch,
                mid_ch,
                in_h,
                in_w,
                cv1_w,
                cv1_s,
                cv1_b,
                cv2_w,
                cv2_s,
                cv2_b,
            )
            run_verilator(
                tb_path,
                [
                    REPO_ROOT / "modules" / "conv2d.sv",
                    REPO_ROOT / "modules" / "batchnorm2d.sv",
                    REPO_ROOT / "modules" / "silu.sv",
                    REPO_ROOT / "modules" / "maxpool2d.sv",
                    REPO_ROOT / "modules" / "concat2d.sv",
                    REPO_ROOT / "modules" / "yolo_conv.sv",
                    REPO_ROOT / "modules" / "sppf.sv",
                ],
                threads=THREADS,
            )
            hw_out = read_mem(output_mem)
            sw_out = flatten_chw(out_q)
        elif name == "Upsample":
            input_mem = layer_dir / "input.mem"
            output_mem = layer_dir / "output.mem"
            write_mem(input_mem, flatten_chw(inp))
            tb_path = layer_dir / "tb_upsample.sv"
            ch = inp.shape[1]
            in_h = inp.shape[2]
            in_w = inp.shape[3]
            _build_tb_upsample(tb_path, input_mem, output_mem, ch, in_h, in_w, 2)
            run_verilator(
                tb_path,
                [REPO_ROOT / "modules" / "upsample2d.sv"],
                threads=THREADS,
            )
            hw_out = read_mem(output_mem)
            sw_out = flatten_chw(out_q)
        elif name == "Concat":
            input_a = inp[0]
            input_b = inp[1]
            a_mem = layer_dir / "a.mem"
            b_mem = layer_dir / "b.mem"
            output_mem = layer_dir / "output.mem"
            write_mem(a_mem, flatten_chw(input_a))
            write_mem(b_mem, flatten_chw(input_b))
            tb_path = layer_dir / "tb_concat.sv"
            _build_tb_concat(
                tb_path,
                a_mem,
                b_mem,
                output_mem,
                input_a.shape[1],
                input_b.shape[1],
                input_a.shape[2],
                input_a.shape[3],
            )
            run_verilator(
                tb_path,
                [REPO_ROOT / "modules" / "concat2d.sv"],
                threads=THREADS,
            )
            hw_out = read_mem(output_mem)
            sw_out = flatten_chw(out_q)
        elif name == "Detect":
            in1, in2, in3 = inp
            input1_mem = layer_dir / "input1.mem"
            input2_mem = layer_dir / "input2.mem"
            input3_mem = layer_dir / "input3.mem"
            output_mem = layer_dir / "output.mem"
            write_mem(input1_mem, flatten_chw(in1))
            write_mem(input2_mem, flatten_chw(in2))
            write_mem(input3_mem, flatten_chw(in3))

            params: dict[str, str] = {}
            for scale_idx, tag in enumerate(["S1", "S2", "S3"]):
                r0_w, r0_s, r0_b = write_conv_bn_files(layer_dir, f"{tag}_r0", layer.cv2[scale_idx][0])
                r1_w, r1_s, r1_b = write_conv_bn_files(layer_dir, f"{tag}_r1", layer.cv2[scale_idx][1])
                r2_w, r2_b = write_conv_bias_files(layer_dir, f"{tag}_r2", layer.cv2[scale_idx][2])
                c0_w, c0_s, c0_b = write_conv_bn_files(layer_dir, f"{tag}_c0", layer.cv3[scale_idx][0])
                c1_w, c1_s, c1_b = write_conv_bn_files(layer_dir, f"{tag}_c1", layer.cv3[scale_idx][1])
                c2_w, c2_b = write_conv_bias_files(layer_dir, f"{tag}_c2", layer.cv3[scale_idx][2])
                params.update(
                    {
                        f"D_{tag}_R0_W": r0_w.as_posix(),
                        f"D_{tag}_R0_BN_S": r0_s.as_posix(),
                        f"D_{tag}_R0_BN_B": r0_b.as_posix(),
                        f"D_{tag}_R1_W": r1_w.as_posix(),
                        f"D_{tag}_R1_BN_S": r1_s.as_posix(),
                        f"D_{tag}_R1_BN_B": r1_b.as_posix(),
                        f"D_{tag}_R2_W": r2_w.as_posix(),
                        f"D_{tag}_R2_B": r2_b.as_posix(),
                        f"D_{tag}_C0_W": c0_w.as_posix(),
                        f"D_{tag}_C0_BN_S": c0_s.as_posix(),
                        f"D_{tag}_C0_BN_B": c0_b.as_posix(),
                        f"D_{tag}_C1_W": c1_w.as_posix(),
                        f"D_{tag}_C1_BN_S": c1_s.as_posix(),
                        f"D_{tag}_C1_BN_B": c1_b.as_posix(),
                        f"D_{tag}_C2_W": c2_w.as_posix(),
                        f"D_{tag}_C2_B": c2_b.as_posix(),
                    }
                )

            tb_path = layer_dir / "tb_detect.sv"
            _build_tb_detect(
                tb_path,
                input1_mem,
                input2_mem,
                input3_mem,
                output_mem,
                in1.shape[1],
                in1.shape[2],
                in1.shape[3],
                in2.shape[1],
                in2.shape[2],
                in2.shape[3],
                in3.shape[1],
                in3.shape[2],
                in3.shape[3],
                params,
            )
            run_verilator(
                tb_path,
                [
                    REPO_ROOT / "modules" / "conv2d.sv",
                    REPO_ROOT / "modules" / "batchnorm2d.sv",
                    REPO_ROOT / "modules" / "silu.sv",
                    REPO_ROOT / "modules" / "yolo_conv.sv",
                    REPO_ROOT / "modules" / "detect.sv",
                ],
                threads=THREADS,
            )
            hw_out = read_mem(output_mem)
            out_scales = out_q
            sw_out = flatten_chw(out_scales[0]) + flatten_chw(out_scales[1]) + flatten_chw(out_scales[2])
        else:
            raise AssertionError(f"Unsupported layer type: {name}")

        _print_samples(name, sw_out, hw_out)
        if hw_out != sw_out:
            raise AssertionError(f"{name} mismatch")

        outputs.append(out_q)
        x = out_q if not isinstance(out_q, list) else out_q[-1]

    print("PASS")


if __name__ == "__main__":
    test_yolov8n_layers()
