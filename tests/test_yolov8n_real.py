from __future__ import annotations

from pathlib import Path
import os

from PIL import Image
import torch
import torchvision.transforms as transforms

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


def _build_tb_yolov8n(
    tb_path: Path,
    input_mem: Path,
    output_mem: Path,
    in_h: int,
    in_w: int,
    params: dict[str, str],
) -> None:
    assignments = ",\n    ".join(f".{name}(\"{path}\")" for name, path in params.items())
    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int FRAC = {FRAC};
  localparam int IN_H = {in_h};
  localparam int IN_W = {in_w};
  localparam int OUT_CH = 64 + 80;
  localparam int OUT_DIM = (IN_H/8)*(IN_W/8) + (IN_H/16)*(IN_W/16) + (IN_H/32)*(IN_W/32);

  logic signed [3*IN_H*IN_W*WIDTH-1:0] in_vec;
  logic signed [OUT_CH*OUT_DIM*WIDTH-1:0] out_vec;
  logic signed [WIDTH-1:0] in_mem [0:3*IN_H*IN_W-1];

  integer i;
  integer fd;

  initial begin
    $readmemh("{input_mem.as_posix()}", in_mem);
    for (i = 0; i < 3*IN_H*IN_W; i = i + 1) begin
      in_vec[i*WIDTH +: WIDTH] = in_mem[i];
    end
    #1;
    fd = $fopen("{output_mem.as_posix()}", "w");
    for (i = 0; i < OUT_CH*OUT_DIM; i = i + 1) begin
      $fdisplay(fd, "%0h", out_vec[i*WIDTH +: WIDTH]);
    end
    $fclose(fd);
    $finish;
  end

  yolov8n #(
    .WIDTH(WIDTH),
    .FRAC(FRAC),
    .IN_H(IN_H),
    .IN_W(IN_W),
    {assignments}
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")


def _run_sw(model, x_q: torch.Tensor) -> list[torch.Tensor]:
    layers = model.model.model
    outputs: list[torch.Tensor | list[torch.Tensor]] = []
    x = x_q

    for layer in layers:
        name = layer.__class__.__name__
        f = getattr(layer, "f", -1)
        if isinstance(f, int):
            inp = x if f == -1 else outputs[f]
        else:
            inp = [x if j == -1 else outputs[j] for j in f]

        if name == "Conv":
            out_q = yolo_conv_q88(inp, layer)
        elif name == "C2f":
            out_q = c2f_q88(inp, layer)
        elif name == "SPPF":
            out_q = sppf_q88(inp, layer)
        elif name == "Upsample":
            out_q = upsample2d_q88(inp, scale=2)
        elif name == "Concat":
            out_q = concat2d_q88(inp)
        elif name == "Detect":
            out_q = detect_q88(inp, layer)
        else:
            raise AssertionError(f"Unsupported layer type: {name}")

        outputs.append(out_q)
        x = out_q if not isinstance(out_q, list) else out_q[-1]

    if not isinstance(outputs[-1], list):
        raise AssertionError("Detect output missing")
    return outputs[-1]


def _collect_param_files(build_dir: Path, layers) -> dict[str, str]:
    weights_dir = build_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    params: dict[str, str] = {}

    def add_conv(prefix: str, conv) -> None:
        w, s, b = write_conv_bn_files(weights_dir, prefix.lower(), conv)
        params[f"{prefix}_W"] = w.as_posix()
        params[f"{prefix}_BN_S"] = s.as_posix()
        params[f"{prefix}_BN_B"] = b.as_posix()

    def add_c2f(prefix: str, c2f) -> None:
        cv1_w, cv1_s, cv1_b = write_conv_bn_files(weights_dir, f"{prefix.lower()}_cv1", c2f.cv1)
        cv2_w, cv2_s, cv2_b = write_conv_bn_files(weights_dir, f"{prefix.lower()}_cv2", c2f.cv2)
        bn0_cv1_w, bn0_cv1_s, bn0_cv1_b = write_conv_bn_files(weights_dir, f"{prefix.lower()}_bn0_cv1", c2f.m[0].cv1)
        bn0_cv2_w, bn0_cv2_s, bn0_cv2_b = write_conv_bn_files(weights_dir, f"{prefix.lower()}_bn0_cv2", c2f.m[0].cv2)

        params[f"{prefix}_CV1_W"] = cv1_w.as_posix()
        params[f"{prefix}_CV1_BN_S"] = cv1_s.as_posix()
        params[f"{prefix}_CV1_BN_B"] = cv1_b.as_posix()
        params[f"{prefix}_CV2_W"] = cv2_w.as_posix()
        params[f"{prefix}_CV2_BN_S"] = cv2_s.as_posix()
        params[f"{prefix}_CV2_BN_B"] = cv2_b.as_posix()
        params[f"{prefix}_BN0_CV1_W"] = bn0_cv1_w.as_posix()
        params[f"{prefix}_BN0_CV1_BN_S"] = bn0_cv1_s.as_posix()
        params[f"{prefix}_BN0_CV1_BN_B"] = bn0_cv1_b.as_posix()
        params[f"{prefix}_BN0_CV2_W"] = bn0_cv2_w.as_posix()
        params[f"{prefix}_BN0_CV2_BN_S"] = bn0_cv2_s.as_posix()
        params[f"{prefix}_BN0_CV2_BN_B"] = bn0_cv2_b.as_posix()

        if len(c2f.m) > 1:
            bn1_cv1_w, bn1_cv1_s, bn1_cv1_b = write_conv_bn_files(weights_dir, f"{prefix.lower()}_bn1_cv1", c2f.m[1].cv1)
            bn1_cv2_w, bn1_cv2_s, bn1_cv2_b = write_conv_bn_files(weights_dir, f"{prefix.lower()}_bn1_cv2", c2f.m[1].cv2)
            params[f"{prefix}_BN1_CV1_W"] = bn1_cv1_w.as_posix()
            params[f"{prefix}_BN1_CV1_BN_S"] = bn1_cv1_s.as_posix()
            params[f"{prefix}_BN1_CV1_BN_B"] = bn1_cv1_b.as_posix()
            params[f"{prefix}_BN1_CV2_W"] = bn1_cv2_w.as_posix()
            params[f"{prefix}_BN1_CV2_BN_S"] = bn1_cv2_s.as_posix()
            params[f"{prefix}_BN1_CV2_BN_B"] = bn1_cv2_b.as_posix()

    conv0 = layers[0]
    conv1 = layers[1]
    c2 = layers[2]
    conv3 = layers[3]
    c4 = layers[4]
    conv5 = layers[5]
    c6 = layers[6]
    conv7 = layers[7]
    c8 = layers[8]
    sppf = layers[9]
    c12 = layers[12]
    c15 = layers[15]
    conv16 = layers[16]
    c18 = layers[18]
    conv19 = layers[19]
    c21 = layers[21]
    detect = layers[22]

    add_conv("CV0", conv0)
    add_conv("CV1", conv1)
    add_c2f("C2", c2)
    add_conv("CV3", conv3)
    add_c2f("C4", c4)
    add_conv("CV5", conv5)
    add_c2f("C6", c6)
    add_conv("CV7", conv7)
    add_c2f("C8", c8)

    sppf_cv1_w, sppf_cv1_s, sppf_cv1_b = write_conv_bn_files(weights_dir, "sppf_cv1", sppf.cv1)
    sppf_cv2_w, sppf_cv2_s, sppf_cv2_b = write_conv_bn_files(weights_dir, "sppf_cv2", sppf.cv2)
    params["SPPF_CV1_W"] = sppf_cv1_w.as_posix()
    params["SPPF_CV1_BN_S"] = sppf_cv1_s.as_posix()
    params["SPPF_CV1_BN_B"] = sppf_cv1_b.as_posix()
    params["SPPF_CV2_W"] = sppf_cv2_w.as_posix()
    params["SPPF_CV2_BN_S"] = sppf_cv2_s.as_posix()
    params["SPPF_CV2_BN_B"] = sppf_cv2_b.as_posix()

    add_c2f("C12", c12)
    add_c2f("C15", c15)
    add_conv("CV16", conv16)
    add_c2f("C18", c18)
    add_conv("CV19", conv19)
    add_c2f("C21", c21)

    for scale_idx, tag in enumerate(["S1", "S2", "S3"]):
        r0_w, r0_s, r0_b = write_conv_bn_files(weights_dir, f"{tag.lower()}_r0", detect.cv2[scale_idx][0])
        r1_w, r1_s, r1_b = write_conv_bn_files(weights_dir, f"{tag.lower()}_r1", detect.cv2[scale_idx][1])
        r2_w, r2_b = write_conv_bias_files(weights_dir, f"{tag.lower()}_r2", detect.cv2[scale_idx][2])
        c0_w, c0_s, c0_b = write_conv_bn_files(weights_dir, f"{tag.lower()}_c0", detect.cv3[scale_idx][0])
        c1_w, c1_s, c1_b = write_conv_bn_files(weights_dir, f"{tag.lower()}_c1", detect.cv3[scale_idx][1])
        c2_w, c2_b = write_conv_bias_files(weights_dir, f"{tag.lower()}_c2", detect.cv3[scale_idx][2])
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

    return params


def test_yolov8n_real() -> None:
    if INPUT_SIZE % 32 != 0:
        raise AssertionError("YOLOV8N_INPUT must be divisible by 32")
    print("running SW pytorch version...")
    model_path = REPO_ROOT / "networks" / "yolov8n" / "yolov8n.pt"
    model = load_yolov8n(model_path)

    image_path = REPO_ROOT / "tests" / "images" / "cat.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
        ]
    )
    x_f = preprocess(image).unsqueeze(0)
    x_q = q88_from_float_tensor(x_f)

    out_scales = _run_sw(model, x_q)
    sw_out = flatten_chw(out_scales[0]) + flatten_chw(out_scales[1]) + flatten_chw(out_scales[2])

    results = model.predict(source=str(image_path), imgsz=INPUT_SIZE, conf=0.25, verbose=False)
    if results:
        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        print("Predictions:")
        for xyxy, cls_idx, conf in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.cls.cpu().tolist(),
            boxes.conf.cpu().tolist(),
        ):
            label = names.get(int(cls_idx), str(int(cls_idx)))
            print(f"  {label} conf={conf:.3f} bbox={xyxy}")

    build_dir = REPO_ROOT / "tests" / "build" / "yolov8n_real"
    build_dir.mkdir(parents=True, exist_ok=True)
    input_mem = build_dir / "input.mem"
    output_mem = build_dir / "output.mem"
    tb_path = build_dir / "tb_yolov8n_real.sv"
    write_mem(input_mem, flatten_chw(x_q))

    params = _collect_param_files(build_dir, model.model.model)
    _build_tb_yolov8n(tb_path, input_mem, output_mem, INPUT_SIZE, INPUT_SIZE, params)

    print("running SV hardware version...")
    sv_sources = [
        REPO_ROOT / "modules" / "conv2d.sv",
        REPO_ROOT / "modules" / "batchnorm2d.sv",
        REPO_ROOT / "modules" / "silu.sv",
        REPO_ROOT / "modules" / "maxpool2d.sv",
        REPO_ROOT / "modules" / "concat2d.sv",
        REPO_ROOT / "modules" / "upsample2d.sv",
        REPO_ROOT / "modules" / "yolo_conv.sv",
        REPO_ROOT / "modules" / "add_vec.sv",
        REPO_ROOT / "modules" / "bottleneck.sv",
        REPO_ROOT / "modules" / "c2f.sv",
        REPO_ROOT / "modules" / "sppf.sv",
        REPO_ROOT / "modules" / "detect.sv",
        REPO_ROOT / "models" / "yolov8n.sv",
    ]
    run_verilator(tb_path, sv_sources, threads=THREADS)

    hw_out = read_mem(output_mem)
    print(f"SW (q) sample: {sw_out[:10]}")
    print(f"HW (q) sample: {hw_out[:10]}")
    if hw_out != sw_out:
        raise AssertionError("yolov8n output mismatch")
    print("PASS")


if __name__ == "__main__":
    test_yolov8n_real()
