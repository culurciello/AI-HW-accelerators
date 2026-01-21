from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os
import sys

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

REPO_ROOT = Path(__file__).resolve().parent
TESTS_DIR = REPO_ROOT / "tests"
sys.path.insert(0, str(TESTS_DIR))

from common import (  # noqa: E402
    FRAC,
    WIDTH,
    build_verilator,
    q88_from_float_tensor,
    read_mem,
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


def conv2d_q88(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    b_q: torch.Tensor | None,
    stride: int,
    padding: int,
) -> torch.Tensor:
    x_f = x_q.to(torch.float32) / SCALE
    w_f = w_q.to(torch.float32) / SCALE
    b_f = b_q.to(torch.float32) / SCALE if b_q is not None else None
    y_f = F.conv2d(x_f, w_f, bias=b_f, stride=stride, padding=padding)
    return q88_from_float_tensor(y_f)


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


def _write_bn_files(
    build_dir: Path,
    name: str,
    scale_q: torch.Tensor,
    bias_q: torch.Tensor,
):
    scale_mem = build_dir / f"{name}_scale.mem"
    bias_mem = build_dir / f"{name}_bias.mem"
    write_mem(scale_mem, scale_q.flatten().tolist())
    write_mem(bias_mem, bias_q.flatten().tolist())
    return scale_mem, bias_mem


def _load_weights():
    resnet_path = REPO_ROOT / "networks" / "resnet18" / "resnet18.py"
    weights_path = REPO_ROOT / "networks" / "resnet18" / "resnet18_weights.py"
    resnet_mod = _load_module(resnet_path, "resnet18_module")
    if not weights_path.exists():
        resnet_mod.export_weights_py(weights_path, weights="DEFAULT")

    state = None
    pt_path = REPO_ROOT / "networks" / "resnet18" / "resnet18_small.pt"
    if pt_path.exists():
        state = torch.load(pt_path, map_location="cpu")
    elif weights_path.exists():
        try:
            weights_mod = _load_module(weights_path, "resnet18_weights")
            state = weights_mod.STATE_DICT
        except SyntaxError:
            state = None

    if state is None:
        model = models.resnet18(weights=None)
        state = model.state_dict()

    return state


def _load_image_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess(image).unsqueeze(0)


def run_demo() -> None:
    print("running SW pytorch version...")
    torch.manual_seed(16)
    state = _load_weights()

    def t(name: str) -> torch.Tensor:
        return torch.tensor(state[name], dtype=torch.float32)

    image_path = REPO_ROOT / "tests" / "images" / "cat.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    input_f = _load_image_tensor(image_path)
    input_q = q88_from_float_tensor(input_f)

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
        return {
            "conv1_w": q88_from_float_tensor(t(f"{prefix}.conv1.weight")),
            "bn1_scale": q88_from_float_tensor(
                _bn_scale_bias(
                    t(f"{prefix}.bn1.weight"),
                    t(f"{prefix}.bn1.bias"),
                    t(f"{prefix}.bn1.running_mean"),
                    t(f"{prefix}.bn1.running_var"),
                    1e-5,
                )[0]
            ),
            "bn1_bias": q88_from_float_tensor(
                _bn_scale_bias(
                    t(f"{prefix}.bn1.weight"),
                    t(f"{prefix}.bn1.bias"),
                    t(f"{prefix}.bn1.running_mean"),
                    t(f"{prefix}.bn1.running_var"),
                    1e-5,
                )[1]
            ),
            "conv2_w": q88_from_float_tensor(t(f"{prefix}.conv2.weight")),
            "bn2_scale": q88_from_float_tensor(
                _bn_scale_bias(
                    t(f"{prefix}.bn2.weight"),
                    t(f"{prefix}.bn2.bias"),
                    t(f"{prefix}.bn2.running_mean"),
                    t(f"{prefix}.bn2.running_var"),
                    1e-5,
                )[0]
            ),
            "bn2_bias": q88_from_float_tensor(
                _bn_scale_bias(
                    t(f"{prefix}.bn2.weight"),
                    t(f"{prefix}.bn2.bias"),
                    t(f"{prefix}.bn2.running_mean"),
                    t(f"{prefix}.bn2.running_var"),
                    1e-5,
                )[1]
            ),
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

    l1_0 = block_params("layer1.0")
    l1_1 = block_params("layer1.1")
    l2_0 = block_params("layer2.0")
    l2_1 = block_params("layer2.1")
    l3_0 = block_params("layer3.0")
    l3_1 = block_params("layer3.1")
    l4_0 = block_params("layer4.0")
    l4_1 = block_params("layer4.1")

    l2_0_ds = downsample_params("layer2.0.downsample")
    l3_0_ds = downsample_params("layer3.0.downsample")
    l4_0_ds = downsample_params("layer4.0.downsample")

    fc_w_q = q88_from_float_tensor(t("fc.weight"))
    fc_b_q = q88_from_float_tensor(t("fc.bias"))

    x_q = conv2d_q88(input_q, conv1_w_q, None, stride=2, padding=3)
    x_q = bn_q88(x_q, bn1_scale_q, bn1_bias_q)
    x_q = relu_q88(x_q)
    x_q = maxpool2d_q88(x_q, kernel=3, stride=2, padding=1)

    def basic_block(x_q: torch.Tensor, params, stride: int = 1, downsample=None):
        identity = x_q
        x_q = conv2d_q88(x_q, params["conv1_w"], None, stride=stride, padding=1)
        x_q = bn_q88(x_q, params["bn1_scale"], params["bn1_bias"])
        x_q = relu_q88(x_q)
        x_q = conv2d_q88(x_q, params["conv2_w"], None, stride=1, padding=1)
        x_q = bn_q88(x_q, params["bn2_scale"], params["bn2_bias"])
        if downsample is not None:
            identity = conv2d_q88(identity, downsample["conv_w"], None, stride=stride, padding=0)
            identity = bn_q88(identity, downsample["bn_scale"], downsample["bn_bias"])
        x_q = add_q88(x_q, identity)
        x_q = relu_q88(x_q)
        return x_q

    x_q = basic_block(x_q, l1_0, stride=1, downsample=None)
    x_q = basic_block(x_q, l1_1, stride=1, downsample=None)
    x_q = basic_block(x_q, l2_0, stride=2, downsample=l2_0_ds)
    x_q = basic_block(x_q, l2_1, stride=1, downsample=None)
    x_q = basic_block(x_q, l3_0, stride=2, downsample=l3_0_ds)
    x_q = basic_block(x_q, l3_1, stride=1, downsample=None)
    x_q = basic_block(x_q, l4_0, stride=2, downsample=l4_0_ds)
    x_q = basic_block(x_q, l4_1, stride=1, downsample=None)

    x_q = avgpool2d_q88_int(x_q, kernel=7, stride=1)
    x_q = x_q.view(1, -1)
    out_q = linear_q88(x_q, fc_w_q, fc_b_q).squeeze(0)

    build_dir = REPO_ROOT / "tests" / "build" / "resnet18_demo"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_mem = build_dir / "input.mem"
    conv1_mem = build_dir / "conv1_weights.mem"
    write_mem(input_mem, input_q.squeeze(0).flatten().tolist())
    write_mem(conv1_mem, conv1_w_q.flatten().tolist())

    bn1_scale_mem, bn1_bias_mem = _write_bn_files(build_dir, "bn1", bn1_scale_q, bn1_bias_q)

    def write_block(name: str, params, downsample):
        write_mem(build_dir / f"{name}_conv1.mem", params["conv1_w"].flatten().tolist())
        write_mem(build_dir / f"{name}_conv2.mem", params["conv2_w"].flatten().tolist())
        s1_mem, b1_mem = _write_bn_files(build_dir, f"{name}_bn1", params["bn1_scale"], params["bn1_bias"])
        s2_mem, b2_mem = _write_bn_files(build_dir, f"{name}_bn2", params["bn2_scale"], params["bn2_bias"])
        ds_conv_mem = ds_s_mem = ds_b_mem = None
        if downsample is not None:
            ds_conv_mem = build_dir / f"{name}_ds_conv.mem"
            write_mem(ds_conv_mem, downsample["conv_w"].flatten().tolist())
            ds_s_mem, ds_b_mem = _write_bn_files(
                build_dir, f"{name}_ds_bn", downsample["bn_scale"], downsample["bn_bias"]
            )
        return s1_mem, b1_mem, s2_mem, b2_mem, ds_conv_mem, ds_s_mem, ds_b_mem

    l1_0_bn1_scale_mem, l1_0_bn1_bias_mem, l1_0_bn2_scale_mem, l1_0_bn2_bias_mem, _, _, _ = write_block(
        "l1_0", l1_0, None
    )
    l1_1_bn1_scale_mem, l1_1_bn1_bias_mem, l1_1_bn2_scale_mem, l1_1_bn2_bias_mem, _, _, _ = write_block(
        "l1_1", l1_1, None
    )
    (
        l2_0_bn1_scale_mem,
        l2_0_bn1_bias_mem,
        l2_0_bn2_scale_mem,
        l2_0_bn2_bias_mem,
        l2_0_ds_conv_mem,
        l2_0_ds_bn_scale_mem,
        l2_0_ds_bn_bias_mem,
    ) = write_block("l2_0", l2_0, l2_0_ds)
    l2_1_bn1_scale_mem, l2_1_bn1_bias_mem, l2_1_bn2_scale_mem, l2_1_bn2_bias_mem, _, _, _ = write_block(
        "l2_1", l2_1, None
    )
    (
        l3_0_bn1_scale_mem,
        l3_0_bn1_bias_mem,
        l3_0_bn2_scale_mem,
        l3_0_bn2_bias_mem,
        l3_0_ds_conv_mem,
        l3_0_ds_bn_scale_mem,
        l3_0_ds_bn_bias_mem,
    ) = write_block("l3_0", l3_0, l3_0_ds)
    l3_1_bn1_scale_mem, l3_1_bn1_bias_mem, l3_1_bn2_scale_mem, l3_1_bn2_bias_mem, _, _, _ = write_block(
        "l3_1", l3_1, None
    )
    (
        l4_0_bn1_scale_mem,
        l4_0_bn1_bias_mem,
        l4_0_bn2_scale_mem,
        l4_0_bn2_bias_mem,
        l4_0_ds_conv_mem,
        l4_0_ds_bn_scale_mem,
        l4_0_ds_bn_bias_mem,
    ) = write_block("l4_0", l4_0, l4_0_ds)
    l4_1_bn1_scale_mem, l4_1_bn1_bias_mem, l4_1_bn2_scale_mem, l4_1_bn2_bias_mem, _, _, _ = write_block(
        "l4_1", l4_1, None
    )

    fc_w_mem = build_dir / "fc_weights.mem"
    fc_b_mem = build_dir / "fc_bias.mem"
    write_mem(fc_w_mem, fc_w_q.flatten().tolist())
    write_mem(fc_b_mem, fc_b_q.flatten().tolist())

    tb_path = build_dir / "tb_resnet18_demo.sv"
    output_mem = build_dir / "output.mem"

    tb_text = f"""`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
module tb;
/* verilator lint_on DECLFILENAME */
  localparam int WIDTH = {WIDTH};
  localparam int IN_SIZE = 3*224*224;
  localparam int OUT_SIZE = 1000;

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

  resnet18 #(
    .WIDTH(WIDTH),
    .FRAC({FRAC}),
    .CONV1_WEIGHTS_FILE("{conv1_mem.as_posix()}"),
    .BN1_SCALE_FILE("{bn1_scale_mem.as_posix()}"),
    .BN1_BIAS_FILE("{bn1_bias_mem.as_posix()}"),
    .L1_0_CONV1_WEIGHTS_FILE("{(build_dir / 'l1_0_conv1.mem').as_posix()}"),
    .L1_0_BN1_SCALE_FILE("{l1_0_bn1_scale_mem.as_posix()}"),
    .L1_0_BN1_BIAS_FILE("{l1_0_bn1_bias_mem.as_posix()}"),
    .L1_0_CONV2_WEIGHTS_FILE("{(build_dir / 'l1_0_conv2.mem').as_posix()}"),
    .L1_0_BN2_SCALE_FILE("{l1_0_bn2_scale_mem.as_posix()}"),
    .L1_0_BN2_BIAS_FILE("{l1_0_bn2_bias_mem.as_posix()}"),
    .L1_1_CONV1_WEIGHTS_FILE("{(build_dir / 'l1_1_conv1.mem').as_posix()}"),
    .L1_1_BN1_SCALE_FILE("{l1_1_bn1_scale_mem.as_posix()}"),
    .L1_1_BN1_BIAS_FILE("{l1_1_bn1_bias_mem.as_posix()}"),
    .L1_1_CONV2_WEIGHTS_FILE("{(build_dir / 'l1_1_conv2.mem').as_posix()}"),
    .L1_1_BN2_SCALE_FILE("{l1_1_bn2_scale_mem.as_posix()}"),
    .L1_1_BN2_BIAS_FILE("{l1_1_bn2_bias_mem.as_posix()}"),
    .L2_0_CONV1_WEIGHTS_FILE("{(build_dir / 'l2_0_conv1.mem').as_posix()}"),
    .L2_0_BN1_SCALE_FILE("{l2_0_bn1_scale_mem.as_posix()}"),
    .L2_0_BN1_BIAS_FILE("{l2_0_bn1_bias_mem.as_posix()}"),
    .L2_0_CONV2_WEIGHTS_FILE("{(build_dir / 'l2_0_conv2.mem').as_posix()}"),
    .L2_0_BN2_SCALE_FILE("{l2_0_bn2_scale_mem.as_posix()}"),
    .L2_0_BN2_BIAS_FILE("{l2_0_bn2_bias_mem.as_posix()}"),
    .L2_0_DS_CONV_WEIGHTS_FILE("{l2_0_ds_conv_mem.as_posix()}"),
    .L2_0_DS_BN_SCALE_FILE("{l2_0_ds_bn_scale_mem.as_posix()}"),
    .L2_0_DS_BN_BIAS_FILE("{l2_0_ds_bn_bias_mem.as_posix()}"),
    .L2_1_CONV1_WEIGHTS_FILE("{(build_dir / 'l2_1_conv1.mem').as_posix()}"),
    .L2_1_BN1_SCALE_FILE("{l2_1_bn1_scale_mem.as_posix()}"),
    .L2_1_BN1_BIAS_FILE("{l2_1_bn1_bias_mem.as_posix()}"),
    .L2_1_CONV2_WEIGHTS_FILE("{(build_dir / 'l2_1_conv2.mem').as_posix()}"),
    .L2_1_BN2_SCALE_FILE("{l2_1_bn2_scale_mem.as_posix()}"),
    .L2_1_BN2_BIAS_FILE("{l2_1_bn2_bias_mem.as_posix()}"),
    .L3_0_CONV1_WEIGHTS_FILE("{(build_dir / 'l3_0_conv1.mem').as_posix()}"),
    .L3_0_BN1_SCALE_FILE("{l3_0_bn1_scale_mem.as_posix()}"),
    .L3_0_BN1_BIAS_FILE("{l3_0_bn1_bias_mem.as_posix()}"),
    .L3_0_CONV2_WEIGHTS_FILE("{(build_dir / 'l3_0_conv2.mem').as_posix()}"),
    .L3_0_BN2_SCALE_FILE("{l3_0_bn2_scale_mem.as_posix()}"),
    .L3_0_BN2_BIAS_FILE("{l3_0_bn2_bias_mem.as_posix()}"),
    .L3_0_DS_CONV_WEIGHTS_FILE("{l3_0_ds_conv_mem.as_posix()}"),
    .L3_0_DS_BN_SCALE_FILE("{l3_0_ds_bn_scale_mem.as_posix()}"),
    .L3_0_DS_BN_BIAS_FILE("{l3_0_ds_bn_bias_mem.as_posix()}"),
    .L3_1_CONV1_WEIGHTS_FILE("{(build_dir / 'l3_1_conv1.mem').as_posix()}"),
    .L3_1_BN1_SCALE_FILE("{l3_1_bn1_scale_mem.as_posix()}"),
    .L3_1_BN1_BIAS_FILE("{l3_1_bn1_bias_mem.as_posix()}"),
    .L3_1_CONV2_WEIGHTS_FILE("{(build_dir / 'l3_1_conv2.mem').as_posix()}"),
    .L3_1_BN2_SCALE_FILE("{l3_1_bn2_scale_mem.as_posix()}"),
    .L3_1_BN2_BIAS_FILE("{l3_1_bn2_bias_mem.as_posix()}"),
    .L4_0_CONV1_WEIGHTS_FILE("{(build_dir / 'l4_0_conv1.mem').as_posix()}"),
    .L4_0_BN1_SCALE_FILE("{l4_0_bn1_scale_mem.as_posix()}"),
    .L4_0_BN1_BIAS_FILE("{l4_0_bn1_bias_mem.as_posix()}"),
    .L4_0_CONV2_WEIGHTS_FILE("{(build_dir / 'l4_0_conv2.mem').as_posix()}"),
    .L4_0_BN2_SCALE_FILE("{l4_0_bn2_scale_mem.as_posix()}"),
    .L4_0_BN2_BIAS_FILE("{l4_0_bn2_bias_mem.as_posix()}"),
    .L4_0_DS_CONV_WEIGHTS_FILE("{l4_0_ds_conv_mem.as_posix()}"),
    .L4_0_DS_BN_SCALE_FILE("{l4_0_ds_bn_scale_mem.as_posix()}"),
    .L4_0_DS_BN_BIAS_FILE("{l4_0_ds_bn_bias_mem.as_posix()}"),
    .L4_1_CONV1_WEIGHTS_FILE("{(build_dir / 'l4_1_conv1.mem').as_posix()}"),
    .L4_1_BN1_SCALE_FILE("{l4_1_bn1_scale_mem.as_posix()}"),
    .L4_1_BN1_BIAS_FILE("{l4_1_bn1_bias_mem.as_posix()}"),
    .L4_1_CONV2_WEIGHTS_FILE("{(build_dir / 'l4_1_conv2.mem').as_posix()}"),
    .L4_1_BN2_SCALE_FILE("{l4_1_bn2_scale_mem.as_posix()}"),
    .L4_1_BN2_BIAS_FILE("{l4_1_bn2_bias_mem.as_posix()}"),
    .FC_WEIGHTS_FILE("{fc_w_mem.as_posix()}"),
    .FC_BIAS_FILE("{fc_b_mem.as_posix()}")
  ) dut (
    .in_vec(in_vec),
    .out_vec(out_vec)
  );
endmodule
"""
    tb_path.write_text(tb_text, encoding="ascii")

    sv_sources = [
        REPO_ROOT / "modules" / "conv2d.sv",
        REPO_ROOT / "modules" / "batchnorm2d.sv",
        REPO_ROOT / "modules" / "relu.sv",
        REPO_ROOT / "modules" / "maxpool2d.sv",
        REPO_ROOT / "modules" / "add_vec.sv",
        REPO_ROOT / "modules" / "avgpool2d.sv",
        REPO_ROOT / "modules" / "linear.sv",
        REPO_ROOT / "models" / "resnet18.sv",
    ]
    threads = int(os.environ.get("VERILATOR_THREADS", "16"))
    reuse = os.environ.get("VERILATOR_REUSE", "1") == "1"
    obj_dir = tb_path.parent / "obj_dir"
    exe = obj_dir / "Vtb"
    if not (reuse and exe.exists()):
        print("running SV hardware version...")
        exe = build_verilator(tb_path, sv_sources, threads=threads, clean=not reuse)
    print("running SV hardware version...")
    run_verilator_exe(exe, tb_path.parent)

    hw_out = read_mem(output_mem)
    sw_out = out_q.tolist()
    sw_float = [val / SCALE for val in sw_out]
    hw_float = [val / SCALE for val in hw_out]
    print(f"SW (q): {sw_out}")
    print(f"HW (q): {hw_out}")
    print(f"SW (f): {sw_float}")
    print(f"HW (f): {hw_float}")
    if hw_out != sw_out:
        raise AssertionError(f"Mismatch:\nHW: {hw_out}\nSW: {sw_out}")
    print("PASS")


if __name__ == "__main__":
    run_demo()
