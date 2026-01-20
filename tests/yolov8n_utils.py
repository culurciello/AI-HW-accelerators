from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO

from common import FRAC, SCALE, q88_from_float_tensor, silu_q88, write_mem


def load_yolov8n(model_path: Path) -> YOLO:
    if model_path.exists():
        return YOLO(str(model_path))
    return YOLO("yolov8n.pt")


def bn_scale_bias_q88(bn) -> tuple[torch.Tensor, torch.Tensor]:
    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    bias = bn.bias.detach() - bn.running_mean.detach() * scale
    return q88_from_float_tensor(scale), q88_from_float_tensor(bias)


def yolo_conv_q88(x_q: torch.Tensor, conv) -> torch.Tensor:
    w_q = q88_from_float_tensor(conv.conv.weight.detach())
    scale_q, bias_q = bn_scale_bias_q88(conv.bn)
    x_ref = x_q.to(torch.float32) / SCALE
    w_ref = w_q.to(torch.float32) / SCALE
    y_f = F.conv2d(x_ref, w_ref, bias=None, stride=conv.conv.stride, padding=conv.conv.padding)
    y_q = q88_from_float_tensor(y_f)
    prod = y_q.to(torch.int32) * scale_q.view(1, -1, 1, 1).to(torch.int32)
    acc = prod + (bias_q.view(1, -1, 1, 1).to(torch.int32) << FRAC)
    rounding = 1 << (FRAC - 1)
    y_q = torch.where(
        acc >= 0,
        (acc + rounding) >> FRAC,
        -(((-acc) + rounding) >> FRAC),
    ).to(torch.int16)
    return silu_q88(y_q)


def conv2d_bias_q88(
    x_q: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    w_q = q88_from_float_tensor(weight.detach())
    b_q = q88_from_float_tensor(bias.detach())
    x_ref = x_q.to(torch.float32) / SCALE
    w_ref = w_q.to(torch.float32) / SCALE
    b_ref = b_q.to(torch.float32) / SCALE
    y_f = F.conv2d(x_ref, w_ref, bias=b_ref, stride=stride, padding=padding)
    return q88_from_float_tensor(y_f)


def maxpool2d_q88(x_q: torch.Tensor, kernel: int = 5, stride: int = 1, padding: int = 2) -> torch.Tensor:
    x_ref = x_q.to(torch.float32) / SCALE
    y_f = F.max_pool2d(x_ref, kernel_size=kernel, stride=stride, padding=padding)
    return q88_from_float_tensor(y_f)


def upsample2d_q88(x_q: torch.Tensor, scale: int = 2) -> torch.Tensor:
    return x_q.repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3)


def concat2d_q88(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tensors, dim=1)


def bottleneck_q88(x_q: torch.Tensor, bottleneck) -> torch.Tensor:
    y_q = yolo_conv_q88(x_q, bottleneck.cv1)
    y_q = yolo_conv_q88(y_q, bottleneck.cv2)
    if getattr(bottleneck, "add", False) and x_q.shape == y_q.shape:
        y_q = (x_q + y_q).to(torch.int16)
    return y_q


def c2f_q88(x_q: torch.Tensor, c2f) -> torch.Tensor:
    y_q = yolo_conv_q88(x_q, c2f.cv1)
    mid_ch = y_q.shape[1] // 2
    x1, x2 = torch.split(y_q, mid_ch, dim=1)
    outs = [x1, x2]
    cur = x2
    for m in c2f.m:
        cur = bottleneck_q88(cur, m)
        outs.append(cur)
    cat = concat2d_q88(outs)
    return yolo_conv_q88(cat, c2f.cv2)


def sppf_q88(x_q: torch.Tensor, sppf) -> torch.Tensor:
    y_q = yolo_conv_q88(x_q, sppf.cv1)
    y1 = maxpool2d_q88(y_q, kernel=5, stride=1, padding=2)
    y2 = maxpool2d_q88(y1, kernel=5, stride=1, padding=2)
    y3 = maxpool2d_q88(y2, kernel=5, stride=1, padding=2)
    cat = concat2d_q88([y_q, y1, y2, y3])
    return yolo_conv_q88(cat, sppf.cv2)


def detect_q88(xs_q: list[torch.Tensor], detect) -> list[torch.Tensor]:
    outs: list[torch.Tensor] = []
    for i, x_q in enumerate(xs_q):
        r0 = yolo_conv_q88(x_q, detect.cv2[i][0])
        r1 = yolo_conv_q88(r0, detect.cv2[i][1])
        r2 = conv2d_bias_q88(r1, detect.cv2[i][2].weight, detect.cv2[i][2].bias)
        c0 = yolo_conv_q88(x_q, detect.cv3[i][0])
        c1 = yolo_conv_q88(c0, detect.cv3[i][1])
        c2 = conv2d_bias_q88(c1, detect.cv3[i][2].weight, detect.cv3[i][2].bias)
        outs.append(torch.cat([r2, c2], dim=1))
    return outs


def flatten_chw(x_q: torch.Tensor) -> list[int]:
    return x_q.squeeze(0).contiguous().flatten().tolist()


def write_conv_bn_files(layer_dir: Path, prefix: str, conv) -> tuple[Path, Path, Path]:
    w_q = q88_from_float_tensor(conv.conv.weight.detach())
    scale_q, bias_q = bn_scale_bias_q88(conv.bn)
    w_mem = layer_dir / f"{prefix}_w.mem"
    s_mem = layer_dir / f"{prefix}_bn_scale.mem"
    b_mem = layer_dir / f"{prefix}_bn_bias.mem"
    write_mem(w_mem, w_q.flatten().tolist())
    write_mem(s_mem, scale_q.flatten().tolist())
    write_mem(b_mem, bias_q.flatten().tolist())
    return w_mem, s_mem, b_mem


def write_conv_bias_files(layer_dir: Path, prefix: str, conv) -> tuple[Path, Path]:
    w_q = q88_from_float_tensor(conv.weight.detach())
    b_q = q88_from_float_tensor(conv.bias.detach())
    w_mem = layer_dir / f"{prefix}_w.mem"
    b_mem = layer_dir / f"{prefix}_b.mem"
    write_mem(w_mem, w_q.flatten().tolist())
    write_mem(b_mem, b_q.flatten().tolist())
    return w_mem, b_mem
