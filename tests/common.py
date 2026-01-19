from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
import os

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WIDTH = 16
FRAC = 8
SCALE = 1 << FRAC


def q88_from_float_tensor(t: torch.Tensor) -> torch.Tensor:
    scaled = t * SCALE
    rounded = torch.where(
        scaled >= 0, torch.floor(scaled + 0.5), torch.ceil(scaled - 0.5)
    )
    return rounded.to(torch.int16)


def linear_q88_torch(x_q: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
    x_f = x_q.to(torch.float32) / SCALE
    w_f = w_q.to(torch.float32) / SCALE
    y_f = x_f @ w_f.t()
    return q88_from_float_tensor(y_f)


def write_mem(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        for val in values:
            handle.write(f"{to_hex(val)}\n")


def to_hex(val: int, width: int = WIDTH) -> str:
    mask = (1 << width) - 1
    return format(val & mask, f"0{width // 4}x")


def sign_extend(val: int, width: int = WIDTH) -> int:
    sign_bit = 1 << (width - 1)
    mask = (1 << width) - 1
    val &= mask
    return (val ^ sign_bit) - sign_bit


def read_mem(path: Path, width: int = WIDTH) -> list[int]:
    values: list[int] = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            values.append(sign_extend(int(text, 16), width))
    return values


def tanh_lut_q88(
    x_q: torch.Tensor,
    lut_size: int = 1024,
    x_min_q: int = -(4 << FRAC),
    x_max_q: int = (4 << FRAC),
) -> torch.Tensor:
    range_q = x_max_q - x_min_q
    idx = ((x_q.to(torch.int32) - x_min_q) * (lut_size - 1)) // range_q
    idx = torch.clamp(idx, 0, lut_size - 1).to(torch.int64)
    xs = torch.empty((lut_size,), dtype=torch.float64)
    for i in range(lut_size):
        signed_val = x_min_q + (i * range_q) // (lut_size - 1)
        xs[i] = signed_val / SCALE
    lut = q88_from_float_tensor(torch.tanh(xs.to(torch.float32))).to(torch.int16)
    flat = lut[idx.view(-1)].view(x_q.shape)
    return flat


def build_verilator(
    tb_path: Path,
    sv_sources: list[Path],
    top: str = "tb",
    threads: int | None = None,
    clean: bool = True,
) -> Path:
    if threads is None:
        threads = int(os.environ.get("VERILATOR_THREADS", "16"))
    build_dir = tb_path.parent / "obj_dir"
    if clean and build_dir.exists():
        shutil.rmtree(build_dir)

    cmd = [
        "verilator",
        "--binary",
        "-Wall",
        "-Wno-fatal",
        "-CFLAGS",
        "-O3",
        "--build-jobs",
        str(threads),
        "--top-module",
        top,
        "--Mdir",
        str(build_dir),
        str(tb_path),
    ] + [str(src) for src in sv_sources]
    if threads and threads > 1:
        cmd[4:4] = ["--threads", str(threads)]

    subprocess.run(cmd, check=True, cwd=tb_path.parent)
    return build_dir / f"V{top}"


def run_verilator_exe(exe: Path, cwd: Path) -> None:
    result = subprocess.run(
        [str(exe)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Simulation failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def run_verilator(
    tb_path: Path,
    sv_sources: list[Path],
    top: str = "tb",
    threads: int | None = None,
) -> None:
    exe = build_verilator(tb_path, sv_sources, top=top, threads=threads, clean=True)
    run_verilator_exe(exe, tb_path.parent)
