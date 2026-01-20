from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


def load_model(weights_path: str | Path | None = None):
    if weights_path is None:
        weights_path = Path(__file__).with_name("yolov8n.pt")
    return YOLO(str(weights_path))


def export_state_dict_pt(path: str | Path) -> None:
    model = load_model()
    path = Path(path)
    torch.save(model.model.state_dict(), path)


if __name__ == "__main__":
    model = load_model()
    out_path = Path(__file__).with_name("yolov8n_state_dict.pt")
    torch.save(model.model.state_dict(), out_path)
    print(f"Saved state_dict to {out_path}")
