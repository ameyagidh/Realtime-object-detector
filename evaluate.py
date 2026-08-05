"""
Report the honest numbers for the README: real mAP on the held-out shelf
validation split, and measured (not cherry-picked) inference FPS on this
machine, on both MPS and CPU.
"""
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "runs" / "shelf_finetune" / "weights" / "best.pt"


def measure_fps(model, device, n_frames=300, imgsz=640):
    import numpy as np

    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    model.to(device)
    # warmup
    for _ in range(10):
        model.predict(dummy, device=device, verbose=False)
    t0 = time.time()
    for _ in range(n_frames):
        model.predict(dummy, device=device, verbose=False)
    elapsed = time.time() - t0
    return n_frames / elapsed


def main():
    assert WEIGHTS.exists(), f"missing {WEIGHTS} — run train.py first"
    model = YOLO(str(WEIGHTS))

    metrics = model.val(data=str(ROOT / "shelf.yaml"), device="mps")
    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "per_class_mAP50": {
            name: float(ap) for name, ap in zip(metrics.names.values(), metrics.box.ap50)
        },
    }

    results["fps_mps"] = measure_fps(YOLO(str(WEIGHTS)), "mps")
    results["fps_cpu"] = measure_fps(YOLO(str(WEIGHTS)), "cpu")

    out_path = ROOT / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
