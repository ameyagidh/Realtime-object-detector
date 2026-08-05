"""
Fine-tune COCO-pretrained YOLOv8n on the shelf-product subset built by
download_coco_subset.py. This is a genuine transfer-learning workflow, not a
from-scratch "custom model" story: we start from Ultralytics' official
COCO-pretrained weights and continue training on the narrower class set.
"""
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent


def main():
    model = YOLO("yolov8n.pt")  # official COCO-pretrained checkpoint
    t0 = time.time()
    model.train(
        data=str(ROOT / "shelf.yaml"),
        epochs=30,
        imgsz=640,
        device="mps",
        project=str(ROOT / "runs"),
        name="shelf_finetune",
        patience=10,
    )
    print(f"training took {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
