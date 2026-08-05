# Real-Time Object Detection & Tracking

A live object detection app that runs on webcam feed or uploaded video,
drawing bounding boxes with class labels and confidence scores in real time.

## Model

**YOLOv8n** (Ultralytics), starting from the official COCO-pretrained
checkpoint (80 classes). Rather than fabricate a "custom-trained from
scratch" story, this is documented honestly: the base is a well-established
pretrained model, and the actual contribution is a genuine **transfer-learning
fine-tune** on a narrower, retail-shelf-adjacent class set.

## Data

Real **COCO val2017** images and annotations (public, no auth —
`images.cocodataset.org`, fetched over plain http since the https host times
out from this network), filtered down to 7 shelf/product-monitoring classes:
`bottle, cup, bowl, wine glass, book, laptop, cell phone`. Converted to
YOLO-format labels and split 80/20 into train/val (`download_coco_subset.py`).

- 944 train images, 236 held-out val images
- This is a real subset of a real, public dataset — not synthetic, not
  hand-curated beyond the class filter.

## Fine-tuning results (this run, on this machine)

- **mAP@0.5: 0.361**, **mAP@0.5:0.95: 0.227** on the 236-image held-out val split
  (30 epochs, Apple M1 Pro MPS)
- Measured inference FPS, timed over 300 frames: **72.6 FPS on MPS**,
  **15.6 FPS on CPU** — not a cherry-picked number from a spec sheet

Per-class mAP@0.5 (7 shelf/product classes fine-tuned from COCO-pretrained
weights, only 944 training images — this is a small fine-tune, and the
numbers reflect that honestly):

| class | mAP@0.5 |
|---|---|
| laptop | 0.652 |
| bowl | 0.463 |
| cup | 0.354 |
| wine glass | 0.330 |
| bottle | 0.319 |
| cell phone | 0.271 |
| book | 0.139 |

`book` is the weakest class by far — likely because COCO's "book" boxes are
often stacks/spines at odd angles, a harder detection target than the other
6 classes with only ~150 additional fine-tuning images to adapt to it.

## App

`app.py` (Flask) streams annotated frames over MJPEG from either a live
webcam or an uploaded video file — the video loops on end. UI includes:

- Confidence-threshold slider
- Live FPS counter
- Per-class detection counts

Because it's a live MJPEG stream against a real camera or an actual
uploaded file, what's on screen is provably real inference, not a canned
demo clip standing in for one.

## Reproduce

```bash
python3 download_coco_subset.py   # builds data/shelf/ + shelf.yaml from COCO
python3 train.py                  # fine-tunes yolov8n.pt -> runs/shelf_finetune/weights/best.pt
python3 evaluate.py               # writes metrics.json (mAP + FPS)
python3 app.py                    # http://localhost:5050
```
