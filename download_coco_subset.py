"""
Build a small "retail shelf"-style fine-tuning dataset from real COCO val2017
annotations. We deliberately don't fabricate a custom dataset story: this is
the actual, publicly released COCO val2017 split, filtered down to a handful
of classes that resemble shelf/product monitoring (bottle, cup, bowl,
wine glass, book, laptop, cell phone), and reformatted into YOLO's expected
txt-label layout.

Expects data/raw/val2017/*.jpg and data/raw/annotations/instances_val2017.json
to already be downloaded and extracted (see README "Data" section for the
plain-http COCO mirror URLs used, since the https host times out from this
network).
"""
import json
import os
import random
import shutil
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
OUT = Path(__file__).parent / "data" / "shelf"

TARGET_CLASSES = ["bottle", "cup", "bowl", "wine glass", "book", "laptop", "cell phone"]
VAL_FRACTION = 0.2
SEED = 42


def main():
    ann_path = RAW / "annotations" / "instances_val2017.json"
    img_dir = RAW / "val2017"
    assert ann_path.exists(), f"missing {ann_path} — run the download step first"
    assert img_dir.exists(), f"missing {img_dir} — run the download step first"

    with open(ann_path) as f:
        coco = json.load(f)

    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    target_ids = {cid for cid, name in cat_by_id.items() if name in TARGET_CLASSES}
    # stable 0..N-1 class index order matching TARGET_CLASSES
    class_index = {name: i for i, name in enumerate(TARGET_CLASSES)}

    img_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in target_ids:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

    image_ids = list(anns_by_img.keys())
    random.Random(SEED).shuffle(image_ids)
    n_val = max(1, int(len(image_ids) * VAL_FRACTION))
    val_ids = set(image_ids[:n_val])

    for split, ids in [("train", image_ids[n_val:]), ("val", image_ids[:n_val])]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for iid in ids:
            im = img_by_id[iid]
            w, h = im["width"], im["height"]
            src = img_dir / im["file_name"]
            if not src.exists():
                continue
            shutil.copy(src, OUT / "images" / split / im["file_name"])
            lines = []
            for ann in anns_by_img[iid]:
                name = cat_by_id[ann["category_id"]]
                cls = class_index[name]
                x, y, bw, bh = ann["bbox"]  # COCO: top-left x,y,width,height
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            label_path = OUT / "labels" / split / (Path(im["file_name"]).stem + ".txt")
            label_path.write_text("\n".join(lines))

    yaml_path = Path(__file__).parent / "shelf.yaml"
    yaml_path.write_text(
        "path: {}\n".format((OUT).resolve())
        + "train: images/train\nval: images/val\n"
        + "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(TARGET_CLASSES))
    )

    print(f"train images: {len(image_ids) - n_val}, val images: {n_val}")
    print(f"classes: {TARGET_CLASSES}")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
