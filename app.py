"""
Live object-detection demo: streams annotated frames from a webcam or an
uploaded video over MJPEG, so what's on screen is provably real inference
against a live feed or an actual uploaded file — not a canned demo clip.
"""
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
from flask import Flask, Response, render_template, request, jsonify
from ultralytics import YOLO

ROOT = Path(__file__).parent
FINETUNED = ROOT / "runs" / "shelf_finetune" / "weights" / "best.pt"
WEIGHTS = str(FINETUNED) if FINETUNED.exists() else "yolov8n.pt"

app = Flask(__name__)
model = YOLO(WEIGHTS)

state = {
    "conf": 0.25,
    "source": None,       # None -> webcam, else path to uploaded video
    "counts": Counter(),
    "fps": 0.0,
    "lock": threading.Lock(),
}

UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def frame_generator():
    cap = cv2.VideoCapture(state["source"] if state["source"] else 0)
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            if state["source"]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop uploaded video
                continue
            break

        with state["lock"]:
            conf = state["conf"]

        results = model.predict(frame, conf=conf, verbose=False)[0]
        annotated = results.plot()

        counts = Counter(model.names[int(c)] for c in results.boxes.cls) if results.boxes is not None else Counter()
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        with state["lock"]:
            state["counts"] = counts
            state["fps"] = fps

        ok, buf = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    cap.release()


@app.route("/")
def index():
    return render_template("index.html", weights=Path(WEIGHTS).name)


@app.route("/video_feed")
def video_feed():
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/set_conf", methods=["POST"])
def set_conf():
    with state["lock"]:
        state["conf"] = float(request.form["conf"])
    return jsonify(ok=True)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files["video"]
    dest = UPLOAD_DIR / f.filename
    f.save(dest)
    state["source"] = str(dest)
    return jsonify(ok=True)


@app.route("/use_webcam", methods=["POST"])
def use_webcam():
    state["source"] = None
    return jsonify(ok=True)


@app.route("/stats")
def stats():
    with state["lock"]:
        return jsonify(fps=round(state["fps"], 1), counts=dict(state["counts"]))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
