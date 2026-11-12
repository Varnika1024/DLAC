from pathlib import Path
from collections import Counter
import base64
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
MODEL_PATH = "yolov8n.pt"

# COCO class IDs: bicycle=1, car=2, motorcycle=3, bus=5, truck=7
TRAFFIC_CLASSES = [1, 2, 3, 5, 7]
CLASS_NAMES = {
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def choose_device():
    """Use Apple's MPS backend when available; otherwise use CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = choose_device()
model = YOLO(MODEL_PATH)

app = FastAPI(title="VisionPort 2 - CO1 Traffic Detection")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
async def health():
    return {
        "status": "ready",
        "model": MODEL_PATH,
        "device": DEVICE,
        "traffic_classes": list(CLASS_NAMES.keys()),
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum size is 10 MB.")

    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read this image. Try JPG or PNG.")

    used_device = DEVICE
    start = time.perf_counter()

    try:
        results = model.predict(
            source=image,
            device=used_device,
            classes=TRAFFIC_CLASSES,
            conf=0.25,
            imgsz=640,
            verbose=False,
        )
        if used_device == "mps":
            torch.mps.synchronize()
    except Exception:
        # If MPS inference fails, retry this image on CPU.
        used_device = "cpu"
        results = model.predict(
            source=image,
            device="cpu",
            classes=TRAFFIC_CLASSES,
            conf=0.25,
            imgsz=640,
            verbose=False,
        )

    inference_ms = round((time.perf_counter() - start) * 1000, 2)
    result = results[0]
    annotated = result.plot()

    # Include every tracked category, even when its count is zero.
    counts = {name: 0 for name in CLASS_NAMES}
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            confidence = round(float(box.conf.item()), 3)
            coordinates = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            detections.append({
                "class": class_name,
                "confidence": confidence,
                "box_xyxy": coordinates,
            })
            if class_name in counts:
                counts[class_name] += 1

    success, encoded = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode the annotated image.")

    image_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return {
        "annotated_image": image_base64,
        "detections": detections,
        "counts": counts,
        "total": len(detections),
        "inference_ms": inference_ms,
        "device": used_device,
        "model": MODEL_PATH,
    }
