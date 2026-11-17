import base64
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "yolov8n.pt"

# Choose Apple Silicon GPU (MPS) when available; otherwise use CPU.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

app = FastAPI(title="VisionPort - Skill 3 Inference Trace")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n[STARTUP] Loading YOLOv8 Nano model...")
model = YOLO(str(MODEL_PATH) if MODEL_PATH.exists() else "yolov8n.pt")
print(f"[STARTUP] Model ready. Device selected: {DEVICE}\n")


@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "YOLOv8 Nano",
        "device": DEVICE,
        "pytorch_mps_available": torch.backends.mps.is_available(),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    request_start = time.perf_counter()
    print("\n========== INFERENCE REQUEST TRACE ==========")

    # 1. API INPUT
    print("[1/5 API INPUT] Request received")
    print(f"             Filename: {file.filename}")
    print(f"             Content type: {file.content_type}")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    print(f"             File size: {len(image_bytes)} bytes")

    # 2. PREPROCESSING
    print("[2/5 PREPROCESSING] Converting uploaded bytes into an image")
    decode_start = time.perf_counter()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    preprocess_ms = (time.perf_counter() - decode_start) * 1000

    if image is None:
        raise HTTPException(status_code=400, detail="Could not read this image.")

    height, width = image.shape[:2]
    print(f"             Image shape: {height} x {width} pixels")
    print(f"             Decode time: {preprocess_ms:.2f} ms")

    # 3. MODEL EXECUTION
    print(f"[3/5 MODEL EXECUTION] Running YOLOv8 Nano on {DEVICE}")
    inference_start = time.perf_counter()
    try:
        predictions = model.predict(
            source=image,
            device=DEVICE,
            classes=[1, 2, 3, 5, 7],  # bicycle, car, motorcycle, bus, truck
            conf=0.25,
            verbose=False,
        )
        actual_device = DEVICE
    except Exception as exc:
        # Fallback in case MPS is unavailable for an operation.
        print(f"             Device error: {exc}")
        print("             Retrying inference on CPU...")
        predictions = model.predict(
            source=image,
            device="cpu",
            classes=[1, 2, 3, 5, 7],
            conf=0.25,
            verbose=False,
        )
        actual_device = "cpu"

    inference_ms = (time.perf_counter() - inference_start) * 1000
    result = predictions[0]
    print(f"             Model execution time: {inference_ms:.2f} ms")

    # 4. POSTPROCESSING
    print("[4/5 POSTPROCESSING] Reading detected objects and drawing boxes")
    detections = []
    names = result.names
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            detections.append({
                "class_id": class_id,
                "class_name": names[class_id],
                "confidence": round(confidence, 3),
                "box": [x1, y1, x2, y2],
            })

    counts = dict(Counter(item["class_name"] for item in detections))
    annotated_image = result.plot()
    success, encoded_image = cv2.imencode(".jpg", annotated_image)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode the result image.")

    image_base64 = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
    print(f"             Number of detected vehicles: {len(detections)}")

    # 5. API RESPONSE
    total_ms = (time.perf_counter() - request_start) * 1000
    response_data = {
        "message": "Inference completed successfully",
        "image": f"data:image/jpeg;base64,{image_base64}",
        "image_width": width,
        "image_height": height,
        "detections": detections,
        "counts": counts,
        "total_vehicles": len(detections),
        "preprocessing_ms": round(preprocess_ms, 2),
        "inference_ms": round(inference_ms, 2),
        "total_request_ms": round(total_ms, 2),
        "device": actual_device,
        "model": "YOLOv8 Nano",
    }
    print("[5/5 API RESPONSE] JSON response prepared and sent to browser")
    print(f"             Total request time: {total_ms:.2f} ms")
    print("========== TRACE COMPLETE ==========\n")
    return response_data
