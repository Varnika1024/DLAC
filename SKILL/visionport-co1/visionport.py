import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

app = FastAPI(title="Indian Traffic Detection API")

# Allow requests from the PyCharm frontend server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n.pt")


@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
  contents = await file.read()
  nparr = np.frombuffer(contents, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  results = model(img)
  annotated_img = results[0].plot()

  _, encoded_img = cv2.imencode(".jpg", annotated_img)
  return StreamingResponse(
      io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg"
  )


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)