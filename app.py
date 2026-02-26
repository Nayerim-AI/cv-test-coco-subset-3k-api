import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import torch

app = FastAPI(title="YOLOv8 Detection API")

WEIGHT_PATH = "yolov8-coco-subset-3k/yolov8_coco_subset_30005/weights/best.pt"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

model = YOLO(WEIGHT_PATH)


@app.get("/")
def health_check():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    results = model.predict(image_np, device=DEVICE, conf=0.3, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class_id": int(box.cls.item()),
                "class_name": model.names[int(box.cls.item())],
                "confidence": float(box.conf.item()),
                "bbox": box.xyxy.tolist()[0]
            })

    return {
        "num_detections": len(detections),
        "detections": detections
    }
