import numpy as np
import cv2
import os
import base64
from typing import List, Optional
from common.types.ObjectDetection import ObjectDetection

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
model: Optional[object] = None


def _get_model():
    global model
    if model is not None:
        return model
    if YOLO is None:
        raise RuntimeError("ultralytics is not available on this system")
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(
            f"YOLO model file not found at '{MODEL_PATH}'. Automatic download is disabled."
        )

    # Only load local model weights; this avoids any automatic network download.
    model = YOLO(MODEL_PATH)
    return model

def toBase64Image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")

def saveImage(image, filename, path='./'):
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(f"{path}/{filename}", image)

def detectObjects(image):
    yolo_model = _get_model()

    results = yolo_model(image, verbose=False)
    detections: List[ObjectDetection] = []
    for result in results:
        for box in result.boxes:
            x, y, w, h = box.xywh[0]
            detections.append(ObjectDetection(
                yolo_model.names[int(box.cls.item())], 
                box.conf.item(), 
                x.item(), 
                y.item(), 
                w.item(), 
                h.item())
            )        
    return detections

def box_label(image, box, label, color, text_color):
    x1, y1, x2, y2 = box
    result = image.copy()
    cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(result, label, (int(x1) + 15, int(y1) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    return result

def plotDetections(image):
    yolo_model = _get_model()

    results = yolo_model(image, verbose=False)
    if len(results) == 0:
        return image
    return results[0].plot()
