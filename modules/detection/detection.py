'''
Detects potholes in the current image and returns the bounding boxes
with classes.
'''
from ..config.config import config
import cv2

def apply_detection(img, model):
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model.to(config["detection"]["device"])

    model.conf = config["detection"]["conf"]
    model.iou = config["detection"]["iou"]
    model.agnostic = config["detection"]["agnostic"]
    model.multi_label = config["detection"]["multi_label"]
    model.classes = config["detection"]["classes"]
    model.max_det = config["detection"]["max_det"]
    model.amp = config["detection"]["amp"]
    model.agnostic = config["detection"]["agnostic"]
    model.augment = config["detection"]["augment"]

    results = model(cv2.resize(frame, (640,640)), size=640)

    return results.pandas().xyxy[0]
