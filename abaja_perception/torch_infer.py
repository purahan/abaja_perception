"""
Torch-backed detector for development/testing on a regular laptop —
no TensorRT, no Jetson required. Use this while you don't have a
fine-tuned model yet, or while working on anything other than the actual
Jetson deployment. Swap to TRTDetector (trt_infer.py) only once you're on
the Jetson with an exported engine.

Two modes:
  - Pretrained COCO weights (e.g. 'yolov8n.pt', auto-downloaded by
    Ultralytics on first run): lets you sanity-check the whole pipeline
    (capture -> inference -> tracker -> overlay -> recording) right now,
    using COCO's own class names (person, car, bicycle, etc.) since COCO
    doesn't have your custom 15 classes.
  - Your own fine-tuned weights matching config/classes.yaml's 15
    classes: once trained, point `torch_weights` at that .pt file and set
    `use_custom_classes: true` so per-class thresholds from the config
    apply correctly.
"""

from ultralytics import YOLO
from .tracker import Detection


class TorchDetector:
    def __init__(self, weights_path: str, device: str = 'cpu'):
        self.model = YOLO(weights_path)
        self.device = device
        # model.names is a dict {id: name} - matches COCO's 80 classes for
        # pretrained weights, or your own class list once you fine-tune.
        self.names = self.model.names

    def detect(self, frame_bgr, thresholds: dict, default_threshold: float = 0.25,
               use_custom_classes: bool = False):
        """Runs inference, applies per-class (or default) confidence
        thresholds, returns a list of tracker.Detection (already NMS'd by
        ultralytics internally, so no separate NMS step needed here)."""
        results = self.model.predict(
            frame_bgr, device=self.device, verbose=False, conf=0.05)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            threshold = thresholds.get(cls_id, default_threshold) if use_custom_classes else default_threshold
            if conf < threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(class_id=cls_id, conf=conf, bbox=(x1, y1, x2, y2)))
        return detections
