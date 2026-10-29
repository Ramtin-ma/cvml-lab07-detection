"""Fixed infrastructure: model loading, pre-processing and decoding of the raw output.

Students do not modify this file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np

from config import COCO_CLASSES, INPUT_SIZE, MODEL_PATH


class YoloDetector:
    """Black-box pretrained YOLO (ONNX). predict() returns ALL raw proposals, undecoded by NMS."""

    def __init__(self, model_path: Path = MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. See assets/README.md."
            )
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        shape = model_input.shape
        self.input_size = int(shape[2]) if isinstance(shape[2], int) else INPUT_SIZE
        self._class_filter: np.ndarray | None = None
        self.class_names = self._read_class_names()

    def _read_class_names(self) -> list[str]:
        """Class names: from the ONNX metadata (Ultralytics exports), else COCO, else generic."""
        try:
            raw = self._session.get_modelmeta().custom_metadata_map.get("names")
            if raw:
                names = ast.literal_eval(raw)
                if isinstance(names, dict):
                    return [str(names[k]) for k in sorted(names)]
                if isinstance(names, (list, tuple)):
                    return [str(n) for n in names]
        except Exception:
            pass
        shape = self._session.get_outputs()[0].shape
        dims = [d for d in shape[1:] if isinstance(d, int)]
        if len(dims) == 2:
            features = min(dims)
            num_classes = features - 4 if dims[0] < dims[1] else features - 5
            if num_classes != len(COCO_CLASSES) and num_classes > 0:
                return [f"class {i}" for i in range(num_classes)]
        return list(COCO_CLASSES)

    def set_class_filter(self, class_ids) -> None:
        """None keeps all classes; otherwise keep only the given class ids."""
        self._class_filter = None if class_ids is None else np.asarray(class_ids, dtype=np.int64)

    def _letterbox(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        ratio = min(self.input_size / height, self.input_size / width)
        new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = (self.input_size - new_w) / 2
        pad_h = (self.input_size - new_h) / 2
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, ratio, left, top

    def predict(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (boxes xyxy in frame pixels, scores, class_ids) for every proposal."""
        padded, ratio, left, top = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(
            padded, 1.0 / 255.0, (self.input_size, self.input_size), swapRB=True
        )
        output = self._session.run(None, {self._input_name: blob})[0][0]

        # YOLOv5 export: (25200, 5 + nc) = x, y, w, h, objectness, class probabilities.
        # Anchor-free export (YOLOv8-style): (4 + nc, 8400) = x, y, w, h, class probabilities.
        if output.ndim != 2 or min(output.shape) < 5:
            raise ValueError(f"unsupported model output shape {output.shape}; expected a YOLOv5/YOLOv8-style head")
        if output.shape[0] < output.shape[1]:
            output = output.T
            class_scores = output[:, 4:]
        else:
            class_scores = output[:, 5:] * output[:, 4:5]
        class_ids = class_scores.argmax(axis=1).astype(np.int64)
        scores = class_scores.max(axis=1).astype(np.float32)

        cx, cy, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        boxes = np.stack(
            [cx - w / 2 - left, cy - h / 2 - top, cx + w / 2 - left, cy + h / 2 - top], axis=1
        ) / ratio
        frame_h, frame_w = frame.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_h - 1)
        boxes = boxes.astype(np.float32)

        if self._class_filter is not None:
            keep = np.isin(class_ids, self._class_filter)
            boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        return boxes, scores, class_ids


def confidence_filter(boxes, scores, class_ids, confidence_threshold):
    """Reference confidence filtering used for the 'Candidate Detections' view."""
    keep = scores >= confidence_threshold
    return boxes[keep], scores[keep], class_ids[keep]
