from __future__ import annotations

import numpy as np


def inference(
    frame: np.ndarray,
    detector,
    confidence_threshold: float,
    nms_threshold: float,
    processing_outputs: dict,
    postprocess=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Already implemented. Do NOT modify.

    Runs the pretrained YOLO model, then hands all raw proposals to postprocess_detections.
    """
    boxes, scores, class_ids = detector.predict(frame)  # e.g. 25200 raw proposals
    processing_outputs["proposals"] = (boxes, scores, class_ids)  # for the GUI only
    postprocess = postprocess or postprocess_detections  # GUI may swap in the official NMS
    return postprocess(boxes, scores, class_ids, confidence_threshold, nms_threshold)


def postprocess_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    confidence_threshold: float,
    nms_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solution: confidence filtering followed by per-class Non-Maximum Suppression.
    """
    # 1. Confidence filtering.
    keep_confident = scores >= confidence_threshold
    boxes, scores, class_ids = boxes[keep_confident], scores[keep_confident], class_ids[keep_confident]

    kept: list[int] = []
    # 2. NMS is applied separately for every class.
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        indices = indices[np.argsort(-scores[indices])]  # 3. highest score first
        while indices.size > 0:  # 4. repeat until no boxes are left
            best = indices[0]
            kept.append(int(best))
            rest = indices[1:]
            if rest.size == 0:
                break
            x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
            y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
            x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
            y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
            intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            area_best = (boxes[best, 2] - boxes[best, 0]) * (boxes[best, 3] - boxes[best, 1])
            area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = intersection / (area_best + area_rest - intersection + 1e-9)
            indices = rest[iou <= nms_threshold]  # discard boxes overlapping the kept one

    # 5. Collect the survivors of all classes.
    kept_indices = np.asarray(kept, dtype=np.int64)
    return boxes[kept_indices], scores[kept_indices], class_ids[kept_indices]
