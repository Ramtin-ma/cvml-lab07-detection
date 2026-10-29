from __future__ import annotations

"""Fixed infrastructure: the official Ultralytics NMS, for comparison with the student's NMS."""

import cv2
import numpy as np

from detector import confidence_filter

_warned_fallback = False


def official_postprocess(boxes, scores, class_ids, confidence_threshold, nms_threshold):
    """Same signature/return value as detection.postprocess_detections."""
    global _warned_fallback
    boxes, scores, class_ids = confidence_filter(boxes, scores, class_ids, confidence_threshold)
    if len(boxes) == 0:
        return boxes, scores, class_ids

    try:
        import torch
        from ultralytics.utils.ops import non_max_suppression

        num_classes = int(class_ids.max()) + 1
        xywh = np.stack(
            [
                (boxes[:, 0] + boxes[:, 2]) / 2,
                (boxes[:, 1] + boxes[:, 3]) / 2,
                boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1],
            ],
            axis=1,
        )
        class_matrix = np.zeros((len(boxes), num_classes), dtype=np.float32)
        class_matrix[np.arange(len(boxes)), class_ids] = scores
        prediction = torch.from_numpy(np.concatenate([xywh, class_matrix], axis=1).T[None])
        result = non_max_suppression(
            prediction, conf_thres=confidence_threshold, iou_thres=nms_threshold
        )[0].cpu().numpy()
        return (
            result[:, :4].astype(np.float32),
            result[:, 4].astype(np.float32),
            result[:, 5].astype(np.int64),
        )
    except ImportError:
        if not _warned_fallback:
            print("ultralytics/torch not installed: falling back to cv2.dnn.NMSBoxes")
            _warned_fallback = True
        keep_all = []
        for class_id in np.unique(class_ids):
            idx = np.where(class_ids == class_id)[0]
            xywh = [
                [float(boxes[i, 0]), float(boxes[i, 1]),
                 float(boxes[i, 2] - boxes[i, 0]), float(boxes[i, 3] - boxes[i, 1])]
                for i in idx
            ]
            kept = cv2.dnn.NMSBoxes(xywh, scores[idx].tolist(), confidence_threshold, nms_threshold)
            keep_all.extend(idx[np.asarray(kept, dtype=int).reshape(-1)].tolist())
        keep_all = np.asarray(keep_all, dtype=int)
        return boxes[keep_all], scores[keep_all], class_ids[keep_all]
