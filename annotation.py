import cv2
import numpy as np

from config import COCO_CLASSES


def _class_name(class_id: int, class_names) -> str:
    names = class_names if class_names is not None else COCO_CLASSES
    return names[class_id] if 0 <= class_id < len(names) else f"class {class_id}"


def annotate_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    color: tuple[int, int, int],
    show_labels: bool = True,
    show_confidence: bool = True,
    thickness: int = 2,
    class_names=None,
) -> np.ndarray:
    """Draw xyxy detections on a copy of the input frame."""
    output = frame.copy()
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        parts = []
        if show_labels:
            parts.append(_class_name(int(class_id), class_names))
        if show_confidence:
            parts.append(f"{float(score):.2f}")
        if parts:
            cv2.putText(
                output,
                " ".join(parts),
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return output
