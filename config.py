from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

APP_NAME = "CVML Lab 07 - YOLO Detection"
ORGANIZATION_NAME = "Sharif CVML Lab"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "assets" / "yolov5s.onnx"
INPUT_SIZE = 640  # YOLOv5 @ 640 produces 3 * (80*80 + 40*40 + 20*20) = 25200 proposals

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
TARGET_FPS = 30

DEFAULT_CAMERA_INDEX = 0
DEFAULT_THEME = "dark"

NMS_STUDENT = "Student"
NMS_ULTRALYTICS = "Ultralytics"

DEFAULT_NMS_MODE = NMS_STUDENT
DEFAULT_CONFIDENCE_PERCENT = 25
DEFAULT_NMS_IOU_PERCENT = 45
MIN_CONFIDENCE_PERCENT = 5  # a very low threshold makes a pure-Python NMS slow
DEFAULT_SHOW_LABELS = True
DEFAULT_SHOW_CONFIDENCE = True

DEFAULT_SETTINGS_MESSAGE = "Settings: Ready"

CANDIDATE_COLOR = (0, 165, 255)  # BGR, orange
FINAL_COLOR = (60, 220, 140)     # BGR, green

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

FILE_FILTER = (
    "Media Files (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff "
    "*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;"
    "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;"
    "Videos (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;"
    "All Files (*)"
)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass(frozen=True)
class DetectionParameters:
    confidence_threshold: float
    nms_threshold: float
    nms_mode: str
    class_filter: tuple[int, ...] | None  # None = all classes
    show_labels: bool
    show_confidence: bool
