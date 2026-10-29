# Lab07_YOLODetection

Welcome to the seventh assignment of the Computer Vision and Machine Learning lab course!

This project provides a real-time PyQt6 application ("YOLO Detection Playground") built around a pretrained YOLO object detector. The model is a black box: you do **not** train it. The goal of this lab is to understand what happens **after** the network, namely confidence thresholding, IoU, and Non-Maximum Suppression (NMS), by watching the effect on a live webcam stream.

## 🎯 Assignment Objectives

A YOLOv5 model at 640x640 input does not return a clean list of objects. It returns **25200 box proposals** (`3 x (80x80 + 40x40 + 20x20)`), most of them useless and many of them overlapping the same object. In this lab you will complete one function located in `detection.py`:

```
postprocess_detections(boxes, scores, class_ids, confidence_threshold, nms_threshold)
```

Students should modify only `detection.py`. Do not modify `main.py`, `gui.py`, `video_thread.py`, `detector.py`, `nms_official.py`, `annotation.py`, `config.py`, or any other project file. `detection.py` also contains a ready-made `inference()` function; leave it unchanged.

The expected processing pipeline is:

```
Camera Frame
     ↓
YOLO model (fixed, pretrained)
     ↓
25200 raw proposals (boxes, scores, class_ids)
     ↓
Confidence filtering          ← your code
     ↓
Per-class NMS (IoU)           ← your code
     ↓
Final Detections
```

## ✨ Features

- Real-time webcam (or video file) detection with PyQt6.
- Three synchronized views: **Original**, **Candidate Detections** (after confidence filtering, before NMS), and **Final Detections** (after NMS).
- Confidence threshold and NMS IoU threshold sliders with visible values.
- Class filter (COCO, 80 classes) with All / None buttons.
- Show/hide labels and confidences.
- Switch between **your** NMS and the **official Ultralytics NMS** to compare results.
- Statistics panel: number of candidates, number of final detections, mean confidence, FPS.
- Dark mode and light mode, full-screen mode, responsive scaling.
- Webcam, image-file and video-file input, camera-index selection, pause, restart, and a reset-to-default button.
- **Pause really pauses:** capture and inference stop and the CPU is idle. Only when you change a slider (or the model) while paused, the frozen frame is processed once more, so you can study NMS on a single frame.
- **Choose your own model:** click the folder icon in the *Model* box to load any YOLOv5/YOLOv8-style `.onnx` model from your computer (the default is `assets/yolov5s.onnx`). Class names are read from the model when available. If a model cannot be loaded, the previous one stays active.
- Settings are restored automatically between sessions.

## 🛠️ Setup & Installation

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment (recommended):

```
python -m venv venv
```

Activate it on Linux/macOS:

```
source venv/bin/activate
```

On Windows PowerShell:

```
venv\Scripts\Activate.ps1
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

4. Make sure the model exists at `assets/yolov5s.onnx` (see `assets/README.md`).

5. Run the application:

```
python main.py
```

## 🎮 Using the Application

### Confidence threshold

Decreasing it makes more proposals survive; increasing it removes weak ones. Watch the **Candidate Detections** view: several overlapping boxes appear around each object.

### NMS IoU threshold

Boxes of the same class whose IoU with a higher-scoring kept box is **greater** than this value are suppressed. Watch the **Final Detections** view: NMS should reduce each cluster of candidates to one box. A very high threshold keeps duplicates; a very low one removes nearby, distinct objects.

### NMS implementation

- **Student NMS (detection.py)**: your implementation. Until you finish it, the Final view stays empty (placeholder behavior).
- **Official Ultralytics NMS**: reference result. Compare both views and look for differences.

If your function raises an exception the application keeps running and shows the error in the statistics panel.

## 📂 Repository Structure

```
cvml-lab07-yolo-detection/
├── .gitignore
├── Readme.md
├── requirements.txt
├── main.py
├── gui.py
├── video_thread.py
├── detector.py
├── nms_official.py
├── annotation.py
├── config.py
├── assets/
│   ├── README.md
│   └── yolov5s.onnx          # pretrained model (not part of the code you edit)
└── detection.py              # The only file students should modify
```

### Module responsibilities

- `main.py` — application entry point only.
- `gui.py` — PyQt6 user interface, controls, themes, persistence, and rendering.
- `video_thread.py` — capture and real-time processing loop.
- `detector.py` — ONNX model loading, letterbox pre-processing, and decoding of the raw output.
- `nms_official.py` — official Ultralytics NMS used for comparison.
- `annotation.py` — bounding-box visualization.
- `config.py` — constants and shared parameter definitions.
- `detection.py` — `inference()` (ready) and `postprocess_detections()` (your implementation).

## ⌨️ Keyboard Shortcuts

- `Space` — Pause / Resume
- `F11` — Full Screen / Windowed
- `R` — Reset Defaults
- `Q` — Quit
