from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from annotation import annotate_detections
from config import (
    CANDIDATE_COLOR,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    FINAL_COLOR,
    MODEL_PATH,
    NMS_ULTRALYTICS,
    TARGET_FPS,
    DetectionParameters,
)
from detection import inference
from detector import YoloDetector, confidence_filter
from nms_official import official_postprocess


def _empty_detections():
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()


class VideoThread(QThread):
    """Capture frames (camera, image or video) and run YOLO + post-processing off the GUI thread."""

    # original, candidate view, final view, candidates, finals, mean confidence, fps
    frame_ready = pyqtSignal(object, object, object, int, int, float, float)
    status_changed = pyqtSignal(str)
    processing_error = pyqtSignal(str)
    model_changed = pyqtSignal(str, object)  # model path, class names
    model_failed = pyqtSignal(str)

    def __init__(
        self,
        camera_index: int = 0,
        source_type: str = "camera",
        file_path: str = "",
        model_path: str | Path = MODEL_PATH,
        parent=None,
    ):
        super().__init__(parent)
        self._camera_index = camera_index
        self._source_type = source_type
        self._file_path = file_path
        self._parameters: DetectionParameters | None = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._restart = threading.Event()
        self._wake = threading.Event()  # wakes the thread when it is idle (paused / still image)
        self._model_path = Path(model_path)
        self._pending_model: Path | None = self._model_path
        self._dirty = True  # parameters or model changed since the last processed frame
        self._paused = False
        self._last_status = ""
        self._last_error = ""

    def set_parameters(self, parameters: DetectionParameters) -> None:
        with self._lock:
            if parameters != self._parameters:
                self._dirty = True
            self._parameters = parameters
        self._wake.set()

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused
        self._wake.set()

    def set_model(self, model_path: str | Path) -> None:
        """Load another ONNX model (done inside the worker thread)."""
        with self._lock:
            self._pending_model = Path(model_path)
        self._wake.set()

    def restart_source(self) -> None:
        self._restart.set()
        self._wake.set()

    def _snapshot(self):
        """Return (parameters, paused, dirty, pending_model) and clear the change flags."""
        with self._lock:
            dirty, pending = self._dirty, self._pending_model
            self._dirty = False
            self._pending_model = None
            return self._parameters, self._paused, dirty, pending

    def _emit_status(self, message: str) -> None:
        if message != self._last_status:
            self._last_status = message
            self.status_changed.emit(message)

    def _emit_error(self, message: str) -> None:
        if message != self._last_error:
            self._last_error = message
            self.processing_error.emit(message)

    @staticmethod
    def _validate(result):
        boxes, scores, class_ids = result
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
        if not (len(boxes) == len(scores) == len(class_ids)):
            raise ValueError("postprocess_detections returned arrays of different lengths")
        return boxes, scores, class_ids

    def _open_source(self):
        """Return (capture, still_image). Exactly one of them is not None on success."""
        if self._source_type == "file":
            image = cv2.imread(self._file_path)
            if image is not None:
                return None, image
            cap = cv2.VideoCapture(self._file_path)
        else:
            cap = cv2.VideoCapture(self._camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        if not cap.isOpened():
            cap.release()
            return None, None
        return cap, None

    def _process(self, detector, frame, parameters, loop_started):
        detector.set_class_filter(parameters.class_filter)
        processing_outputs: dict = {}
        postprocess = official_postprocess if parameters.nms_mode == NMS_ULTRALYTICS else None
        try:
            final = self._validate(
                inference(
                    frame,
                    detector,
                    parameters.confidence_threshold,
                    parameters.nms_threshold,
                    processing_outputs,
                    postprocess,
                )
            )
            self._emit_error("")
        except Exception as error:  # a bug in the student's code must not kill the app
            self._emit_error(f"{type(error).__name__}: {error}")
            final = _empty_detections()

        proposals = processing_outputs.get("proposals", _empty_detections())
        candidates = confidence_filter(*proposals, parameters.confidence_threshold)
        candidate_view = annotate_detections(
            frame, *candidates, CANDIDATE_COLOR,
            parameters.show_labels, parameters.show_confidence, thickness=1,
            class_names=detector.class_names,
        )
        final_view = annotate_detections(
            frame, *final, FINAL_COLOR, parameters.show_labels, parameters.show_confidence,
            class_names=detector.class_names,
        )
        mean_conf = float(final[1].mean()) if len(final[1]) else 0.0
        elapsed = time.perf_counter() - loop_started
        fps = 1.0 / elapsed if elapsed > 0 else TARGET_FPS
        self.frame_ready.emit(
            bgr_to_qimage(frame), bgr_to_qimage(candidate_view), bgr_to_qimage(final_view),
            len(candidates[1]), len(final[1]), mean_conf, fps,
        )

    def _load_model(self, path: Path, current):
        """Try to load a model. Returns the new detector, or the current one if loading fails."""
        try:
            detector = YoloDetector(path)
        except Exception as error:  # missing file, not a YOLO model, missing onnxruntime, ...
            self.model_failed.emit(f"{path.name}: {error}")
            self._emit_status(f"Model error: {error}")
            return current
        self.model_changed.emit(str(path), list(detector.class_names))
        self._emit_status(f"Model loaded: {path.name}")
        return detector

    def run(self) -> None:
        detector = None
        cap = None
        still_image = None
        last_frame = None
        frame_processed = False
        was_paused = False
        frame_period = 1.0 / TARGET_FPS
        try:
            while self._running.is_set():
                self._wake.clear()  # clear BEFORE reading the state so no wake-up is lost
                if self._restart.is_set():
                    self._restart.clear()
                    if cap is not None:
                        cap.release()
                    cap, still_image, last_frame, frame_processed = None, None, None, False

                parameters, paused, dirty, pending_model = self._snapshot()
                if pending_model is not None:
                    detector = self._load_model(pending_model, detector)
                    dirty = True
                if parameters is None or detector is None:
                    self._wake.wait(0.2)  # nothing to do yet (no parameters or no valid model)
                    if detector is None:
                        with self._lock:
                            self._dirty = True
                    continue

                if cap is None and still_image is None:
                    cap, still_image = self._open_source()
                    if cap is None and still_image is None:
                        self._emit_status("Source is unavailable. Retrying...")
                        self._wake.wait(1.0)
                        continue
                    self._emit_status("Source connected")
                    last_frame, frame_processed = None, False

                loop_started = time.perf_counter()
                if still_image is not None:
                    last_frame = still_image
                frozen = (paused or still_image is not None) and last_frame is not None

                if frozen:
                    # Paused (or a single image): do NOTHING unless something changed, and then
                    # process the same frame exactly once.
                    if paused:
                        self._emit_status("Paused")
                    was_paused = paused
                    if not dirty and frame_processed:
                        self._wake.wait()  # sleep until the GUI changes something
                        continue
                    frame = last_frame
                else:
                    if was_paused and cap is not None and self._source_type == "camera":
                        for _ in range(5):  # drop stale frames buffered while paused
                            cap.grab()
                    if was_paused:
                        self._emit_status("Source connected")
                    was_paused = False
                    ret, frame = cap.read()
                    if not ret:
                        if self._source_type == "file":  # video file: loop
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                        if not ret:
                            self._emit_status("Frame read failed. Reconnecting...")
                            cap.release()
                            cap = None
                            self._wake.wait(0.25)
                            continue
                    last_frame = frame

                self._process(detector, frame, parameters, loop_started)
                frame_processed = True
                if not frozen:
                    elapsed = time.perf_counter() - loop_started
                    time.sleep(max(0.0, frame_period - elapsed))
        finally:
            if cap is not None:
                cap.release()
            self._emit_status("Source stopped")

    def stop(self) -> None:
        self._running.clear()
        self._wake.set()
        self.wait()
