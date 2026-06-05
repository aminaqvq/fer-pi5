import os
import sys
import time
import math
import queue
import threading
import logging
import logging.handlers
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_BACKEND = "tflite-runtime"
except ImportError:
    import tensorflow as tf  # pyright: ignore[reportMissingModuleSource]
    tflite = tf.lite
    TFLITE_BACKEND = "tensorflow-lite"


# -----------------------------
# Global constants / logger
# -----------------------------
LABELS = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]

logger = logging.getLogger("fer_infer")


@dataclass
class Config:
    # Model files
    tflite_path: str = "/home/amina/workspaces/fer-pi5/export/final_stage2_balanced_clean/fer_mbv3_stage2_final_fp16.tflite"
    yunet_path: str = "/home/amina/workspaces/fer-pi5/src/deploy/face_detection_yunet_2023mar.onnx"

    # Camera
    camera_source: Any = 0
    cam_w: int = 640
    cam_h: int = 480
    cam_fps: int = 30
    use_mjpg: bool = False
    mirror_flip: bool = True

    # Processing resolution
    proc_w: int = 640
    proc_h: int = 480

    # Detection
    det_w: int = 256
    det_h: int = 192
    detect_every: int = 3
    score_th: float = 0.70
    nms_th: float = 0.30
    top_k: int = 1000  # Pi 实时场景通常不需要 5000，降低 NMS 开销

    # Classification
    img_size: int = 224
    infer_every: int = 2
    max_faces: int = 4
    max_infer_faces: int = 4  # 原代码默认 1 容易造成“只画框不分类”的错觉，这里默认最多 4 张脸都分类
    tflite_threads: int = 4
    conf_th: float = 0.38  # 全局默认阈值降低一点，减少非弱项被判 Unknown
    pad_ratio: float = 0.18

    # 推理显示层校准：
    # 基于 stage2 final_test 的 per-class F1/recall，三个弱项是 disgust、fear、sad。
    # 这是后处理，不改变模型本身；只影响实时 demo 的显示/保存判断。
    # bias 是 logit 偏置：0.22 约等于该类概率乘以 exp(0.22)=1.25 后再归一化。
    # 想关掉就改成 False；想换弱项就改下面两个字典里的类别名。
    enable_prob_calibration: bool = True
    class_logit_bias: Dict[str, float] = field(
        default_factory=lambda: {
            # fear 仍然偏差时，不建议只继续猛加 disgust；否则 fear/neutral/sad 很容易被吸成 disgust。
            # 这版把 fear 提高、disgust 稍微收一点，让 fear 更有机会成为 top1。
            "disgust": 0.42,
            "fear": 0.50,
            "sad": 0.10,
            "anger": 0.38,
        }
    )
    # 每类阈值。这里把所有类都列出来，Unknown 会明显减少；保存仍由 save_min_conf 控制。
    class_conf_th: Dict[str, float] = field(
        default_factory=lambda: {
            "anger": 0.28,
            "disgust": 0.26,
            "fear": 0.28,
            "happy": 0.44,
            "sad": 0.42,
            "surprise": 0.40,
            "neutral": 0.40,
        }
    )
    weak_labels: Tuple[str, ...] = ("disgust", "fear", "sad")

    # Unknown 降噪：top1 未过阈值时，如果与 top2 有足够间隔，也允许显示 top1。
    # 这样比把所有阈值粗暴降到 0.20 更稳。
    enable_unknown_margin_fallback: bool = True
    unknown_fallback_conf: float = 0.32
    unknown_fallback_margin: float = 0.07
    weak_unknown_fallback_conf: float = 0.25
    weak_unknown_fallback_margin: float = 0.035

    # UI 稳定：如果当前帧掉到 Unknown，短时间保留上一帧有效表情，减少闪烁。
    # 注意：hold_last 只是显示层，不会触发保存。
    hold_last_label_on_unknown: bool = True
    hold_last_label_frames: int = 6
    hold_last_label_min_conf: float = 0.34
    hold_last_label_conf_decay: float = 0.95

    # 重要：这里必须和训练阶段一致。
    # 可选："imagenet", "zero_one", "minus_one_one", "none"
    # 如果你训练时只是 image/255，请改成 "zero_one"。
    # 如果你训练时是 (image/127.5)-1，请改成 "minus_one_one"。
    preprocess_mode: str = "imagenet"
    color_order: str = "RGB"  # 可选："RGB" 或 "BGR"；训练时如果用 OpenCV 原图训练可尝试 BGR

    # Tracking
    track_max_missing: int = 10
    track_max_dist: float = 90.0

    # Saving
    save_dir: str = "/home/amina/workspaces/fer-pi5/docs/图片/best_by_class"
    save_min_conf: float = 0.55
    save_min_sharpness: float = 60.0
    save_unknown: bool = False
    save_best_only: bool = False  # False = 所有符合条件的样本都保存，方便后期筛选
    save_cooldown_frames: int = 12  # 防止同一 track 每帧疯狂保存

    # FPS / UI
    target_fps: int = 30
    wait_key_ms: int = 1

    # Logging
    log_dir: str = "logs"
    log_level: int = logging.INFO  # Pi 上 DEBUG 每帧写日志会拖慢实时性能
    log_every_n_frames: int = 10
    opencv_threads: int = 2  # 避免 OpenCV 与 TFLite 过度抢 CPU，可在 1/2/4 间实测


CFG = Config()


# -----------------------------
# Logging
# -----------------------------
def setup_logger(log_dir: str = "logs", level: int = logging.DEBUG) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "fer_infer.log")

    lg = logging.getLogger("fer_infer")
    lg.setLevel(level)
    lg.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 避免重复添加 handler
    if not lg.handlers:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=8 * 1024 * 1024,
            backupCount=8,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        console_handler.setLevel(logging.INFO)

        lg.addHandler(file_handler)
        lg.addHandler(console_handler)

    lg.info("Logger initialized: %s", os.path.abspath(log_path))
    return lg


# -----------------------------
# Camera reader
# -----------------------------
class CameraReader:
    def __init__(self, source: Any, width: int, height: int, fps: int, use_mjpg: bool = False):
        self.cap = cv2.VideoCapture(source, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {source}")

        try:
            if use_mjpg:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            logger.exception("Failed to set some camera properties")

        self._lock = threading.Lock()
        self._stop = False
        self._ok = False
        self._frame: Optional[np.ndarray] = None
        self._frame_ts = 0.0
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "CameraReader":
        self._th.start()
        return self

    def _loop(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read()
            ts = time.perf_counter()
            with self._lock:
                self._ok = bool(ok)
                if ok and frame is not None:
                    self._frame = frame
                    self._frame_ts = ts
            time.sleep(0.0005)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        with self._lock:
            if not self._ok or self._frame is None:
                return False, None, 0.0
            return True, self._frame.copy(), self._frame_ts

    def release(self) -> None:
        self._stop = True
        try:
            self._th.join(timeout=1.0)
        except Exception:
            logger.exception("Failed to join camera thread")
        self.cap.release()


# -----------------------------
# FPS helpers
# -----------------------------
class FPSMeter:
    def __init__(self, alpha: float = 0.08):
        self.alpha = alpha
        self.prev_t = time.perf_counter()
        self.fps_ema = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self.prev_t
        self.prev_t = now
        fps = 1.0 / dt if dt > 1e-9 else 0.0
        if self.fps_ema <= 0.0:
            self.fps_ema = fps
        else:
            self.fps_ema = (1.0 - self.alpha) * self.fps_ema + self.alpha * fps
        return self.fps_ema


class FramePacer:
    def __init__(self, target_fps: int):
        self.target_fps = max(1, int(target_fps))
        self.frame_interval = 1.0 / float(self.target_fps)

    def pace(self, loop_start: float, frame_idx: int) -> Tuple[float, bool]:
        elapsed = time.perf_counter() - loop_start
        sleep_time = self.frame_interval - elapsed
        slower_than_target = sleep_time <= 0
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            logger.warning(
                "Frame processing slower than target FPS: frame=%d elapsed=%.2fms target=%.2fms",
                frame_idx,
                elapsed * 1000.0,
                self.frame_interval * 1000.0,
            )
        return elapsed, slower_than_target


# -----------------------------
# Model preprocessing / inference
# -----------------------------
def _shape_matches(actual: Tuple[int, ...], expected: Tuple[int, ...]) -> bool:
    if len(actual) != len(expected):
        return False
    return all(e in (-1, 0) or a == e for a, e in zip(actual, expected))


def _infer_input_layout(input_shape: Tuple[int, ...]) -> str:
    # TFLite FER 模型通常是 NHWC: [1,224,224,3]，也可能是 NCHW: [1,3,224,224]
    if len(input_shape) != 4:
        raise ValueError(f"Expected 4D input tensor, got shape={input_shape}")
    if input_shape[1] in (1, 3):
        return "NCHW"
    if input_shape[-1] in (1, 3):
        return "NHWC"
    # shape 里可能有 -1；默认 NHWC，因为 TFLite 更常见
    return "NHWC"


def preprocess_roi(bgr: np.ndarray, input_shape: Any, cfg: Config = CFG) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty ROI received by preprocess_roi")

    target_shape = tuple(int(v) for v in input_shape)
    layout = _infer_input_layout(target_shape)

    if cfg.color_order.upper() == "RGB":
        roi = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif cfg.color_order.upper() == "BGR":
        roi = bgr.copy()
    else:
        raise ValueError(f"Unsupported color_order={cfg.color_order!r}; use RGB or BGR")

    roi = cv2.resize(roi, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
    x = roi.astype(np.float32)

    mode = cfg.preprocess_mode.lower()
    if mode == "imagenet":
        x = x / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    elif mode == "zero_one":
        x = x / 255.0
    elif mode == "minus_one_one":
        x = (x / 127.5) - 1.0
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported preprocess_mode={cfg.preprocess_mode!r}")

    if layout == "NHWC":
        x4 = np.expand_dims(x, axis=0)
    else:
        x4 = np.expand_dims(np.transpose(x, (2, 0, 1)), axis=0)

    if not _shape_matches(tuple(x4.shape), target_shape):
        raise ValueError(
            f"Unsupported input shape: produced={x4.shape}, model_expected={target_shape}, layout={layout}"
        )

    return x4


class TFLiteFER:
    def __init__(self, model_path: str, num_threads: int = 4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.in_det = self.interpreter.get_input_details()[0]
        self.out_det = self.interpreter.get_output_details()[0]
        self.input_shape = tuple(int(v) for v in self.in_det["shape"])
        self.layout = _infer_input_layout(self.input_shape)

        logger.info(
            "[TFLite:%s] Input: shape=%s dtype=%s quant=%s layout=%s",
            TFLITE_BACKEND,
            self.in_det["shape"],
            self.in_det["dtype"],
            self.in_det.get("quantization"),
            self.layout,
        )
        logger.info(
            "[TFLite:%s] Output: shape=%s dtype=%s quant=%s",
            TFLITE_BACKEND,
            self.out_det["shape"],
            self.out_det["dtype"],
            self.out_det.get("quantization"),
        )
        if len(LABELS) != int(np.prod(self.out_det["shape"])) and int(np.prod(self.out_det["shape"])) > 1:
            logger.warning(
                "Label count may not match output size: labels=%d output_numel=%d. Check LABELS order/map.",
                len(LABELS),
                int(np.prod(self.out_det["shape"])),
            )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float32).reshape(-1)
        exp_y = np.exp(logits - np.max(logits))
        denom = float(np.sum(exp_y))
        if denom <= 0 or not np.isfinite(denom):
            raise ValueError(f"Invalid softmax denominator: {denom}")
        return exp_y / denom

    def infer(self, x: np.ndarray) -> np.ndarray:
        scale, zp = self.in_det.get("quantization", (0.0, 0))
        input_dtype = self.in_det["dtype"]

        if input_dtype == np.int8:
            if scale == 0:
                raise ValueError("Input int8 but quant scale is 0.")
            x_in = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
        elif input_dtype == np.uint8:
            if scale == 0:
                # Some uint8 models expect raw uint8 input.
                x_in = np.clip(np.round(x), 0, 255).astype(np.uint8)
            else:
                x_in = np.clip(np.round(x / scale + zp), 0, 255).astype(np.uint8)
        else:
            x_in = x.astype(input_dtype)

        self.interpreter.set_tensor(self.in_det["index"], x_in)
        self.interpreter.invoke()

        yq = self.interpreter.get_tensor(self.out_det["index"])
        oscale, ozp = self.out_det.get("quantization", (0.0, 0))
        output_dtype = self.out_det["dtype"]

        if output_dtype in (np.int8, np.uint8) and oscale != 0:
            y = (yq.astype(np.float32) - float(ozp)) * float(oscale)
        else:
            y = yq.astype(np.float32)

        y = y.reshape(-1)
        # 如果模型已经输出概率，直接归一；否则 softmax。
        if np.all(y >= 0.0) and np.isclose(float(np.sum(y)), 1.0, atol=1e-3):
            probs = y / max(float(np.sum(y)), 1e-8)
        else:
            probs = self._softmax(y)
        return probs


# -----------------------------
# Detection / tracking
# -----------------------------
class FaceDetectorYuNet:
    def __init__(self, model_path: str, input_size: Tuple[int, int], score_th: float, nms_th: float, top_k: int):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found: {model_path}")

        if hasattr(cv2, "FaceDetectorYN_create"):
            self.det = cv2.FaceDetectorYN_create(model_path, "", input_size, score_th, nms_th, top_k)
        else:
            self.det = cv2.FaceDetectorYN.create(model_path, "", input_size, score_th, nms_th, top_k)

    def detect(self, frame_bgr: np.ndarray, det_w: int, det_h: int) -> List[Dict[str, Any]]:
        H, W = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        self.det.setInputSize((det_w, det_h))

        faces = self.det.detect(small)
        if isinstance(faces, tuple):
            faces = faces[1]
        if faces is None or len(faces) == 0:
            return []

        sx = W / float(det_w)
        sy = H / float(det_h)
        out: List[Dict[str, Any]] = []

        for f in faces.astype(np.float32):
            x, y, w, h = f[0:4]
            lms = f[4:14].reshape(5, 2)
            score = float(f[14])

            box = clamp_box([int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)], W, H)
            landmarks = [(int(px * sx), int(py * sy)) for px, py in lms]
            out.append({"box": box, "landmarks": landmarks, "det_conf": score})

        return out


@dataclass
class Track:
    track_id: int
    box: List[int]
    landmarks: List[Tuple[int, int]]
    det_conf: float
    last_seen_frame: int
    label: str = "Unknown"
    cls_conf: float = 0.0
    probs: Optional[np.ndarray] = None  # 校准后的概率，用于显示和最终判断
    raw_probs: Optional[np.ndarray] = None  # 模型原始概率，方便日志排查
    roi_box: Optional[List[int]] = None
    last_cls_frame: int = -999999
    sharpness: float = 0.0
    last_saved_frame: int = -999999
    label_decision: str = "init"  # threshold / margin_fallback / hold_last / unknown


class LandmarkTracker:
    def __init__(self, max_missing: int = 10, max_dist: float = 90.0):
        self.max_missing = max_missing
        self.max_dist = max_dist
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    @staticmethod
    def _center_from_landmarks(landmarks: List[Tuple[int, int]], box: List[int]) -> Tuple[float, float]:
        if landmarks and len(landmarks) == 5:
            xs = [p[0] for p in landmarks]
            ys = [p[1] for p in landmarks]
            return float(sum(xs)) / 5.0, float(sum(ys)) / 5.0
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update(self, detections: List[Dict[str, Any]], frame_idx: int) -> List[Track]:
        active_ids = list(self.tracks.keys())
        det_used = set()
        trk_used = set()

        pairs = []
        for tid in active_ids:
            tr = self.tracks[tid]
            tcx, tcy = self._center_from_landmarks(tr.landmarks, tr.box)
            for di, det in enumerate(detections):
                dcx, dcy = self._center_from_landmarks(det["landmarks"], det["box"])
                dist = math.hypot(tcx - dcx, tcy - dcy)
                pairs.append((dist, tid, di))

        pairs.sort(key=lambda item: item[0])
        for dist, tid, di in pairs:
            if dist > self.max_dist or tid in trk_used or di in det_used:
                continue
            det = detections[di]
            tr = self.tracks[tid]
            tr.box = det["box"]
            tr.landmarks = det["landmarks"]
            tr.det_conf = float(det["det_conf"])
            tr.last_seen_frame = frame_idx
            trk_used.add(tid)
            det_used.add(di)

        for di, det in enumerate(detections):
            if di in det_used:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(
                track_id=tid,
                box=det["box"],
                landmarks=det["landmarks"],
                det_conf=float(det["det_conf"]),
                last_seen_frame=frame_idx,
            )

        stale_ids = [tid for tid, tr in self.tracks.items() if frame_idx - tr.last_seen_frame > self.max_missing]
        for tid in stale_ids:
            logger.debug("Drop stale track: id=%d frame=%d", tid, frame_idx)
            self.tracks.pop(tid, None)

        return self.get_active(frame_idx)

    def get_active(self, frame_idx: int) -> List[Track]:
        tracks = [tr for tr in self.tracks.values() if frame_idx - tr.last_seen_frame <= self.max_missing]
        tracks.sort(key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]), reverse=True)
        return tracks


# -----------------------------
# Image saving
# -----------------------------
class AsyncImageSaver:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.q: "queue.Queue[Optional[Tuple[str, np.ndarray]]]" = queue.Queue(maxsize=64)
        self._stop = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self) -> None:
        while not self._stop:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                break
            path, image = item
            try:
                ok = cv2.imwrite(path, image, JPEG_PARAMS)
                if not ok:
                    logger.error("cv2.imwrite returned False: %s", path)
                else:
                    logger.info("Saved image: %s", path)
            except Exception:
                logger.exception("Failed to save image: %s", path)
            finally:
                self.q.task_done()

    def submit(self, path: str, image: np.ndarray) -> bool:
        try:
            self.q.put_nowait((path, image.copy()))
            return True
        except queue.Full:
            logger.warning("Save queue full; dropped image: %s", path)
            return False

    def close(self) -> None:
        self._stop = True
        try:
            self.q.put(None, timeout=0.5)
        except queue.Full:
            logger.warning("Could not enqueue saver sentinel because queue is full")
        try:
            self._th.join(timeout=2.0)
        except Exception:
            logger.exception("Failed to join saver thread")


def safe_label_name(label: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    cleaned = "".join(ch if ch in keep else "_" for ch in label)
    return cleaned or "Unknown"


def make_unique_path(directory: str, filename: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(filename)
    for _ in range(32):
        alt = os.path.join(directory, f"{stem}_{uuid.uuid4().hex[:6]}{ext}")
        if not os.path.exists(alt):
            return alt
    raise RuntimeError(f"Could not create unique filename in {directory}")


def build_save_paths(save_dir: str, label: str, conf: float, frame_idx: int, track_id: int) -> Tuple[str, str]:
    label_dir = os.path.join(save_dir, safe_label_name(label))
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    uid = uuid.uuid4().hex[:6]
    base = f"{ts}_frame{frame_idx:06d}_track{track_id:03d}_conf{conf:.2f}_{uid}"
    crop_path = make_unique_path(label_dir, f"{base}_crop.jpg")
    annot_path = make_unique_path(label_dir, f"{base}_annot.jpg")
    return crop_path, annot_path


@dataclass
class SaveRequest:
    label: str
    conf: float
    sharpness: float
    crop_path: str
    annot_path: str
    crop: np.ndarray
    frame_idx: int
    track_id: int


# -----------------------------
# Geometry / drawing helpers
# -----------------------------
def clamp_box(box: List[int], W: int, H: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(0, min(W, int(x2)))
    y2 = max(0, min(H, int(y2)))
    if x2 <= x1 + 1:
        x2 = min(W, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(H, y1 + 2)
    return [x1, y1, x2, y2]


def expand_square_roi(box: List[int], W: int, H: int, pad_ratio: float) -> List[int]:
    x1, y1, x2, y2 = clamp_box(box, W, H)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(2, int(round(max(bw, bh) * (1.0 + pad_ratio))))

    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    nx2 = nx1 + side
    ny2 = ny1 + side

    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > W:
        nx1 -= nx2 - W
        nx2 = W
    if ny2 > H:
        ny1 -= ny2 - H
        ny2 = H

    return clamp_box([nx1, ny1, nx2, ny2], W, H)


def draw_landmarks(frame: np.ndarray, landmarks: List[Tuple[int, int]]) -> None:
    colors = [(0, 255, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 128, 255)]
    H, W = frame.shape[:2]
    for i, (x, y) in enumerate(landmarks):
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(frame, (x, y), 2, colors[i % len(colors)], -1)


def measure_sharpness(image_bgr: np.ndarray) -> float:
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def scale_box(box: List[int], sx: float, sy: float, W: int, H: int) -> List[int]:
    x1, y1, x2, y2 = box
    return clamp_box([round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)], W, H)


def scale_landmarks(landmarks: List[Tuple[int, int]], sx: float, sy: float) -> List[Tuple[int, int]]:
    return [(int(round(x * sx)), int(round(y * sy))) for x, y in landmarks]


def maybe_resize(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == target_w and h == target_h:
        return frame
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def should_save_result(
    tr: Track,
    cfg: Config,
    best_records: Dict[str, Dict[str, float]],
    frame_idx: int,
) -> bool:
    if getattr(tr, "label_decision", "") == "hold_last":
        return False
    if tr.label == "Unknown" and not cfg.save_unknown:
        return False
    if tr.label not in LABELS and tr.label != "Unknown":
        return False
    if tr.cls_conf < cfg.save_min_conf:
        return False
    if tr.sharpness < cfg.save_min_sharpness:
        return False
    if frame_idx - tr.last_saved_frame < cfg.save_cooldown_frames:
        return False
    if not cfg.save_best_only:
        return True

    prev = best_records.get(tr.label)
    if prev is None:
        return True
    prev_conf = float(prev.get("conf", 0.0))
    prev_sharp = float(prev.get("sharpness", 0.0))
    if tr.cls_conf > prev_conf + 0.01:
        return True
    if abs(tr.cls_conf - prev_conf) <= 0.01 and tr.sharpness > prev_sharp + 5.0:
        return True
    return False


def topk_to_string(probs: np.ndarray, k: int = 3) -> str:
    if probs is None or probs.size == 0:
        return "[]"
    idx = np.argsort(-probs)[:k]
    return ", ".join(f"{LABELS[int(i)]}:{float(probs[int(i)]):.4f}" for i in idx if int(i) < len(LABELS))


def calibrate_probs(probs: np.ndarray, cfg: Config = CFG) -> np.ndarray:
    """对推理概率做轻量校准，让弱项更容易被显示出来。

    做法：把概率转到 log 空间，对指定类别加一个很小的 bias，再 softmax 归一化。
    这比直接给概率 +0.05 更稳，不会出现概率和不等于 1 的情况。
    """
    p = np.asarray(probs, dtype=np.float32).reshape(-1).copy()
    if p.size == 0:
        return p

    # 只校准已知的 7 个 FER 类，避免误动模型可能输出的额外字段。
    n = min(len(LABELS), p.size)
    eps = 1e-8

    if not getattr(cfg, "enable_prob_calibration", False):
        denom = max(float(np.sum(p[:n])), eps)
        p[:n] = p[:n] / denom
        return p

    scores = np.log(np.clip(p[:n], eps, 1.0))
    applied = False

    for label, bias in getattr(cfg, "class_logit_bias", {}).items():
        if label not in LABELS:
            logger.warning("Unknown calibration label ignored: %s", label)
            continue
        idx = LABELS.index(label)
        if idx >= n:
            continue
        b = float(bias)
        if abs(b) < 1e-12:
            continue
        scores[idx] += b
        applied = True

    if not applied:
        denom = max(float(np.sum(p[:n])), eps)
        p[:n] = p[:n] / denom
        return p

    scores = scores - np.max(scores)
    adj = np.exp(scores).astype(np.float32)
    adj = adj / max(float(np.sum(adj)), eps)
    p[:n] = adj
    return p


def conf_threshold_for(label: str, cfg: Config = CFG) -> float:
    return float(getattr(cfg, "class_conf_th", {}).get(label, cfg.conf_th))


def decide_label(
    probs: np.ndarray,
    candidate_label: str,
    conf: float,
    need_conf: float,
    tr: Track,
    cfg: Config,
    frame_idx: int,
) -> Tuple[str, float, str, float]:
    """决定最终显示标签。

    先按每类阈值判断；不过阈值时，如果 top1 和 top2 拉开了差距，
    允许低置信度显示 top1；最后才用短时 hold_last 减少 Unknown 闪烁。
    """
    n = min(len(LABELS), probs.size)
    if n <= 0:
        return "Unknown", 0.0, "unknown", 0.0

    top_idx = np.argsort(-probs[:n])
    second_conf = float(probs[int(top_idx[1])]) if n > 1 else 0.0
    margin = float(conf - second_conf)

    if conf >= need_conf:
        return candidate_label, conf, "threshold", margin

    if getattr(cfg, "enable_unknown_margin_fallback", False):
        weak = candidate_label in getattr(cfg, "weak_labels", ())
        min_conf = float(cfg.weak_unknown_fallback_conf if weak else cfg.unknown_fallback_conf)
        min_margin = float(cfg.weak_unknown_fallback_margin if weak else cfg.unknown_fallback_margin)
        if conf >= min_conf and margin >= min_margin:
            return candidate_label, conf, "margin_fallback", margin

    if getattr(cfg, "hold_last_label_on_unknown", False):
        prev_label = getattr(tr, "label", "Unknown")
        prev_conf = float(getattr(tr, "cls_conf", 0.0))
        prev_frame = int(getattr(tr, "last_cls_frame", -999999))
        recent = (frame_idx - prev_frame) <= int(getattr(cfg, "hold_last_label_frames", 0))
        valid_prev = prev_label not in ("", "Unknown", "Error")
        if valid_prev and recent and prev_conf >= float(getattr(cfg, "hold_last_label_min_conf", 1.0)):
            shown_conf = max(conf, prev_conf * float(getattr(cfg, "hold_last_label_conf_decay", 0.95)))
            return prev_label, shown_conf, "hold_last", margin

    return "Unknown", conf, "unknown", margin


def classify_track(
    tr: Track,
    raw_frame: np.ndarray,
    proc_w: int,
    proc_h: int,
    raw_w: int,
    raw_h: int,
    fer: TFLiteFER,
    cfg: Config,
    frame_idx: int,
) -> Tuple[bool, float, Optional[np.ndarray]]:
    sx = raw_w / float(proc_w)
    sy = raw_h / float(proc_h)
    proc_box = clamp_box(tr.box, proc_w, proc_h)
    raw_box = scale_box(proc_box, sx, sy, raw_w, raw_h)
    rx1, ry1, rx2, ry2 = expand_square_roi(raw_box, raw_w, raw_h, cfg.pad_ratio)
    roi = raw_frame[ry1:ry2, rx1:rx2]
    tr.roi_box = [rx1, ry1, rx2, ry2]

    if roi is None or roi.size == 0:
        logger.warning("Empty ROI: frame=%d track=%d box=%s roi_box=%s", frame_idx, tr.track_id, tr.box, tr.roi_box)
        tr.label = "Unknown"
        tr.cls_conf = 0.0
        tr.label_decision = "empty_roi"
        return False, 0.0, None

    t0 = time.perf_counter()
    x = preprocess_roi(roi, fer.in_det["shape"], cfg)
    raw_probs = fer.infer(x)
    probs = calibrate_probs(raw_probs, cfg)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    if probs.size < len(LABELS):
        raise ValueError(f"Model output has fewer values than labels: output={probs.size}, labels={len(LABELS)}")

    cls_id = int(np.argmax(probs[: len(LABELS)]))
    conf = float(probs[cls_id])
    candidate_label = LABELS[cls_id]
    need_conf = conf_threshold_for(candidate_label, cfg)
    label, shown_conf, decision, margin = decide_label(probs, candidate_label, conf, need_conf, tr, cfg, frame_idx)

    tr.raw_probs = raw_probs
    tr.probs = probs
    tr.label = label
    tr.cls_conf = shown_conf
    tr.label_decision = decision
    tr.last_cls_frame = frame_idx
    tr.sharpness = measure_sharpness(roi)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Classify: frame=%d track=%d det_box=%s roi_box=%s raw_frame=%s crop=%s tensor_shape=%s "
            "preprocess=%s color=%s calibrated=%s raw_top3=[%s] top1=%s:%.4f th=%.2f margin=%.4f decision=%s final=%s shown_conf=%.4f top3=[%s] sharpness=%.2f infer_ms=%.2f",
            frame_idx,
            tr.track_id,
            str(tr.box),
            str(tr.roi_box),
            str(raw_frame.shape),
            str(roi.shape),
            str(x.shape),
            cfg.preprocess_mode,
            cfg.color_order,
            cfg.enable_prob_calibration,
            topk_to_string(raw_probs, 3),
            LABELS[cls_id],
            conf,
            need_conf,
            margin,
            decision,
            tr.label,
            shown_conf,
            topk_to_string(probs, 3),
            tr.sharpness,
            infer_ms,
        )
    return True, infer_ms, roi


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    cfg: Config,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> None:
    H, W = frame.shape[:2]
    for tr in tracks:
        x1, y1, x2, y2 = scale_box(tr.box, scale_x, scale_y, W, H)
        ok_cls = tr.label not in ("", "Unknown", "Error") and tr.cls_conf >= conf_threshold_for(tr.label, cfg)
        color = (0, 255, 0) if ok_cls else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if tr.landmarks:
            draw_landmarks(frame, scale_landmarks(tr.landmarks, scale_x, scale_y))

        label = tr.label or "Unknown"
        suffix = "~" if getattr(tr, "label_decision", "") == "hold_last" else ""
        text = f"ID{tr.track_id} {label}{suffix} {tr.cls_conf:.2f}"
        cv2.putText(frame, text, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

        if tr.probs is not None:
            top_idx = int(np.argmax(tr.probs[: len(LABELS)]))
            top_label = LABELS[top_idx]
            top_conf = float(tr.probs[top_idx])
            cv2.putText(
                frame,
                f"top={top_label}:{top_conf:.2f}",
                (x1, min(H - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
            )


def draw_status(
    frame: np.ndarray,
    fps: float,
    cfg: Config,
    latency_ms: float,
    num_tracks: int,
    last_det_ms: float,
    last_cls_ms: float,
    loop_ms: float,
    best_records: Dict[str, Dict[str, float]],
) -> None:
    H, _ = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps:.1f} / {cfg.target_fps}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(frame, f"Latency: {latency_ms:.1f} ms", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Faces:{num_tracks} det={last_det_ms:.1f}ms cls={last_cls_ms:.1f}ms loop={loop_ms:.1f}ms",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 255, 200),
        1,
    )

    y = 106
    for label in LABELS:
        if label in best_records and y < H - 8:
            rec = best_records[label]
            msg = f"best {label}: c={rec['conf']:.2f} s={rec['sharpness']:.0f}"
            cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
            y += 18


def log_frame_summary(
    frame_idx: int,
    fps: float,
    detections: Optional[List[Dict[str, Any]]],
    tracks: List[Track],
    last_det_ms: float,
    last_cls_ms: float,
    loop_ms: float,
    latency_ms: float,
    cfg: Config,
) -> None:
    if (
        cfg.log_every_n_frames <= 0
        or frame_idx % cfg.log_every_n_frames != 0
        or not logger.isEnabledFor(logging.DEBUG)
    ):
        return
    det_count = len(detections) if detections is not None else -1
    track_msg = []
    for tr in tracks:
        track_msg.append(
            {
                "id": tr.track_id,
                "box": tr.box,
                "roi": tr.roi_box,
                "det_conf": round(float(tr.det_conf), 4),
                "label": tr.label,
                "cls_conf": round(float(tr.cls_conf), 4),
                "sharpness": round(float(tr.sharpness), 2),
                "decision": getattr(tr, "label_decision", ""),
            }
        )
    logger.debug(
        "Frame: idx=%d fps=%.2f det_count=%s tracks=%s det_ms=%.2f cls_ms=%.2f loop_ms=%.2f latency_ms=%.2f",
        frame_idx,
        fps,
        det_count,
        track_msg,
        last_det_ms,
        last_cls_ms,
        loop_ms,
        latency_ms,
    )


def main() -> None:
    global logger
    logger = setup_logger(log_dir=CFG.log_dir, level=CFG.log_level)
    try:
        cv2.setNumThreads(int(getattr(CFG, "opencv_threads", 0)))
        logger.info("OpenCV threads: %s", cv2.getNumThreads())
    except Exception:
        logger.exception("Failed to set OpenCV threads")
    logger.info("Platform: %s, TFLite backend: %s", sys.platform, TFLITE_BACKEND)
    logger.info("TFLite: %s", CFG.tflite_path)
    logger.info("YuNet : %s", CFG.yunet_path)
    logger.info("Camera source: %s", CFG.camera_source)
    logger.info("Labels order: %s", LABELS)
    logger.info("Preprocess: mode=%s color_order=%s img_size=%d", CFG.preprocess_mode, CFG.color_order, CFG.img_size)
    logger.info(
        "Calibration: enabled=%s class_logit_bias=%s class_conf_th=%s unknown_fallback=%s",
        CFG.enable_prob_calibration,
        CFG.class_logit_bias,
        CFG.class_conf_th,
        CFG.enable_unknown_margin_fallback,
    )
    logger.info("Target FPS: %d", CFG.target_fps)

    fer = TFLiteFER(CFG.tflite_path, num_threads=CFG.tflite_threads)
    detector = FaceDetectorYuNet(CFG.yunet_path, (CFG.det_w, CFG.det_h), CFG.score_th, CFG.nms_th, CFG.top_k)
    tracker = LandmarkTracker(CFG.track_max_missing, CFG.track_max_dist)
    saver = AsyncImageSaver(CFG.save_dir)
    cam: Optional[CameraReader] = None

    try:
        cam = CameraReader(CFG.camera_source, CFG.cam_w, CFG.cam_h, CFG.cam_fps, CFG.use_mjpg).start()

        ok, frame, _ = False, None, 0.0
        for i in range(150):
            ok, frame, _ = cam.read()
            if ok and frame is not None:
                if CFG.mirror_flip:
                    frame = cv2.flip(frame, 1)
                logger.info("First frame received: try=%d shape=%s", i + 1, str(frame.shape))
                break
            time.sleep(0.02)

        if not ok or frame is None:
            logger.error("Camera read failed at start")
            raise RuntimeError("Camera read failed at start")

        fpsm = FPSMeter(alpha=0.08)
        pacer = FramePacer(CFG.target_fps)
        frame_idx = 0
        last_det_ms = 0.0
        last_cls_ms = 0.0
        latency_ms = 0.0
        best_records: Dict[str, Dict[str, float]] = {}

        while True:
            loop_start = time.perf_counter()
            detections: Optional[List[Dict[str, Any]]] = None
            save_requests: List[SaveRequest] = []
            cls_times: List[float] = []

            ok, raw_frame, frame_ts = cam.read()
            if not ok or raw_frame is None:
                logger.warning("Camera frame unavailable: frame=%d", frame_idx)
                time.sleep(0.003)
                continue

            if CFG.mirror_flip:
                raw_frame = cv2.flip(raw_frame, 1)

            raw_h, raw_w = raw_frame.shape[:2]
            proc_frame = maybe_resize(raw_frame, CFG.proc_w, CFG.proc_h)
            proc_h, proc_w = proc_frame.shape[:2]
            latency_ms = (time.perf_counter() - frame_ts) * 1000.0

            # Detect / track
            if frame_idx % max(1, CFG.detect_every) == 0:
                try:
                    t_det = time.perf_counter()
                    detections = detector.detect(proc_frame, CFG.det_w, CFG.det_h)
                    last_det_ms = (time.perf_counter() - t_det) * 1000.0
                    tracks = tracker.update(detections, frame_idx)
                    logger.debug("Detect: frame=%d faces=%d det_ms=%.2f", frame_idx, len(detections), last_det_ms)
                except Exception:
                    logger.exception("Detection failed: frame=%d", frame_idx)
                    tracks = tracker.get_active(frame_idx)
            else:
                tracks = tracker.get_active(frame_idx)

            tracks = tracks[: max(0, CFG.max_faces)]

            # Classify: 对每一个检测到的 track 都尝试分类；用 max_infer_faces 控制单帧上限
            infer_budget = max(0, CFG.max_infer_faces)
            inferred_this_frame = 0
            for tr in tracks:
                if inferred_this_frame >= infer_budget:
                    logger.debug(
                        "Infer budget reached: frame=%d budget=%d skipped_track=%d",
                        frame_idx,
                        infer_budget,
                        tr.track_id,
                    )
                    continue

                if tr.last_cls_frame >= 0 and (frame_idx - tr.last_cls_frame) < max(1, CFG.infer_every):
                    continue

                try:
                    ok_cls, infer_ms, roi = classify_track(tr, raw_frame, proc_w, proc_h, raw_w, raw_h, fer, CFG, frame_idx)
                    if ok_cls:
                        inferred_this_frame += 1
                        cls_times.append(infer_ms)

                    if roi is not None and should_save_result(tr, CFG, best_records, frame_idx):
                        crop_path, annot_path = build_save_paths(CFG.save_dir, tr.label, tr.cls_conf, frame_idx, tr.track_id)
                        save_requests.append(
                            SaveRequest(
                                label=tr.label,
                                conf=tr.cls_conf,
                                sharpness=tr.sharpness,
                                crop_path=crop_path,
                                annot_path=annot_path,
                                crop=roi.copy(),
                                frame_idx=frame_idx,
                                track_id=tr.track_id,
                            )
                        )
                        tr.last_saved_frame = frame_idx
                except Exception:
                    logger.exception("Classification failed: frame=%d track=%d box=%s", frame_idx, tr.track_id, tr.box)
                    tr.label = "Error"
                    tr.cls_conf = 0.0

            if cls_times:
                last_cls_ms = float(np.mean(cls_times))

            # Draw live view
            draw_tracks(proc_frame, tracks, CFG, scale_x=1.0, scale_y=1.0)

            fps = fpsm.tick()
            loop_ms_before_draw_status = (time.perf_counter() - loop_start) * 1000.0
            draw_status(
                proc_frame,
                fps,
                CFG,
                latency_ms,
                len(tracks),
                last_det_ms,
                last_cls_ms,
                loop_ms_before_draw_status,
                best_records,
            )

            # Save crop and annotated raw frame with unique names
            if save_requests:
                raw_annotated = raw_frame.copy()
                raw_scale_x = raw_w / float(proc_w)
                raw_scale_y = raw_h / float(proc_h)
                draw_tracks(raw_annotated, tracks, CFG, scale_x=raw_scale_x, scale_y=raw_scale_y)
                draw_status(
                    raw_annotated,
                    fps,
                    CFG,
                    latency_ms,
                    len(tracks),
                    last_det_ms,
                    last_cls_ms,
                    loop_ms_before_draw_status,
                    best_records,
                )

                for req in save_requests:
                    saver.submit(req.crop_path, req.crop)
                    saver.submit(req.annot_path, raw_annotated)
                    best_records[req.label] = {"conf": req.conf, "sharpness": req.sharpness}
                    logger.info(
                        "Save requested: frame=%d track=%d label=%s conf=%.4f sharpness=%.2f crop=%s annot=%s",
                        req.frame_idx,
                        req.track_id,
                        req.label,
                        req.conf,
                        req.sharpness,
                        req.crop_path,
                        req.annot_path,
                    )

            loop_ms = (time.perf_counter() - loop_start) * 1000.0
            log_frame_summary(frame_idx, fps, detections, tracks, last_det_ms, last_cls_ms, loop_ms, latency_ms, CFG)

            cv2.imshow("FER Multi + YuNet 5-point", proc_frame)

            # 先 waitKey，再 sleep，避免 sleep 后窗口事件不响应；总帧率仍由 pacer 控制在 30 FPS 左右
            key = cv2.waitKey(max(1, CFG.wait_key_ms)) & 0xFF
            if key in (27, ord("q")):
                logger.info("Exit requested by key: %s", key)
                break

            pacer.pace(loop_start, frame_idx)
            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if cam is not None:
            cam.release()
        saver.close()
        cv2.destroyAllWindows()
        logger.info("Bye")


if __name__ == "__main__":
    main()