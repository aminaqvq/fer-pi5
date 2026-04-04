#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import threading
from dataclasses import dataclass
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import cv2

# ========= TFLite 兼容导入：优先 tflite-runtime，回退 tensorflow =========
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_BACKEND = "tflite-runtime"
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    TFLITE_BACKEND = "tensorflow-lite"


# ===================== 常量 =====================
IMG_SIZE = 224
LABELS = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ===================== 配置 =====================
@dataclass
class PipelineConfig:
    tflite_path: str
    yunet_path: str

    cam_id: int = 0
    cam_w: int = 640
    cam_h: int = 480
    cam_fps: int = 30
    mjpg: bool = True

    det_w: int = 320
    det_h: int = 240
    detect_every: int = 3  # 每 N 帧跑一次检测

    # YuNet 参数
    score_th: float = 0.9
    nms_th: float = 0.3
    top_k: int = 5000

    # 推理
    tflite_threads: int = 6
    use_xnnpack: bool = True

    # 分类显示/策略
    conf_th: float = 0.5
    smooth_n: int = 10
    pad_ratio: float = 0.20  # 人脸框外扩
    light: bool = False

    # 跟踪
    tracker: str = "MOSSE"   # MOSSE(最快) / KCF(稍稳)
    # 仅在检测到单脸时启用跟踪；多脸/无脸会清空跟踪器


def choose_camera_backend() -> int:
    # Windows 用 DSHOW；Linux/Pi 用 V4L2
    return cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2


# ===================== 摄像头读取线程 =====================
class CameraReader:
    def __init__(self, cam_id: int, backend: int, width: int, height: int, fps: int, mjpg: bool = True):
        self.cap = cv2.VideoCapture(cam_id, backend)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera.")

        if mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,         fps)

        self._lock = threading.Lock()
        self._stop = False
        self._ok = False
        self._frame = None
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._th.start()
        return self

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            with self._lock:
                self._ok = ok
                if ok:
                    self._frame = frame
            time.sleep(0.001)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if not self._ok or self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._stop = True
        try:
            self._th.join(timeout=1.0)
        except Exception:
            pass
        self.cap.release()


# ===================== 工具：FPS 统计（EMA） =====================
class FPSMeter:
    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha
        self.fps_ema = 0.0
        self.t_prev = time.perf_counter()

    def tick(self) -> float:
        t_now = time.perf_counter()
        dt = t_now - self.t_prev
        self.t_prev = t_now
        fps = (1.0 / dt) if dt > 0 else 0.0
        self.fps_ema = fps if self.fps_ema == 0 else (1 - self.ema_alpha) * self.fps_ema + self.ema_alpha * fps
        return self.fps_ema


# ===================== TFLite FER 推理 =====================
def preprocess_roi(bgr: np.ndarray) -> np.ndarray:
    roi = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = roi.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.expand_dims(x, 0)


class TFLiteFER:
    def __init__(self, model_path: str, num_threads: int = 4, try_xnnpack: bool = True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        kwargs = {"model_path": model_path, "num_threads": num_threads}
        if try_xnnpack:
            try:
                delegate = tflite.load_delegate("libtensorflowlite_delegate_xnnpack.so")
                kwargs["experimental_delegates"] = [delegate]
                print("[TFLite] XNNPACK delegate enabled.")
            except Exception as e:
                print(f"[TFLite] XNNPACK not available: {e}")

        self.interpreter = tflite.Interpreter(**kwargs)
        self.interpreter.allocate_tensors()
        self.in_det = self.interpreter.get_input_details()[0]
        self.out_det = self.interpreter.get_output_details()[0]

        print(f"[TFLite:{TFLITE_BACKEND}] Input:  shape={self.in_det['shape']}, dtype={self.in_det['dtype']}, quant={self.in_det.get('quantization')}")
        print(f"[TFLite:{TFLITE_BACKEND}] Output: shape={self.out_det['shape']}, dtype={self.out_det['dtype']}, quant={self.out_det.get('quantization')}")

    def infer_probs(self, x_float: np.ndarray) -> np.ndarray:
        # input quant
        scale, zp = self.in_det.get("quantization", (0.0, 0))
        if self.in_det["dtype"] == np.int8:
            if scale == 0:
                raise ValueError("Input int8 but quant scale is 0.")
            x_q = np.round(x_float / scale + zp).astype(np.int8)
            self.interpreter.set_tensor(self.in_det["index"], x_q)
        else:
            self.interpreter.set_tensor(self.in_det["index"], x_float.astype(self.in_det["dtype"]))

        self.interpreter.invoke()

        yq = self.interpreter.get_tensor(self.out_det["index"])
        oscale, ozp = self.out_det.get("quantization", (0.0, 0))

        if self.out_det["dtype"] == np.int8:
            if oscale == 0:
                raise ValueError("Output int8 but quant scale is 0.")
            y = (yq.astype(np.float32) - ozp) * oscale
        else:
            y = yq.astype(np.float32)

        y = np.reshape(y, (-1,))
        exp_y = np.exp(y - np.max(y))
        probs = exp_y / np.sum(exp_y)
        return probs.reshape(-1)


# ===================== YuNet 检测（小图加速） =====================
class FaceDetectorYuNet:
    def __init__(self, model_path: str, input_size: Tuple[int, int], score_th: float, nms_th: float, top_k: int):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found: {model_path}")

        if hasattr(cv2, "FaceDetectorYN_create"):
            self.det = cv2.FaceDetectorYN_create(model_path, "", input_size, score_th, nms_th, top_k)
        else:
            self.det = cv2.FaceDetectorYN.create(model_path, "", input_size, score_th, nms_th, top_k)

    def detect(self, frame_bgr: np.ndarray, det_w: int, det_h: int) -> Tuple[List[List[int]], List[float]]:
        H, W = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        self.det.setInputSize((det_w, det_h))

        faces = self.det.detect(small)
        if isinstance(faces, tuple):
            faces = faces[1]
        if faces is None or len(faces) == 0:
            return [], []

        sx = W / det_w
        sy = H / det_h

        boxes, confs = [], []
        for f in faces.astype(np.float32):
            x, y, w, h = f[0:4]
            score = float(f[4])
            x1 = int(x * sx)
            y1 = int(y * sy)
            x2 = int((x + w) * sx)
            y2 = int((y + h) * sy)
            boxes.append([x1, y1, x2, y2])
            confs.append(score)
        return boxes, confs


# ===================== 跟踪器（检测间隔帧更稳） =====================
class FaceTracker:
    def __init__(self, tracker_name: str = "MOSSE"):
        self.tracker_name = tracker_name.upper()
        self.tracker = None
        self.active = False

    def _create(self):
        # MOSSE 需要 opencv-contrib；没有就回退到 KCF/CSRT(更慢)
        name = self.tracker_name
        if name == "MOSSE" and hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()
        if name == "KCF" and hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()

        # 兼容老 API
        if name == "MOSSE" and hasattr(cv2, "TrackerMOSSE_create"):
            return cv2.TrackerMOSSE_create()
        if name == "KCF" and hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()

        # 最终兜底：CSRT（更稳但更慢）
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            print("[Tracker] Fallback to CSRT (slower).")
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2, "TrackerCSRT_create"):
            print("[Tracker] Fallback to CSRT (slower).")
            return cv2.TrackerCSRT_create()

        raise RuntimeError("No available OpenCV tracker. Install opencv-contrib-python / opencv-contrib-python-headless.")

    @staticmethod
    def _box_xyxy_to_xywh(box: List[int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    @staticmethod
    def _box_xywh_to_xyxy(xywh: Tuple[float, float, float, float]) -> List[int]:
        x, y, w, h = xywh
        return [int(x), int(y), int(x + w), int(y + h)]

    def reset(self):
        self.tracker = None
        self.active = False

    def init(self, frame: np.ndarray, box_xyxy: List[int]):
        self.tracker = self._create()
        self.active = self.tracker.init(frame, self._box_xyxy_to_xywh(box_xyxy))

    def update(self, frame: np.ndarray) -> Optional[List[int]]:
        if not self.active or self.tracker is None:
            return None
        ok, xywh = self.tracker.update(frame)
        if not ok:
            self.reset()
            return None
        return self._box_xywh_to_xyxy(xywh)


# ===================== 可视化 =====================
def draw_barchart(frame, probs, labels, x0, y0=40, bar_w=160, bar_h=18):
    max_p = float(np.max(probs))
    for i, (label, p) in enumerate(zip(labels, probs)):
        y = y0 + i * (bar_h + 5)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (50, 50, 50), -1)
        bar_len = int(bar_w * float(p))
        color = (0, 255, 0) if float(p) == max_p else (100, 180, 250)
        cv2.rectangle(frame, (x0, y), (x0 + bar_len, y + bar_h), color, -1)
        cv2.putText(frame, f"{label} {p:.2f}", (x0 - 135, y + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def clamp_box_xyxy(box: List[int], W: int, H: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1 + 1:
        x2 = min(W, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(H, y1 + 2)
    return [x1, y1, x2, y2]


def expand_square_roi(box: List[int], W: int, H: int, pad_ratio: float) -> List[int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    side = int(max(bw, bh) * (1.0 + pad_ratio))

    x1n = max(0, cx - side // 2)
    y1n = max(0, cy - side // 2)
    x2n = min(W, x1n + side)
    y2n = min(H, y1n + side)
    return [x1n, y1n, x2n, y2n]


# ===================== 主流程 =====================
def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--tflite", type=str, required=True, help="TFLite 模型路径")
    parser.add_argument("--yunet",  type=str, required=True, help="YuNet ONNX 路径")

    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--w", type=int, default=640)
    parser.add_argument("--h", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no_mjpg", action="store_true")

    parser.add_argument("--det_w", type=int, default=320)
    parser.add_argument("--det_h", type=int, default=240)
    parser.add_argument("--detect_every", type=int, default=3)

    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--no_xnnpack", action="store_true")

    parser.add_argument("--score_th", type=float, default=0.9)
    parser.add_argument("--nms_th", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=5000)

    parser.add_argument("--conf_th", type=float, default=0.5)
    parser.add_argument("--smooth_n", type=int, default=10)
    parser.add_argument("--pad_ratio", type=float, default=0.20)
    parser.add_argument("--light", action="store_true")

    parser.add_argument("--tracker", type=str, default="MOSSE", choices=["MOSSE", "KCF"], help="非检测帧使用的跟踪器")

    a = parser.parse_args()
    return PipelineConfig(
        tflite_path=a.tflite,
        yunet_path=a.yunet,
        cam_id=a.cam,
        cam_w=a.w,
        cam_h=a.h,
        cam_fps=a.fps,
        mjpg=(not a.no_mjpg),
        det_w=a.det_w,
        det_h=a.det_h,
        detect_every=a.detect_every,
        score_th=a.score_th,
        nms_th=a.nms_th,
        top_k=a.top_k,
        tflite_threads=a.threads,
        use_xnnpack=(not a.no_xnnpack),
        conf_th=a.conf_th,
        smooth_n=a.smooth_n,
        pad_ratio=a.pad_ratio,
        light=a.light,
        tracker=a.tracker,
    )


def main():
    cfg = parse_args()

    print(f"[Init] Platform: {sys.platform}, TFLite backend: {TFLITE_BACKEND}")
    print(f"[Init] TFLite: {cfg.tflite_path}")
    fer = TFLiteFER(cfg.tflite_path, num_threads=cfg.tflite_threads, try_xnnpack=cfg.use_xnnpack)

    print("[Init] Opening camera ...")
    cam = CameraReader(
        cam_id=cfg.cam_id,
        backend=choose_camera_backend(),
        width=cfg.cam_w,
        height=cfg.cam_h,
        fps=cfg.cam_fps,
        mjpg=cfg.mjpg,
    ).start()

    # 等待第一帧
    ok, frame = False, None
    for _ in range(60):
        ok, frame = cam.read()
        if ok:
            break
        time.sleep(0.02)
    if not ok or frame is None:
        cam.release()
        raise RuntimeError("Camera read failed at start.")

    H, W = frame.shape[:2]
    print(f"[Init] Camera frame: {W}x{H}")
    print(f"[Init] YuNet: {cfg.yunet_path}")
    detector = FaceDetectorYuNet(cfg.yunet_path, (cfg.det_w, cfg.det_h), cfg.score_th, cfg.nms_th, cfg.top_k)

    tracker = FaceTracker(cfg.tracker)
    smooth_queue = deque(maxlen=cfg.smooth_n)

    fpsm = FPSMeter(ema_alpha=0.1)

    last_det_ms = 0.0
    last_cls_ms = 0.0

    frame_idx = 0
    try:
        while True:
            loop_t0 = time.perf_counter()
            ok, frame = cam.read()
            if not ok or frame is None:
                continue
            H, W = frame.shape[:2]

            # -------- 检测 or 跟踪 --------
            boxes: List[List[int]] = []
            do_detect = (frame_idx % max(1, cfg.detect_every) == 0) or (not tracker.active)

            if do_detect:
                det_t0 = time.perf_counter()
                boxes, _ = detector.detect(frame, cfg.det_w, cfg.det_h)
                last_det_ms = (time.perf_counter() - det_t0) * 1000.0

                # 只在“单脸”场景启用跟踪（与 FER 任务更匹配）
                if len(boxes) == 1:
                    b = clamp_box_xyxy(boxes[0], W, H)
                    tracker.init(frame, b)
                else:
                    tracker.reset()
                    smooth_queue.clear()
            else:
                b = tracker.update(frame)
                if b is None:
                    boxes = []
                    smooth_queue.clear()
                else:
                    boxes = [clamp_box_xyxy(b, W, H)]

            # -------- 分类（只对单脸做平滑/显示）--------
            last_cls_ms = 0.0
            if len(boxes) == 1:
                x1, y1, x2, y2 = boxes[0]
                roi_box = expand_square_roi([x1, y1, x2, y2], W, H, cfg.pad_ratio)
                rx1, ry1, rx2, ry2 = roi_box
                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size != 0:
                    cls_t0 = time.perf_counter()
                    x = preprocess_roi(roi)
                    probs = fer.infer_probs(x)
                    last_cls_ms = (time.perf_counter() - cls_t0) * 1000.0

                    smooth_queue.append(probs)
                    probs_mean = np.mean(smooth_queue, axis=0) if len(smooth_queue) else probs

                    cls_id = int(np.argmax(probs_mean))
                    conf = float(probs_mean[cls_id])

                    color = (0, 255, 0) if conf >= cfg.conf_th else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{LABELS[cls_id]} {conf:.2f}",
                                (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                    if not cfg.light:
                        draw_barchart(frame, probs_mean, LABELS, x0=frame.shape[1] - 180, y0=40)

            # -------- HUD --------
            fps = fpsm.tick()
            loop_ms = (time.perf_counter() - loop_t0) * 1000.0
            info1 = f"FPS: {fps:.1f}"
            info2 = f"det_every={cfg.detect_every}  det={last_det_ms:.1f}ms  cls={last_cls_ms:.1f}ms  loop={loop_ms:.1f}ms  tracker={cfg.tracker}"
            cv2.putText(frame, info1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            cv2.putText(frame, info2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1)

            cv2.imshow("FER-Pi5 (YuNet + Tracker + TFLite)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            frame_idx += 1

    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("Bye.")


if __name__ == "__main__":
    main()

#python3 infer_pi_refactor.py \
#  --tflite /path/to/model.tflite \
#  --yunet  /path/to/face_detection_yunet.onnx \
#  --w 640 --h 480 --fps 30 \
#  --det_w 320 --det_h 240 \
#  --detect_every 4 \
#  --threads 6 \
#  --tracker MOSSE