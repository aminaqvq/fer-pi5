import os
import sys
import time
import math
import threading
from dataclasses import dataclass, field
from typing import Optionasl, Tuple, List, Dict

import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_BACKEND = "tflite-runtime"
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    TFLITE_BACKEND = "tensorflow-lite"


IMG_SIZE = 224
LABELS = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Config:
    tflite_path: str = "/home/amina/workspaces/fer-pi5/export/tf_model/model_simplified_float32.tflite"
    yunet_path: str = "/home/amina/workspaces/fer-pi5/src/inference/face_detection_yunet_2023mar.onnx"

    camera_source = 0   # 0 / 1 / "/dev/video0"
    cam_w: int = 640
    cam_h: int = 480
    cam_fps: int = 30
    use_mjpg: bool = False

    det_w: int = 320
    det_h: int = 240
    detect_every: int = 2
    score_th: float = 0.7
    nms_th: float = 0.3
    top_k: int = 5000

    tflite_threads: int = 4
    conf_th: float = 0.45
    pad_ratio: float = 0.18
    max_faces: int = 8

    target_fps: int = 30

    # 基于五点中心做轻量跟踪
    track_max_missing: int = 10
    track_max_dist: float = 90.0

    save_dir: str = "./best_by_class"
    mirror_flip: bool = True


CFG = Config()


class CameraReader:
    def __init__(self, source, width: int, height: int, fps: int, use_mjpg: bool = False):
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_ANY)
        else:
            self.cap = cv2.VideoCapture(source, cv2.CAP_ANY)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {source}")

        try:
            if use_mjpg:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._lock = threading.Lock()
        self._stop = False
        self._ok = False
        self._frame = None
        self._frame_ts = 0.0
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._th.start()
        return self

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            ts = time.perf_counter()
            with self._lock:
                self._ok = ok
                if ok and frame is not None:
                    self._frame = frame
                    self._frame_ts = ts
            time.sleep(0.0005)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        with self._lock:
            if not self._ok or self._frame is None:
                return False, None, 0.0
            return True, self._frame.copy(), self._frame_ts

    def release(self):
        self._stop = True
        try:
            self._th.join(timeout=1.0)
        except Exception:
            pass
        self.cap.release()


class FPSMeter:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.prev_t = time.perf_counter()
        self.fps_ema = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self.prev_t
        self.prev_t = now
        fps = 1.0 / dt if dt > 0 else 0.0
        if self.fps_ema == 0.0:
            self.fps_ema = fps
        else:
            self.fps_ema = (1.0 - self.alpha) * self.fps_ema + self.alpha * fps
        return self.fps_ema


def preprocess_roi(bgr: np.ndarray, input_shape) -> np.ndarray:
    roi = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    x = roi.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.expand_dims(x, 0)  # NHWC

    target_shape = tuple(int(v) for v in input_shape)
    if x.shape == target_shape:
        return x

    x_hcw = np.transpose(x, (0, 1, 3, 2))  # (1, H, C, W)
    if x_hcw.shape == target_shape:
        return x_hcw

    x_chw = np.transpose(x, (0, 3, 1, 2))  # (1, C, H, W)
    if x_chw.shape == target_shape:
        return x_chw

    raise ValueError(f"Unsupported input shape: got {x.shape}, expected {target_shape}")


class TFLiteFER:
    def __init__(self, model_path: str, num_threads: int = 4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.in_det = self.interpreter.get_input_details()[0]
        self.out_det = self.interpreter.get_output_details()[0]

        print(f"[TFLite:{TFLITE_BACKEND}] Input:  shape={self.in_det['shape']}, dtype={self.in_det['dtype']}, quant={self.in_det.get('quantization')}")
        print(f"[TFLite:{TFLITE_BACKEND}] Output: shape={self.out_det['shape']}, dtype={self.out_det['dtype']}, quant={self.out_det.get('quantization')}")

    def infer(self, x: np.ndarray) -> np.ndarray:
        scale, zp = self.in_det.get("quantization", (0.0, 0))
        if self.in_det["dtype"] == np.int8:
            if scale == 0:
                raise ValueError("Input int8 but quant scale is 0.")
            x_q = np.round(x / scale + zp).astype(np.int8)
            self.interpreter.set_tensor(self.in_det["index"], x_q)
        else:
            self.interpreter.set_tensor(self.in_det["index"], x.astype(self.in_det["dtype"]))

        self.interpreter.invoke()

        yq = self.interpreter.get_tensor(self.out_det["index"])
        oscale, ozp = self.out_det.get("quantization", (0.0, 0))
        if self.out_det["dtype"] == np.int8:
            if oscale == 0:
                raise ValueError("Output int8 but quant scale is 0.")
            y = (yq.astype(np.float32) - ozp) * oscale
        else:
            y = yq.astype(np.float32)

        y = y.reshape(-1)
        exp_y = np.exp(y - np.max(y))
        probs = exp_y / np.sum(exp_y)
        return probs


class FaceDetectorYuNet:
    def __init__(self, model_path: str, input_size: Tuple[int, int], score_th: float, nms_th: float, top_k: int):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found: {model_path}")

        if hasattr(cv2, "FaceDetectorYN_create"):
            self.det = cv2.FaceDetectorYN_create(model_path, "", input_size, score_th, nms_th, top_k)
        else:
            self.det = cv2.FaceDetectorYN.create(model_path, "", input_size, score_th, nms_th, top_k)

    def detect(self, frame_bgr: np.ndarray, det_w: int, det_h: int) -> List[Dict]:
        H, W = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        self.det.setInputSize((det_w, det_h))

        faces = self.det.detect(small)
        if isinstance(faces, tuple):
            faces = faces[1]
        if faces is None or len(faces) == 0:
            return []

        sx = W / det_w
        sy = H / det_h
        out = []

        for f in faces.astype(np.float32):
            x, y, w, h = f[0:4]
            lms = f[4:14].reshape(5, 2)
            score = float(f[14])

            x1 = int(x * sx)
            y1 = int(y * sy)
            x2 = int((x + w) * sx)
            y2 = int((y + h) * sy)

            landmarks = []
            for px, py in lms:
                landmarks.append((int(px * sx), int(py * sy)))

            out.append({
                "box": [x1, y1, x2, y2],
                "landmarks": landmarks,
                "det_conf": score,
            })

        return out


@dataclass
class Track:
    track_id: int
    box: List[int]
    landmarks: List[Tuple[int, int]]
    det_conf: float
    last_seen_frame: int
    label: str = ""
    cls_conf: float = 0.0
    probs: Optional[np.ndarray] = None
    roi_box: Optional[List[int]] = None


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

    def update(self, detections: List[Dict], frame_idx: int) -> List[Track]:
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

        pairs.sort(key=lambda x: x[0])
        for dist, tid, di in pairs:
            if dist > self.max_dist:
                continue
            if tid in trk_used or di in det_used:
                continue
            det = detections[di]
            tr = self.tracks[tid]
            tr.box = det["box"]
            tr.landmarks = det["landmarks"]
            tr.det_conf = det["det_conf"]
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
                det_conf=det["det_conf"],
                last_seen_frame=frame_idx,
            )

        stale_ids = []
        for tid, tr in self.tracks.items():
            if frame_idx - tr.last_seen_frame > self.max_missing:
                stale_ids.append(tid)
        for tid in stale_ids:
            self.tracks.pop(tid, None)

        tracks = [tr for tr in self.tracks.values() if frame_idx - tr.last_seen_frame <= self.max_missing]
        tracks.sort(key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]), reverse=True)
        return tracks

    def get_active(self, frame_idx: int) -> List[Track]:
        tracks = [tr for tr in self.tracks.values() if frame_idx - tr.last_seen_frame <= self.max_missing]
        tracks.sort(key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]), reverse=True)
        return tracks


def clamp_box(box: List[int], W: int, H: int) -> List[int]:
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

    nx1 = max(0, cx - side // 2)
    ny1 = max(0, cy - side // 2)
    nx2 = min(W, nx1 + side)
    ny2 = min(H, ny1 + side)
    return [nx1, ny1, nx2, ny2]


def draw_landmarks(frame: np.ndarray, landmarks: List[Tuple[int, int]]):
    colors = [(0, 255, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 128, 255)]
    for i, (x, y) in enumerate(landmarks):
        color = colors[i % len(colors)]
        cv2.circle(frame, (x, y), 2, color, -1)


def mark_best_if_needed(label: str, conf: float, best_records: Dict[str, Dict], pending_saves: Dict[str, float]):
    prev = best_records.get(label)
    if prev is not None and conf <= prev["conf"]:
        return
    old_pending = pending_saves.get(label)
    if old_pending is None or conf > old_pending:
        pending_saves[label] = conf


def save_annotated_frame(save_dir: str, label: str, conf: float, annotated_frame: np.ndarray, best_records: Dict[str, Dict]):
    if annotated_frame is None or annotated_frame.size == 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"best_{label}.jpg")
    cv2.imwrite(out_path, annotated_frame)
    best_records[label] = {"conf": conf, "path": out_path}
    print(f"[SAVE] {label}: {conf:.4f} -> {out_path}")


def main():
    print(f"[Init] Platform: {sys.platform}, TFLite backend: {TFLITE_BACKEND}")
    print(f"[Init] TFLite: {CFG.tflite_path}")
    print(f"[Init] YuNet : {CFG.yunet_path}")
    print(f"[Init] Camera source: {CFG.camera_source}")

    fer = TFLiteFER(CFG.tflite_path, num_threads=CFG.tflite_threads)
    detector = FaceDetectorYuNet(
        CFG.yunet_path,
        (CFG.det_w, CFG.det_h),
        CFG.score_th,
        CFG.nms_th,
        CFG.top_k,
    )
    tracker = LandmarkTracker(CFG.track_max_missing, CFG.track_max_dist)

    cam = CameraReader(
        CFG.camera_source,
        CFG.cam_w,
        CFG.cam_h,
        CFG.cam_fps,
        CFG.use_mjpg,
    ).start()

    ok, frame, frame_ts = False, None, 0.0
    for i in range(150):
        ok, frame, frame_ts = cam.read()
        if ok and frame is not None:
            if CFG.mirror_flip:
                frame = cv2.flip(frame, 1)
            print(f"[Init] First frame received at try={i + 1}, shape={frame.shape}")
            break
        time.sleep(0.02)

    if not ok or frame is None:
        cam.release()
        raise RuntimeError("Camera read failed at start.")

    fpsm = FPSMeter(alpha=0.10)
    target_frame_time = 1.0 / max(1, CFG.target_fps)

    frame_idx = 0
    last_det_ms = 0.0
    last_cls_ms = 0.0
    latency_ms = 0.0
    best_records: Dict[str, Dict] = {}

    try:
        while True:
            loop_start = time.perf_counter()

            ok, frame, frame_ts = cam.read()
            if not ok or frame is None:
                continue

            if CFG.mirror_flip:
                frame = cv2.flip(frame, 1)

            H, W = frame.shape[:2]
            latency_ms = (time.perf_counter() - frame_ts) * 1000.0

            if frame_idx % CFG.detect_every == 0:
                t0 = time.perf_counter()
                detections = detector.detect(frame, CFG.det_w, CFG.det_h)
                last_det_ms = (time.perf_counter() - t0) * 1000.0
                tracks = tracker.update(detections, frame_idx)
            else:
                tracks = tracker.get_active(frame_idx)

            tracks = tracks[:CFG.max_faces]
            cls_times = []
            pending_saves: Dict[str, float] = {}

            for tr in tracks:
                box = clamp_box(tr.box, W, H)
                rx1, ry1, rx2, ry2 = expand_square_roi(box, W, H, CFG.pad_ratio)
                roi = frame[ry1:ry2, rx1:rx2]
                tr.roi_box = [rx1, ry1, rx2, ry2]
                if roi.size == 0:
                    continue

                t1 = time.perf_counter()
                x = preprocess_roi(roi, fer.in_det["shape"])
                probs = fer.infer(x)
                cls_times.append((time.perf_counter() - t1) * 1000.0)

                cls_id = int(np.argmax(probs))
                conf = float(probs[cls_id])
                tr.probs = probs
                tr.label = LABELS[cls_id] if conf >= CFG.conf_th else "low_conf"
                tr.cls_conf = conf

                if tr.label in LABELS:
                    mark_best_if_needed(tr.label, conf, best_records, pending_saves)

            last_cls_ms = float(np.mean(cls_times)) if cls_times else 0.0

            for tr in tracks:
                x1, y1, x2, y2 = clamp_box(tr.box, W, H)
                color = (0, 255, 0) if tr.cls_conf >= CFG.conf_th else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if tr.landmarks:
                    draw_landmarks(frame, tr.landmarks)

                text = f"ID{tr.track_id} {tr.label} {tr.cls_conf:.2f}"
                cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

                if tr.probs is not None:
                    top_idx = int(np.argmax(tr.probs))
                    top_label = LABELS[top_idx]
                    top_conf = float(tr.probs[top_idx])
                    cv2.putText(frame, f"top={top_label}:{top_conf:.2f}", (x1, min(H - 10, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1)

            fps = fpsm.tick()
            loop_ms = (time.perf_counter() - loop_start) * 1000.0

            cv2.putText(frame, f"FPS: {fps:.1f} / {CFG.target_fps}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(frame, f"Latency: {latency_ms:.1f} ms", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(tracks)} det={last_det_ms:.1f}ms cls(avg)={last_cls_ms:.1f}ms loop={loop_ms:.1f}ms", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1)

            y0 = 106
            for label in LABELS:
                if label in best_records:
                    msg = f"best {label}: {best_records[label]['conf']:.2f}"
                    cv2.putText(frame, msg, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
                    y0 += 18

            for label, conf in pending_saves.items():
                save_annotated_frame(CFG.save_dir, label, conf, frame.copy(), best_records)

            cv2.imshow("FER Multi + YuNet 5-point", frame)

            elapsed = time.perf_counter() - loop_start
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

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