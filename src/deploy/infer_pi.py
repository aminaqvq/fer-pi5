#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
from collections import deque
import threading

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


# ===== 默认配置（按需改路径）=====
DEFAULT_TFLITE_PATH = r"/home/amina/workspaces/fer-pi5/checkpoints/exported/tf_model/model.sim_float16.tflite"
DEFAULT_YUNET_PATH  = r"/home/amina/workspaces/fer-pi5/checkpoints/exported/face_detection_yunet_2023mar.onnx"

IMG_SIZE = 224
LABELS = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ===================== 摄像头读取线程（提升稳定帧率） =====================
class CameraReader:
    def __init__(self, cam_id: int, backend: int, width: int, height: int, fps: int, mjpg: bool = True):
        self.cap = cv2.VideoCapture(cam_id, backend)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera.")

        # 强烈建议 USB 摄像头使用 MJPG
        if mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,         fps)

        self.frame = None
        self.ok = False
        self.lock = threading.Lock()
        self.stop_flag = False
        self.th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.th.start()
        return self

    def _loop(self):
        while not self.stop_flag:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok = ok
                if ok:
                    self.frame = frame
            # 小睡一下避免线程占满（一般 0~1ms 都行）
            time.sleep(0.001)

    def read(self):
        with self.lock:
            if not self.ok or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.stop_flag = True
        try:
            self.th.join(timeout=1.0)
        except Exception:
            pass
        self.cap.release()


def choose_camera_backend():
    # Windows 用 DSHOW；Linux/Pi 用 V4L2
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_V4L2


# ===================== TFLite 推理 =====================
def preprocess_roi(bgr: np.ndarray) -> np.ndarray:
    roi = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = roi.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.expand_dims(x, 0)


def create_tflite(model_path: str, num_threads: int = 4, try_xnnpack: bool = True):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    kwargs = {"model_path": model_path, "num_threads": num_threads}

    # 尝试启用 XNNPACK delegate（可用则加速；不可用就忽略）
    if try_xnnpack:
        try:
            delegate = tflite.load_delegate("libtensorflowlite_delegate_xnnpack.so")
            kwargs["experimental_delegates"] = [delegate]
            print("[TFLite] XNNPACK delegate enabled.")
        except Exception as e:
            print(f"[TFLite] XNNPACK not available: {e}")

    interpreter = tflite.Interpreter(**kwargs)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    print(f"[TFLite:{TFLITE_BACKEND}] Input:  shape={in_det['shape']}, dtype={in_det['dtype']}, quant={in_det.get('quantization')}")
    print(f"[TFLite:{TFLITE_BACKEND}] Output: shape={out_det['shape']}, dtype={out_det['dtype']}, quant={out_det.get('quantization')}")
    return interpreter, in_det, out_det


def tflite_infer(interpreter, in_det, out_det, x_float: np.ndarray):
    scale, zp = in_det.get("quantization", (0.0, 0))

    if in_det["dtype"] == np.int8:
        if scale == 0:
            raise ValueError("Input int8 but quant scale is 0.")
        x_q = np.round(x_float / scale + zp).astype(np.int8)
        interpreter.set_tensor(in_det["index"], x_q)
    else:
        interpreter.set_tensor(in_det["index"], x_float.astype(in_det["dtype"]))

    interpreter.invoke()

    yq = interpreter.get_tensor(out_det["index"])
    oscale, ozp = out_det.get("quantization", (0.0, 0))

    if out_det["dtype"] == np.int8:
        if oscale == 0:
            raise ValueError("Output int8 but quant scale is 0.")
        y = (yq.astype(np.float32) - ozp) * oscale
    else:
        y = yq.astype(np.float32)

    y = np.reshape(y, (-1,))
    exp_y = np.exp(y - np.max(y))
    probs = exp_y / np.sum(exp_y)
    probs = probs.reshape(-1)

    if probs.size != len(LABELS):
        cls_id, conf = len(LABELS) - 1, 1.0
    else:
        cls_id = int(np.argmax(probs))
        conf = float(probs[cls_id])
    return cls_id, conf, probs


# ===================== YuNet 人脸检测（小图加速版） =====================
def create_yunet(model_path: str, input_size, score_th=0.9, nms_th=0.3, top_k=5000):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YuNet model not found: {model_path}")

    if hasattr(cv2, "FaceDetectorYN_create"):
        det = cv2.FaceDetectorYN_create(model_path, "", input_size, score_th, nms_th, top_k)
    else:
        det = cv2.FaceDetectorYN.create(model_path, "", input_size, score_th, nms_th, top_k)
    return det


def yunet_detect_fast(detector, frame_bgr: np.ndarray, det_w: int, det_h: int):
    H, W = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_LINEAR)

    detector.setInputSize((det_w, det_h))
    faces = detector.detect(small)
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


# ===================== 画条形图 & 文本 =====================
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


# ===================== 主程序 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", type=str, default=DEFAULT_TFLITE_PATH, help="TFLite 模型路径")
    parser.add_argument("--yunet",  type=str, default=DEFAULT_YUNET_PATH,  help="YuNet ONNX 路径")

    # Pi5 + USB 摄像头推荐默认：640x480@30 + MJPG
    parser.add_argument("--cam", type=int, default=0, help="摄像头ID")
    parser.add_argument("--w", type=int, default=640, help="摄像头宽")
    parser.add_argument("--h", type=int, default=480, help="摄像头高")
    parser.add_argument("--fps", type=int, default=30, help="摄像头FPS")
    parser.add_argument("--no_mjpg", action="store_true", help="禁用 MJPG（不推荐）")

    # 检测小图（核心提速）
    parser.add_argument("--det_w", type=int, default=320, help="检测用缩小宽")
    parser.add_argument("--det_h", type=int, default=240, help="检测用缩小高")

    # 每 N 帧检测一次（核心提速）
    parser.add_argument("--detect_every", type=int, default=3, help="每N帧做一次人脸检测（建议 2~4）")

    # TFLite
    parser.add_argument("--threads", type=int, default=6, help="TFLite 线程数（Pi5 建议 4~6）")
    parser.add_argument("--no_xnnpack", action="store_true", help="禁用 XNNPACK delegate")

    # YuNet 参数
    parser.add_argument("--score_th", type=float, default=0.9)
    parser.add_argument("--nms_th", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=5000)

    # 分类输出/显示
    parser.add_argument("--conf_th", type=float, default=0.5)
    parser.add_argument("--smooth_n", type=int, default=10)
    parser.add_argument("--light", action="store_true", help="轻量模式（关闭置信度条）")

    args = parser.parse_args()

    print(f"[Init] Platform: {sys.platform}, TFLite backend: {TFLITE_BACKEND}")
    print("[Init] Loading TFLite:", args.tflite)
    interpreter, in_det, out_det = create_tflite(
        args.tflite,
        num_threads=args.threads,
        try_xnnpack=(not args.no_xnnpack),
    )

    print("[Init] Opening camera ...")
    backend = choose_camera_backend()
    cam = CameraReader(
        cam_id=args.cam,
        backend=backend,
        width=args.w,
        height=args.h,
        fps=args.fps,
        mjpg=(not args.no_mjpg),
    ).start()

    # 等待拿到第一帧
    for _ in range(50):
        ok, frame = cam.read()
        if ok:
            break
        time.sleep(0.02)
    if not ok:
        cam.release()
        raise RuntimeError("Camera read failed at start.")

    H, W = frame.shape[:2]
    print(f"[Init] Camera frame: {W}x{H}")
    print("[Init] Loading YuNet:", args.yunet)
    yunet = create_yunet(args.yunet, (args.det_w, args.det_h), args.score_th, args.nms_th, args.top_k)

    smooth_queue = deque(maxlen=args.smooth_n)

    # FPS & profiling
    fps_ema = 0.0
    t_prev = time.perf_counter()

    # 记录最近一次检测到的人脸框（用于 detect_every 间隔帧复用）
    last_boxes = []
    last_det_time = 0.0

    frame_idx = 0
    while True:
        loop_t0 = time.perf_counter()
        ok, frame = cam.read()
        if not ok:
            continue

        H, W = frame.shape[:2]

        # --- 按间隔做检测 ---
        do_detect = (frame_idx % max(1, args.detect_every) == 0)
        det_t0 = time.perf_counter()
        if do_detect:
            boxes, confs = yunet_detect_fast(yunet, frame, args.det_w, args.det_h)
            last_boxes = boxes
            last_det_time = time.perf_counter() - det_t0
            # 多脸/无脸时清空平滑（与你原逻辑一致）
            if len(last_boxes) != 1:
                smooth_queue.clear()
        else:
            boxes = last_boxes

        # --- 对检测框做分类 ---
        cls_time = 0.0
        for box in boxes:
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            if bw <= 2 or bh <= 2:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pad_ratio = 0.20
            side = int(max(bw, bh) * (1.0 + pad_ratio))

            x1n = max(0, cx - side // 2)
            y1n = max(0, cy - side // 2)
            x2n = min(W, x1n + side)
            y2n = min(H, y1n + side)

            roi = frame[y1n:y2n, x1n:x2n]
            if roi.size == 0:
                continue

            cls_t0 = time.perf_counter()
            x = preprocess_roi(roi)
            _, _, probs = tflite_infer(interpreter, in_det, out_det, x)
            cls_time += (time.perf_counter() - cls_t0)

            smooth_queue.append(probs)
            probs_mean = np.mean(smooth_queue, axis=0) if len(smooth_queue) else probs
            cls_id_s = int(np.argmax(probs_mean))
            conf_s = float(probs_mean[cls_id_s])

            color = (0, 255, 0) if conf_s >= args.conf_th else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{LABELS[cls_id_s]} {conf_s:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            if not args.light:
                draw_barchart(frame, probs_mean, LABELS, x0=frame.shape[1] - 180, y0=40)

        # --- FPS 计算（EMA 平滑）---
        t_now = time.perf_counter()
        dt = t_now - t_prev
        t_prev = t_now
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_ema = fps if fps_ema == 0 else (0.9 * fps_ema + 0.1 * fps)

        # --- 叠加信息 ---
        loop_time = time.perf_counter() - loop_t0
        info1 = f"FPS: {fps_ema:.1f}  (target~25)"
        info2 = f"detect_every={args.detect_every}  det={last_det_time*1000:.1f}ms  cls={cls_time*1000:.1f}ms  loop={loop_time*1000:.1f}ms"
        cv2.putText(frame, info1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(frame, info2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1)

        cv2.imshow("FER-Pi5 (YuNet small + TFLite)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        frame_idx += 1

    cam.release()
    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    main()