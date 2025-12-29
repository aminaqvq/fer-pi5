import numpy as np
import cv2
import tensorflow as tf
import argparse
from collections import deque
from tqdm import tqdm  # 导入 tqdm

# ===== 配置 =====
TFLITE_PATH = r"D:\fer-pi5\checkpoints\exported\tf_model\model.sim_float32.tflite"
YUNET_PATH = r"D:\fer-pi5\checkpoints\exported\face_detection_yunet_2023mar.onnx"
IMG_SIZE = 224
LABELS = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

CONF_TH = 0.5
SMOOTH_N = 10
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ===== TFLite 推理函数 =====
def preprocess_roi(bgr):
    roi = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = roi.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.expand_dims(x, 0)


def create_tflite(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    print(f"[TFLite] Input shape: {in_det['shape']}, Output shape: {out_det['shape']}")
    return interpreter, in_det, out_det


def tflite_infer(interpreter, in_det, out_det, x_float):
    scale, zp = in_det["quantization"]
    if in_det["dtype"] == np.int8:
        x_q = np.round(x_float / scale + zp).astype(np.int8)
        interpreter.set_tensor(in_det["index"], x_q)
    else:
        interpreter.set_tensor(in_det["index"], x_float.astype(in_det["dtype"]))
    interpreter.invoke()

    yq = interpreter.get_tensor(out_det["index"])
    oscale, ozp = out_det["quantization"]
    if out_det["dtype"] == np.int8:
        y = (yq.astype(np.float32) - ozp) * oscale
    else:
        y = yq.astype(np.float32)

    y = np.reshape(y, (-1,))
    exp_y = np.exp(y - np.max(y))
    probs = exp_y / np.sum(exp_y)
    probs = probs.reshape(-1)

    if probs.size != len(LABELS):
        print(f"[WARN] Output size {probs.size} != {len(LABELS)}")
        cls_id, conf = len(LABELS) - 1, 1.0
    else:
        cls_id = int(np.argmax(probs))
        conf = float(probs[cls_id])

    return cls_id, conf, probs


# ===== YuNet 人脸检测 =====
def create_yunet(model_path, input_size, score_th=0.9, nms_th=0.3, top_k=5000):
    if hasattr(cv2, "FaceDetectorYN_create"):
        return cv2.FaceDetectorYN_create(model_path, "", input_size, score_th, nms_th, top_k)
    else:
        return cv2.FaceDetectorYN.create(model_path, "", input_size, score_th, nms_th, top_k)


def yunet_detect(detector, frame_bgr):
    faces = detector.detect(frame_bgr)
    if isinstance(faces, tuple):
        faces = faces[1]
    if faces is None or len(faces) == 0:
        return [], []
    boxes, confs = [], []
    for f in faces.astype(np.float32):
        x, y, w, h = f[0:4]
        boxes.append([int(x), int(y), int(x + w), int(y + h)])
        confs.append(float(f[4]))
    return boxes, confs


# ===== 绘制条形图 =====
def draw_barchart(frame, probs, labels, x0=450, y0=40, bar_w=160, bar_h=20):
    max_p = np.max(probs)
    for i, (label, p) in enumerate(zip(labels, probs)):
        y = y0 + i * (bar_h + 5)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (50, 50, 50), -1)
        bar_len = int(bar_w * p)
        color = (0, 255, 0) if p == max_p else (100, 180, 250)
        cv2.rectangle(frame, (x0, y), (x0 + bar_len, y + bar_h), color, -1)
        cv2.putText(frame, f"{label} {p:.2f}", (x0 - 130, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ===== 主程序：视频处理 =====
def main():
    # 你的视频输入输出路径（直接修改这里）
    input_path = r"E:\情绪.mp4"
    output_path = r"E:\情绪_结果.mp4"
    light_mode = False  # 设置为 True 则不显示置信度条

    print("[Init] Loading TFLite:", TFLITE_PATH)
    interpreter, in_det, out_det = create_tflite(TFLITE_PATH)

    print("[Init] Loading YuNet:", YUNET_PATH)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("无法打开视频文件！")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    yunet = create_yunet(YUNET_PATH, (w, h))
    smooth_queue = deque(maxlen=SMOOTH_N)

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 使用 tqdm 进度条显示
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            yunet.setInputSize((w, h))
            det_boxes, _ = yunet_detect(yunet, frame)
            if len(det_boxes) != 1:
                smooth_queue.clear()

            for box in det_boxes:
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                pad_ratio = 0.2
                side = int(max(bw, bh) * (1.0 + pad_ratio))
                x1n = max(0, cx - side // 2)
                y1n = max(0, cy - side // 2)
                x2n = min(w, x1n + side)
                y2n = min(h, y1n + side)
                roi = frame[y1n:y2n, x1n:x2n]
                if roi.size == 0:
                    continue

                x = preprocess_roi(roi)
                cls_id, conf, probs = tflite_infer(interpreter, in_det, out_det, x)
                smooth_queue.append(probs)
                probs_mean = np.mean(smooth_queue, axis=0)
                cls_id_smooth = int(np.argmax(probs_mean))
                conf_smooth = float(probs_mean[cls_id_smooth])

                color = (0, 255, 0) if conf_smooth >= CONF_TH else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_txt = f"{LABELS[cls_id_smooth]} {conf_smooth:.2f}"
                cv2.putText(frame, label_txt, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if not light_mode:
                    draw_barchart(frame, probs_mean, LABELS, x0=frame.shape[1] - 180, y0=40)

            writer.write(frame)

            # 更新进度条
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"[完成] 视频处理结束，已保存至：{output_path}")


if __name__ == "__main__":
    main()