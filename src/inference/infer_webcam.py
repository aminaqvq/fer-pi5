# -*- coding: utf-8 -*-
"""
infer_webcam.py
--------------------------------------------------
实时表情识别（含表情追踪稳定器）
适用于 PC / 树莓派（CPU/GPU 推理）
"""

import sys, os, time, cv2, torch, numpy as np
from collections import deque
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))
from model_mbv3 import get_model

# ============================================================
# 配置区
# ============================================================
MODEL_PATH = r"D:\fer-pi5\checkpoints\best_model_stage2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# ⚠️ 是否允许“粘滞框”（当前帧没检测到时保留上帧边框）
ENABLE_STICKY_BOX = False   # 你要的效果 => False：没脸就不画框
MISS_TTL = 3               # 若 True：最多保留 MISS_TTL 帧

EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}

# ============================================================
# 预处理管线（与训练尽量一致的话，均值/方差可调整为 ImageNet）
# ============================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # infer_webcam.py 里把 normalize 改成：
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))

])

# ============================================================
# 模型加载（清理 module. / n_averaged）
# ============================================================
def load_clean_state_dict(ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    raw_sd = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in raw_sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    cleaned.pop("n_averaged", None)
    use_ema = ckpt.get("use_ema", False)
    arch = ckpt.get("arch", "mbv3-large")
    return cleaned, use_ema, arch

model = get_model("large", num_classes=7, pretrained=False, device=DEVICE)
state_dict, use_ema, arch = load_clean_state_dict(MODEL_PATH, DEVICE)
_ = model.load_state_dict(state_dict, strict=False)
model.to(DEVICE).eval()
print(f"✅ MobileNetV3 模型加载完成 ({DEVICE}) | arch={arch} | EMA={use_ema}")

# ============================================================
# 人脸检测器初始化
# ============================================================
FACE_MODEL = "face_detection_yunet_2023mar.onnx"
use_yunet = False

yunet_path = os.path.join(os.path.dirname(__file__), FACE_MODEL)
if os.path.exists(yunet_path):
    try:
        face_detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 240))
        use_yunet = True
        print("✅ 使用 YuNet 检测器")
    except Exception as e:
        print(f"⚠️ YuNet 加载失败：{e}\n👉 切换到 Haar 检测器")
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
else:
    print("⚠️ 未找到 YuNet 模型文件，使用 Haar 检测器")
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ============================================================
# 表情追踪稳定器（核心逻辑）
# ============================================================
class EmotionTracker:
    def __init__(self, patience=5, min_conf=0.6):
        self.history = deque(maxlen=patience)
        self.last_emotion = "Neutral"
        self.last_conf = 0.0
        self.min_conf = min_conf

    def update(self, emotion, conf):
        self.history.append(emotion)
        # 出现次数过半 + 置信度足够时更新
        if self.history.count(emotion) >= len(self.history) // 2:
            if conf >= self.min_conf or emotion == self.last_emotion:
                self.last_emotion = emotion
                self.last_conf = conf
        return self.last_emotion, self.last_conf

tracker = EmotionTracker(patience=7, min_conf=0.55)

# ============================================================
# 摄像头初始化
# ============================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头！")
    raise SystemExit

print("🎥 摄像头已启动，按 Q 退出")

# 粘滞框状态（如果开启）
last_faces = []
miss_left = 0

# ============================================================
# 主循环
# ============================================================
prev_time = time.time()
fps = 0

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_draw = frame.copy()
        h, w = frame.shape[:2]

        # ---------- 人脸检测 ----------
        faces = []
        if use_yunet:
            face_detector.setInputSize((w, h))
            # 可选：阈值更稳一些（OpenCV 4.7+ 支持）
            try:
                face_detector.setScoreThreshold(0.85)  # 置信度阈值
                face_detector.setNMSThreshold(0.3)
                face_detector.setTopK(5000)
            except Exception:
                pass

            _, detected = face_detector.detect(frame)
            if detected is not None and len(detected) > 0:
                # detected: [x, y, w, h, score, l0x,l0y, ..., l4x,l4y]
                det = detected.astype(np.float32)
                # 过滤低分与太小的框
                keep = (det[:, 4] >= 0.85) & (det[:, 2] >= 30) & (det[:, 3] >= 30)
                det = det[keep]
                faces = det[:, :4].astype(np.int32).tolist() if len(det) > 0 else []
            else:
                faces = []  # 这一帧没脸 → 不画框
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30)
            )
            faces = detected.astype(np.int32).tolist() if detected is not None and len(detected) > 0 else []

        # ---------- （仅当这一帧检测到）才做推理与绘制 ----------
        for face in faces:
            x, y, fw, fh = map(int, face[:4])
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w, x + fw), min(h, y + fh)
            if x2 <= x or y2 <= y:
                continue

            face_img = frame[y:y2, x:x2]
            if face_img.size == 0:
                continue

            img_tensor = transform(face_img).unsqueeze(0).to(DEVICE)
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_label = int(torch.argmax(probs, dim=1))
            conf = float(probs[0, pred_label])
            emotion = EMOTION_LABELS.get(pred_label, "Unknown")

            stable_emotion, stable_conf = tracker.update(emotion, conf)

            cv2.rectangle(img_draw, (x, y), (x2, y2), (0, 255, 0), 2)
            text = f"{stable_emotion} ({stable_conf * 100:.1f}%)"
            cv2.putText(img_draw, text, (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, (curr_time - prev_time)))
        prev_time = curr_time
        cv2.putText(img_draw, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("FER2013 - Real-time Emotion Recognition", img_draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("🟢 推理结束。")