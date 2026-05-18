import json
import os
import shutil
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import tensorflow as tf
from onnxsim import simplify


# =============================================================================
# CONFIG
# =============================================================================
CONFIG: Dict[str, Any] = {
    # -------------------------- model / checkpoint ---------------------------
    "device": "cpu",
    "num_classes": 7,
    "model_name": "mobilenetv3_large",
    "ckpt": r"E:\fer-pi5\checkpoints\best_model_stage3.pth",

    # --------------------------- export options ------------------------------
    "outdir": r"E:\fer-pi5\export",
    "opset": 13,
    "img_size": 224,
    "input_layout": "NCHW",  # NCHW or NHWC for PyTorch/ONNX export
    "input_name": "input",
    "output_name": "output",
    "onnx_simplify": True,
    "check_with_ort": True,

    # -------------------------- preprocessing --------------------------------
    "normalize": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "rgb": True,
    "resize_interpolation": "bilinear",

    # -------------------------- csv input ------------------------------------
    # csv_mode:
    #   - "pixels"     : FER2013 style columns like emotion,pixels
    #   - "image_path" : columns like image,label or path,label
    "csv_mode": "pixels",
    "csv_path": r"E:\fer-pi5\data\csv\train.csv",
    "csv_label_col": "emotion",

    # image_path mode
    "csv_image_col": "",
    "csv_base_dir": "",

    # pixels mode
    "csv_pixels_col": "pixels",
    "csv_pixels_sep": " ",
    "csv_pixels_h": 48,
    "csv_pixels_w": 48,
    "csv_gray_to_rgb": True,

    # optional row filters
    "csv_split_col": "",
    "csv_split_value": "",
    "csv_shuffle": False,
    "csv_shuffle_seed": 42,

    # -------------------------- quantization ---------------------------------
    # "float32" / "fp16" / "int8"
    "quant": "fp16",
    "tflite_input_type": "float32",   # float32 / int8 / uint8 (used for INT8 export)
    "tflite_output_type": "float32",  # float32 / int8 / uint8 (used for INT8 export)
    "calib_limit": 1000,
    "verify_limit": 200,

    # --------------------------- cleanup / misc ------------------------------
    "overwrite": True,
    "save_verification_report": True,
    "verification_report_name": "verification_report.json",
}


# =============================================================================
# basic utils
# =============================================================================
def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def remove_path(path: str) -> None:
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


def to_abs(path: str) -> str:
    return str(Path(path).resolve())


def get_device(device_name: str) -> torch.device:
    if str(device_name).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def interpolation_mode(name: str) -> int:
    name = str(name).strip().lower()
    if name == "nearest":
        return Image.NEAREST
    if name == "bicubic":
        return Image.BICUBIC
    return Image.BILINEAR


def tf_dtype_from_name(name: str) -> tf.dtypes.DType:
    name = str(name).lower()
    if name == "int8":
        return tf.int8
    if name == "uint8":
        return tf.uint8
    return tf.float32


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out = OrderedDict()
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


# =============================================================================
# model build / ckpt load
# =============================================================================
def build_model(cfg: Dict[str, Any]) -> nn.Module:
    num_classes = int(cfg["num_classes"])

    import_errors: List[str] = []

    try:
        from model_mbv3 import MobileNetV3Large  # type: ignore
        model = MobileNetV3Large(pretrained=False, num_classes=num_classes)
        log(f"MobileNetV3-large initialized (pretrained=False, num_classes={num_classes})")
        return model
    except Exception as e:
        import_errors.append(f"from model_mbv3 import MobileNetV3Large -> {repr(e)}")

    try:
        from model_mbv3 import mobilenet_v3_large  # type: ignore
        model = mobilenet_v3_large(num_classes=num_classes)
        log(f"mobilenet_v3_large initialized (num_classes={num_classes})")
        return model
    except Exception as e:
        import_errors.append(f"from model_mbv3 import mobilenet_v3_large -> {repr(e)}")

    try:
        from torchvision.models import mobilenet_v3_large  # type: ignore
        model = mobilenet_v3_large(num_classes=num_classes)
        log(f"torchvision mobilenet_v3_large initialized (num_classes={num_classes})")
        return model
    except Exception as e:
        import_errors.append(f"from torchvision.models import mobilenet_v3_large -> {repr(e)}")

    joined = "\n".join(import_errors)
    raise ImportError(
        "Unable to build model. Please adjust build_model() to match your project.\n"
        + joined
    )


def extract_state_dict(loaded: Any) -> Dict[str, Any]:
    if isinstance(loaded, dict):
        preferred_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "ema_state_dict",
        ]
        for key in preferred_keys:
            value = loaded.get(key, None)
            if isinstance(value, dict) and len(value) > 0:
                return value
        if all(isinstance(k, str) for k in loaded.keys()):
            return loaded
    raise ValueError("Unable to extract state_dict from checkpoint.")


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = strip_module_prefix(extract_state_dict(loaded))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log(f"[ckpt] missing keys: {len(missing)}")
    if unexpected:
        log(f"[ckpt] unexpected keys: {len(unexpected)}")
    log(f"[ckpt] loaded: {ckpt_path}")


# =============================================================================
# image / csv loading
# =============================================================================
def load_csv_samples(cfg: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    csv_path = cfg["csv_path"]
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    split_col = str(cfg.get("csv_split_col", "") or "").strip()
    split_value = cfg.get("csv_split_value", "")
    if split_col and split_col in df.columns and split_value != "":
        df = df[df[split_col] == split_value]

    if bool(cfg.get("csv_shuffle", False)):
        df = df.sample(frac=1.0, random_state=int(cfg.get("csv_shuffle_seed", 42))).reset_index(drop=True)

    mode = str(cfg.get("csv_mode", "image_path")).strip().lower()

    if mode == "pixels":
        pixels_col = str(cfg.get("csv_pixels_col", "pixels"))
        label_col = str(cfg.get("csv_label_col", "emotion"))

        if pixels_col not in df.columns:
            raise KeyError(
                f"CSV does not contain pixels column '{pixels_col}'. Actual columns: {list(df.columns)}"
            )

        samples: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            item: Dict[str, Any] = {"pixels": row[pixels_col]}
            if label_col in df.columns:
                item["label"] = int(row[label_col])
            samples.append(item)

    else:
        candidates = [
            str(cfg.get("csv_image_col", "") or "").strip(),
            "image",
            "img",
            "path",
            "filepath",
            "file",
            "filename",
        ]
        candidates = [c for c in candidates if c]

        image_col = None
        for c in candidates:
            if c in df.columns:
                image_col = c
                break

        if image_col is None:
            raise KeyError(
                f"CSV image column not found. Tried {candidates}, actual columns: {list(df.columns)}"
            )

        label_col = str(cfg.get("csv_label_col", "label"))
        base_dir = str(cfg.get("csv_base_dir", "") or "").strip()

        samples = []
        for _, row in df.iterrows():
            rel_path = str(row[image_col]).strip()
            full_path = rel_path
            if base_dir and not os.path.isabs(rel_path):
                full_path = os.path.join(base_dir, rel_path)

            item = {"image_path": full_path}
            if label_col in df.columns:
                item["label"] = int(row[label_col])
            samples.append(item)

    if limit is not None:
        samples = samples[: int(limit)]

    if not samples:
        raise ValueError("No valid samples loaded from CSV.")

    return samples


def decode_pixels_to_pil(pixels_text: Any, cfg: Dict[str, Any]) -> Image.Image:
    sep = str(cfg.get("csv_pixels_sep", " "))
    h = int(cfg.get("csv_pixels_h", 48))
    w = int(cfg.get("csv_pixels_w", 48))

    arr = np.fromstring(str(pixels_text), sep=sep, dtype=np.float32)
    expected = h * w
    if arr.size != expected:
        raise ValueError(f"Invalid pixels length: expected {expected}, got {arr.size}")

    arr = arr.reshape(h, w)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")

    if bool(cfg.get("csv_gray_to_rgb", True)):
        img = img.convert("RGB")
    return img


def load_sample_as_pil(sample: Dict[str, Any], cfg: Dict[str, Any]) -> Image.Image:
    mode = str(cfg.get("csv_mode", "image_path")).strip().lower()
    if mode == "pixels":
        return decode_pixels_to_pil(sample["pixels"], cfg)

    image_path = sample["image_path"]
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    img = img.convert("RGB" if bool(cfg.get("rgb", True)) else "L")
    return img


def pil_to_float01(img: Image.Image, cfg: Dict[str, Any]) -> np.ndarray:
    size = int(cfg["img_size"])
    interp = interpolation_mode(str(cfg.get("resize_interpolation", "bilinear")))
    img = img.resize((size, size), interp)

    if bool(cfg.get("rgb", True)):
        img = img.convert("RGB")
    else:
        img = img.convert("L")

    x = np.asarray(img, dtype=np.float32)

    if x.ndim == 2:
        x = np.expand_dims(x, axis=-1)

    x = x / 255.0
    return x


def normalize_hwc_float01(x_hwc: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    x = x_hwc.astype(np.float32)
    if bool(cfg.get("normalize", True)):
        mean = np.asarray(cfg.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(cfg.get("std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape(1, 1, -1)
        x = (x - mean) / std
    return x.astype(np.float32)


def hwc_to_layout_batch(x_hwc: np.ndarray, layout: str) -> np.ndarray:
    layout = str(layout).upper()
    if layout == "NHWC":
        return np.expand_dims(x_hwc, axis=0).astype(np.float32)
    x_chw = np.transpose(x_hwc, (2, 0, 1))
    return np.expand_dims(x_chw, axis=0).astype(np.float32)


def preprocess_for_pytorch_onnx(img: Image.Image, cfg: Dict[str, Any]) -> np.ndarray:
    x_hwc = pil_to_float01(img, cfg)
    x_hwc = normalize_hwc_float01(x_hwc, cfg)
    return hwc_to_layout_batch(x_hwc, str(cfg.get("input_layout", "NCHW")))


def preprocess_for_savedmodel_layout(img: Image.Image, cfg: Dict[str, Any], layout: str) -> np.ndarray:
    x_hwc = pil_to_float01(img, cfg)
    x_hwc = normalize_hwc_float01(x_hwc, cfg)
    return hwc_to_layout_batch(x_hwc, layout)


def preprocess_for_int8_calibration(img: Image.Image, cfg: Dict[str, Any], layout: str) -> np.ndarray:
    x_hwc = pil_to_float01(img, cfg)
    x_hwc = normalize_hwc_float01(x_hwc, cfg)
    return hwc_to_layout_batch(x_hwc, layout)


# =============================================================================
# export / onnx / tf / tflite
# =============================================================================
def export_onnx(model: nn.Module, onnx_path: str, cfg: Dict[str, Any]) -> str:
    model.eval()
    device = next(model.parameters()).device

    img_size = int(cfg["img_size"])
    input_layout = str(cfg.get("input_layout", "NCHW")).upper()

    if input_layout == "NHWC":
        dummy = torch.randn(1, img_size, img_size, 3, dtype=torch.float32, device=device)
    else:
        dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32, device=device)

    input_name = str(cfg.get("input_name", "input"))
    output_name = str(cfg.get("output_name", "output"))

    log(f"[onnx] exporting to: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
            opset_version=int(cfg.get("opset", 13)),
            do_constant_folding=True,
        )
    log("[onnx] export done.")
    return onnx_path


def simplify_onnx_model(onnx_path: str) -> str:
    simplified_path = str(Path(onnx_path).with_name(Path(onnx_path).stem + "_simplified.onnx"))
    log("[onnxsim] simplifying ...")
    model = onnx.load(onnx_path)
    model_simplified, ok = simplify(model)
    if not ok:
        raise RuntimeError("onnxsim simplify failed.")
    onnx.save(model_simplified, simplified_path)
    log(f"[onnxsim] saved: {simplified_path}")
    return simplified_path


def ort_quick_check(onnx_path: str, sample_input: np.ndarray) -> None:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: sample_input.astype(np.float32)})
    arr = np.asarray(outputs[0])
    log(f"[ort] ok. output shape: {arr.shape}, dtype: {arr.dtype}")


def convert_onnx_to_saved_model(onnx_path: str, saved_model_dir: str, cfg: Dict[str, Any]) -> str:
    try:
        import onnx2tf
    except Exception as e:
        raise ImportError(f"Failed to import onnx2tf: {e}")

    if os.path.isdir(saved_model_dir) and bool(cfg.get("overwrite", True)):
        shutil.rmtree(saved_model_dir)

    log(f"[onnx2tf] converting ONNX -> SavedModel: {saved_model_dir}")

    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
        output_signaturedefs=True,
    )

    pb = Path(saved_model_dir) / "saved_model.pb"
    if not pb.exists():
        raise FileNotFoundError(
            f"onnx2tf finished but SavedModel was not created: {pb}"
        )

    log("[onnx2tf] SavedModel export done.")
    return saved_model_dir


def get_saved_model_single_input_layout(saved_model_dir: str) -> Tuple[str, str]:
    loaded = tf.saved_model.load(saved_model_dir)
    if "serving_default" not in loaded.signatures:
        raise RuntimeError("SavedModel does not contain 'serving_default' signature.")

    fn = loaded.signatures["serving_default"]
    _, kw = fn.structured_input_signature
    if len(kw) != 1:
        raise RuntimeError(f"Expected single input SavedModel, got {list(kw.keys())}")

    input_name = list(kw.keys())[0]
    spec = kw[input_name]
    shape = spec.shape

    layout = "NHWC"
    if shape.rank == 4:
        dims = list(shape)
        if dims[1] in (1, 3) and dims[-1] not in (1, 3):
            layout = "NCHW"
        elif dims[-1] in (1, 3):
            layout = "NHWC"
    return input_name, layout


def make_tf_representative_dataset(cfg: Dict[str, Any], layout: str):
    samples = load_csv_samples(cfg, limit=int(cfg["calib_limit"]))

    def wrapper(input_key: str):
        def _generator():
            for sample in samples:
                img = load_sample_as_pil(sample, cfg)
                x = preprocess_for_int8_calibration(img, cfg, layout).astype(np.float32)
                yield {input_key: x}
        return _generator()

    return wrapper


def convert_saved_model_to_tflite(saved_model_dir: str, outdir: str, cfg: Dict[str, Any]) -> str:
    quant = str(cfg.get("quant", "float32")).lower()
    input_name, saved_layout = get_saved_model_single_input_layout(saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quant == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quant == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        input_type_name = str(cfg.get("tflite_input_type", "float32")).lower()
        output_type_name = str(cfg.get("tflite_output_type", "float32")).lower()
        converter.inference_input_type = tf_dtype_from_name(input_type_name)
        converter.inference_output_type = tf_dtype_from_name(output_type_name)

        rep_factory = make_tf_representative_dataset(cfg, saved_layout)

        def representative_dataset():
            for item in rep_factory(input_name):
                yield item

        converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    name_map = {
        "float32": "model_float32.tflite",
        "fp16": "model_fp16.tflite",
        "int8": "model_int8.tflite",
    }
    tflite_path = os.path.join(outdir, name_map.get(quant, "model_float32.tflite"))
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    log(f"[tflite] saved: {tflite_path}")
    return tflite_path


# =============================================================================
# inference / verification
# =============================================================================
def run_pytorch(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        xt = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        y = model(xt)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if isinstance(y, dict):
            y = next(iter(y.values()))
        return y.detach().cpu().numpy()


def run_onnx(onnx_path: str, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    y = sess.run(None, {input_name: x.astype(np.float32)})[0]
    return np.asarray(y)


def load_tflite_interpreter(tflite_path: str) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


def infer_layout_from_shape(shape: List[int]) -> str:
    if len(shape) != 4:
        return "NHWC"
    if shape[1] in (1, 3) and shape[-1] not in (1, 3):
        return "NCHW"
    return "NHWC"


def quantize_tensor_if_needed(x_float32: np.ndarray, detail: Dict[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    quant = detail.get("quantization", (0.0, 0))
    scale, zero_point = quant

    if dtype == np.float32 or scale in (0.0, None):
        return x_float32.astype(np.float32)

    q = np.round(x_float32 / scale + zero_point)
    if dtype == np.int8:
        q = np.clip(q, -128, 127).astype(np.int8)
    elif dtype == np.uint8:
        q = np.clip(q, 0, 255).astype(np.uint8)
    else:
        q = q.astype(dtype)
    return q


def dequantize_tensor_if_needed(y: np.ndarray, detail: Dict[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    quant = detail.get("quantization", (0.0, 0))
    scale, zero_point = quant

    if dtype == np.float32 or scale in (0.0, None):
        return y.astype(np.float32)

    return (y.astype(np.float32) - float(zero_point)) * float(scale)


def run_tflite(tflite_path: str, x_float32: np.ndarray) -> np.ndarray:
    interpreter = load_tflite_interpreter(tflite_path)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    x_feed = quantize_tensor_if_needed(x_float32, input_detail)
    interpreter.set_tensor(input_detail["index"], x_feed)
    interpreter.invoke()
    y = interpreter.get_tensor(output_detail["index"])
    y = dequantize_tensor_if_needed(y, output_detail)
    return y.astype(np.float32)


def argmax_safe(y: np.ndarray) -> Optional[int]:
    arr = np.asarray(y)
    if arr.size == 0:
        return None
    return int(np.argmax(arr.reshape(-1)))


def verification_report_path(cfg: Dict[str, Any]) -> str:
    return os.path.join(cfg["outdir"], str(cfg.get("verification_report_name", "verification_report.json")))


def verify_with_csv(model: nn.Module, onnx_path: str, tflite_path: str, cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    limit = int(cfg.get("verify_limit", 50))
    samples = load_csv_samples(cfg, limit=limit)

    tflite_interpreter = load_tflite_interpreter(tflite_path)
    tflite_layout = infer_layout_from_shape(list(tflite_interpreter.get_input_details()[0]["shape"]))

    records: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        img = load_sample_as_pil(sample, cfg)

        x_pt = preprocess_for_pytorch_onnx(img, cfg)
        x_tfl = preprocess_for_savedmodel_layout(img, cfg, tflite_layout)

        y_pt = run_pytorch(model, x_pt, device)
        y_ox = run_onnx(onnx_path, x_pt)
        y_tf = run_tflite(tflite_path, x_tfl)

        pt_flat = y_pt.reshape(-1).astype(np.float32)
        ox_flat = y_ox.reshape(-1).astype(np.float32)
        tf_flat = y_tf.reshape(-1).astype(np.float32)

        rec: Dict[str, Any] = {
            "index": idx,
            "label": sample.get("label", None),
            "pytorch_pred": argmax_safe(pt_flat),
            "onnx_pred": argmax_safe(ox_flat),
            "tflite_pred": argmax_safe(tf_flat),
            "pytorch_onnx_max_abs_diff": float(np.max(np.abs(pt_flat - ox_flat))),
            "pytorch_tflite_max_abs_diff": float(np.max(np.abs(pt_flat - tf_flat))),
            "onnx_tflite_max_abs_diff": float(np.max(np.abs(ox_flat - tf_flat))),
            "pytorch_onnx_mean_abs_diff": float(np.mean(np.abs(pt_flat - ox_flat))),
            "pytorch_tflite_mean_abs_diff": float(np.mean(np.abs(pt_flat - tf_flat))),
            "onnx_tflite_mean_abs_diff": float(np.mean(np.abs(ox_flat - tf_flat))),
        }

        label = sample.get("label", None)
        if label is not None:
            rec["pytorch_correct"] = int(rec["pytorch_pred"] == label)
            rec["onnx_correct"] = int(rec["onnx_pred"] == label)
            rec["tflite_correct"] = int(rec["tflite_pred"] == label)

        records.append(rec)

    def avg(key: str) -> float:
        vals = [float(r[key]) for r in records]
        return float(np.mean(vals)) if vals else 0.0

    summary: Dict[str, Any] = {
        "num_samples": len(records),
        "avg_pytorch_onnx_max_abs_diff": avg("pytorch_onnx_max_abs_diff"),
        "avg_pytorch_tflite_max_abs_diff": avg("pytorch_tflite_max_abs_diff"),
        "avg_onnx_tflite_max_abs_diff": avg("onnx_tflite_max_abs_diff"),
        "avg_pytorch_onnx_mean_abs_diff": avg("pytorch_onnx_mean_abs_diff"),
        "avg_pytorch_tflite_mean_abs_diff": avg("pytorch_tflite_mean_abs_diff"),
        "avg_onnx_tflite_mean_abs_diff": avg("onnx_tflite_mean_abs_diff"),
    }

    if records and records[0].get("label", None) is not None:
        for key in ["pytorch_correct", "onnx_correct", "tflite_correct"]:
            summary[key.replace("_correct", "_acc")] = float(np.mean([r[key] for r in records]))

    report = {
        "config": {
            "quant": cfg.get("quant"),
            "img_size": cfg.get("img_size"),
            "input_layout": cfg.get("input_layout"),
            "csv_mode": cfg.get("csv_mode"),
            "verify_limit": cfg.get("verify_limit"),
        },
        "summary": summary,
        "records": records,
    }

    if bool(cfg.get("save_verification_report", True)):
        path = verification_report_path(cfg)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log(f"[verify] report saved: {path}")

    return report


# =============================================================================
# main
# =============================================================================
def main() -> None:
    cfg = dict(CONFIG)
    cfg["outdir"] = to_abs(cfg["outdir"])
    ensure_dir(cfg["outdir"])

    device = get_device(str(cfg.get("device", "cpu")))
    log(f"[env] device = {device.type}")

    model = build_model(cfg)
    model.to(device)
    model.eval()

    load_checkpoint(model, str(cfg["ckpt"]), device)

    outdir = str(cfg["outdir"])
    onnx_path = os.path.join(outdir, "model.onnx")
    saved_model_dir = os.path.join(outdir, "saved_model")

    if bool(cfg.get("overwrite", True)):
        remove_path(onnx_path)
        remove_path(str(Path(onnx_path).with_name("model_simplified.onnx")))
        remove_path(saved_model_dir)
        remove_path(os.path.join(outdir, "model_float32.tflite"))
        remove_path(os.path.join(outdir, "model_fp16.tflite"))
        remove_path(os.path.join(outdir, "model_int8.tflite"))
        remove_path(verification_report_path(cfg))

    export_onnx(model, onnx_path, cfg)
    final_onnx_path = onnx_path

    if bool(cfg.get("onnx_simplify", True)):
        final_onnx_path = simplify_onnx_model(onnx_path)

    csv_samples = load_csv_samples(cfg, limit=1)
    first_img = load_sample_as_pil(csv_samples[0], cfg)
    first_input = preprocess_for_pytorch_onnx(first_img, cfg)

    if bool(cfg.get("check_with_ort", True)):
        ort_quick_check(final_onnx_path, first_input)

    convert_onnx_to_saved_model(final_onnx_path, saved_model_dir, cfg)
    tflite_path = convert_saved_model_to_tflite(saved_model_dir, outdir, cfg)

    report = verify_with_csv(model, final_onnx_path, tflite_path, cfg, device)

    log("[done] export finished.")
    log(f"[done] ONNX    : {final_onnx_path}")
    log(f"[done] TF      : {saved_model_dir}")
    log(f"[done] TFLite  : {tflite_path}")
    log(f"[done] Summary : {json.dumps(report['summary'], ensure_ascii=False)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
