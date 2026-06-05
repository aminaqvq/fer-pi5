from __future__ import annotations

import argparse
import csv
import datetime as dt
import inspect
import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from dataset import FER2013Hybrid, IMG_SIZE
    from metrics import LABELS, NUM_CLASSES
    from model_mbv3 import get_model, load_checkpoint_into_model
except Exception as exc:
    print("\n[export_final_model] Import failed.")
    print(f"Current file : {__file__}")
    print(f"Current dir  : {THIS_DIR}")
    print("Expected     : dataset.py, metrics.py, model_mbv3.py in the same directory.")
    print(f"Original err : {type(exc).__name__}: {exc}")
    raise


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "project_root": os.environ.get("FER_PROJECT_ROOT", r"D:\fer-pi5"),

    # Final selected model: Stage2 balanced clean historical best.
    "ckpt_candidates": [
        r"checkpoints\best_model_stage2_efficientnet_b0_balanced_clean.pth",
        r"checkpoints\best_model_stage3_efficientnet_b0_final.pth",
    ],
    "ckpt": None,

    "model_variant": "efficientnet_b0",
    "num_classes": 7,
    "img_size": 224,
    "device": "cpu",
    "pretrained": False,
    "repvgg_deploy_convert": False,
    "strict_checkpoint_load": True,

    # Export layout. Keep NCHW because MobileNetV3 PyTorch model expects NCHW.
    "input_name": "input",
    "output_name": "logits",
    "input_layout": "NCHW",
    "opset": 18,
    "try_dynamo_export": True,
    "dynamic_batch": False,
    "onnx_simplify": True,
    "onnx_check": True,

    # Conversion.
    "convert_tf": True,
    "quant": "fp16",  # float32 / fp16 / int8
    "calib_limit": 500,

    # Verification.
    "verify_csv": r"data\csv\test.csv",
    "img_base": None,
    "verify_split": "test",
    "verify_limit": 300,
    "verify_stride": 1,

    # Output.
    "outdir": r"export\final_efficientnet_b0",
    "artifact_stem": "fer_efficientnet_b0_stage2_final",
    "overwrite": True,

    # Safety gates.
    "max_allowed_onnx_diff": 1e-4,
    "max_allowed_tflite_diff": 5e-2,
    "fail_on_large_diff": False,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    print(str(message), flush=True)


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def is_none_like(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "None", "none", "null"}


def resolve_path(root: Path, value: Any) -> Optional[Path]:
    if is_none_like(value):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def resolve_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    root = Path(str(out["project_root"])).expanduser().resolve()
    out["project_root"] = str(root)

    out["outdir"] = str(resolve_path(root, out["outdir"]))
    out["verify_csv"] = str(resolve_path(root, out["verify_csv"]))
    out["img_base"] = None if is_none_like(out.get("img_base")) else str(resolve_path(root, out["img_base"]))

    if not is_none_like(out.get("ckpt")):
        out["ckpt"] = str(resolve_path(root, out["ckpt"]))

    candidates: List[str] = []
    for item in out.get("ckpt_candidates", []):
        resolved = resolve_path(root, item)
        if resolved is not None:
            candidates.append(str(resolved))
    out["ckpt_candidates"] = candidates
    return out


def choose_checkpoint(cfg: Mapping[str, Any]) -> Path:
    if not is_none_like(cfg.get("ckpt")):
        path = Path(str(cfg["ckpt"]))
        if not path.exists():
            raise FileNotFoundError(f"Explicit checkpoint not found: {path}")
        return path

    checked: List[str] = []
    for item in cfg.get("ckpt_candidates", []):
        path = Path(str(item))
        checked.append(str(path))
        if path.exists():
            return path

    raise FileNotFoundError("No final checkpoint found. Checked:\n" + "\n".join(checked))


def safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    tmp.replace(path)


def file_info(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 4),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_and_load_model(cfg: Mapping[str, Any], ckpt_path: Path) -> nn.Module:
    device = torch.device(str(cfg.get("device", "cpu")))
    model = get_model(
        variant=str(cfg.get("model_variant", "large")),
        num_classes=int(cfg.get("num_classes", NUM_CLASSES)),
        pretrained=bool(cfg.get("pretrained", False)),
        device=device,
        verbose=True,
        compile_model=False,
    )
    load_checkpoint_into_model(
        model,
        ckpt_path,
        device=device,
        strict=bool(cfg.get("strict_checkpoint_load", True)),
    )

    # RepVGGplus: fuse multi-branch training graph into single-branch deploy graph.
    if bool(cfg.get("repvgg_deploy_convert", False)):
        if hasattr(model, "switch_repvggplus_to_deploy"):
            print("[export] converting RepVGGplus training graph to deploy graph", flush=True)
            model.switch_repvggplus_to_deploy()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def make_dummy_input(cfg: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    img_size = int(cfg.get("img_size", IMG_SIZE))
    layout = str(cfg.get("input_layout", "NCHW")).upper()
    if layout != "NCHW":
        raise ValueError("This project exports PyTorch MobileNetV3 in NCHW layout. Use input_layout='NCHW'.")
    return torch.randn(1, 3, img_size, img_size, dtype=torch.float32, device=device)


def export_onnx(model: nn.Module, onnx_path: Path, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    device = next(model.parameters()).device
    dummy = make_dummy_input(cfg, device)
    input_name = str(cfg.get("input_name", "input"))
    output_name = str(cfg.get("output_name", "logits"))
    opset = int(cfg.get("opset", 17))
    dynamic_batch = bool(cfg.get("dynamic_batch", False))

    common_kwargs: Dict[str, Any] = {
        "input_names": [input_name],
        "output_names": [output_name],
        "opset_version": opset,
        "do_constant_folding": True,
    }
    if dynamic_batch:
        common_kwargs["dynamic_axes"] = {input_name: {0: "batch"}, output_name: {0: "batch"}}

    export_meta: Dict[str, Any] = {
        "path": str(onnx_path),
        "opset": opset,
        "input_name": input_name,
        "output_name": output_name,
        "dynamic_batch": dynamic_batch,
        "method": None,
        "fallback_used": False,
    }

    log(f"[onnx] exporting: {onnx_path}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        if bool(cfg.get("try_dynamo_export", True)) and "dynamo" in inspect.signature(torch.onnx.export).parameters:
            try:
                torch.onnx.export(model, dummy, str(onnx_path), dynamo=True, **common_kwargs)
                export_meta["method"] = "torch.onnx.export(dynamo=True)"
                log("[onnx] export done with dynamo=True")
                return export_meta
            except Exception as exc:
                export_meta["fallback_used"] = True
                export_meta["dynamo_error"] = f"{type(exc).__name__}: {exc}"
                log(f"[onnx] dynamo=True failed, falling back to legacy exporter: {exc}")

        torch.onnx.export(model, dummy, str(onnx_path), **common_kwargs)
        export_meta["method"] = "torch.onnx.export(legacy)"
        log("[onnx] export done with legacy exporter")
        return export_meta


def check_onnx_model(onnx_path: Path) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    log("[onnx] checker passed")


def simplify_onnx_model(onnx_path: Path) -> Tuple[Path, Dict[str, Any]]:
    meta = {"enabled": True, "ok": False, "input": str(onnx_path)}
    simplified_path = onnx_path.with_name(onnx_path.stem + "_simplified.onnx")
    try:
        import onnx
        from onnxsim import simplify

        log("[onnxsim] simplifying...")
        model = onnx.load(str(onnx_path))
        simplified, ok = simplify(model)
        if not ok:
            raise RuntimeError("onnxsim returned ok=False")
        onnx.save(simplified, str(simplified_path))
        meta.update({"ok": True, "output": str(simplified_path)})
        log(f"[onnxsim] saved: {simplified_path}")
        return simplified_path, meta
    except Exception as exc:
        meta.update({"ok": False, "error": f"{type(exc).__name__}: {exc}", "output": str(onnx_path)})
        log(f"[onnxsim] skipped/failed, using original ONNX: {exc}")
        return onnx_path, meta


def run_onnx(onnx_path: Path, x_nchw: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: x_nchw.astype(np.float32)})[0]
    return np.asarray(output, dtype=np.float32)


# ---------------------------------------------------------------------------
# TensorFlow / TFLite conversion
# ---------------------------------------------------------------------------
def convert_onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    meta = {"enabled": True, "onnx_path": str(onnx_path), "saved_model_dir": str(saved_model_dir)}
    try:
        import onnx2tf
    except Exception as exc:
        raise ImportError(
            "onnx2tf is required for ONNX -> SavedModel -> TFLite conversion. "
            "Install it or run with --skip-tf. "
            f"Original error: {type(exc).__name__}: {exc}"
        )

    if saved_model_dir.exists() and bool(cfg.get("overwrite", True)):
        shutil.rmtree(saved_model_dir)

    log(f"[onnx2tf] converting to SavedModel: {saved_model_dir}")
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        # onnx2tf >=2.4.0 defaults to flatbuffer_direct, whose SavedModel
        # exporter currently fails on MobileNetV3 HARD_SWISH. Use the legacy
        # TensorFlow converter path because the rest of this script expects
        # a real SavedModel directory and then calls tf.lite.TFLiteConverter.
        tflite_backend="tf_converter",
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
        output_signaturedefs=True,
    )

    pb = saved_model_dir / "saved_model.pb"
    if not pb.exists():
        raise FileNotFoundError(f"onnx2tf did not create SavedModel: {pb}")
    meta["ok"] = True
    return meta


def get_saved_model_input_info(saved_model_dir: Path) -> Tuple[str, str, List[int]]:
    import tensorflow as tf

    loaded = tf.saved_model.load(str(saved_model_dir))
    if "serving_default" not in loaded.signatures:
        raise RuntimeError("SavedModel has no serving_default signature")
    fn = loaded.signatures["serving_default"]
    _args, kwargs = fn.structured_input_signature
    if len(kwargs) != 1:
        raise RuntimeError(f"Expected one SavedModel input, got {list(kwargs.keys())}")
    input_name = list(kwargs.keys())[0]
    shape = [int(x) if x is not None else -1 for x in kwargs[input_name].shape.as_list()]
    layout = infer_layout_from_shape(shape)
    return input_name, layout, shape


def infer_layout_from_shape(shape: Sequence[int]) -> str:
    if len(shape) != 4:
        return "NHWC"
    if shape[1] in (1, 3) and shape[-1] not in (1, 3):
        return "NCHW"
    return "NHWC"


def nchw_to_layout(x_nchw: np.ndarray, layout: str) -> np.ndarray:
    layout = str(layout).upper()
    if layout == "NCHW":
        return x_nchw.astype(np.float32)
    if layout == "NHWC":
        return np.transpose(x_nchw, (0, 2, 3, 1)).astype(np.float32)
    raise ValueError(f"Unsupported layout: {layout}")


def make_representative_dataset(cfg: Mapping[str, Any], input_name: str, layout: str):
    dataset = FER2013Hybrid(
        str(cfg["verify_csv"]),
        None if is_none_like(cfg.get("img_base")) else str(cfg.get("img_base")),
        str(cfg.get("verify_split", "test")),
        img_size=int(cfg.get("img_size", IMG_SIZE)),
        include_label=True,
        strict=True,
    )
    limit = min(int(cfg.get("calib_limit", 500)), len(dataset))

    def generator():
        for idx in range(limit):
            x, _label = dataset[idx]
            x_nchw = x.unsqueeze(0).cpu().numpy().astype(np.float32)
            yield {input_name: nchw_to_layout(x_nchw, layout)}

    return generator


def convert_saved_model_to_tflite(saved_model_dir: Path, tflite_path: Path, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    import tensorflow as tf

    quant = str(cfg.get("quant", "fp16")).lower().strip()
    input_name, layout, shape = get_saved_model_input_info(saved_model_dir)
    meta = {
        "saved_model_dir": str(saved_model_dir),
        "tflite_path": str(tflite_path),
        "quant": quant,
        "saved_model_input_name": input_name,
        "saved_model_input_layout": layout,
        "saved_model_input_shape": shape,
    }

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if quant == "float32":
        pass
    elif quant == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quant == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        converter.representative_dataset = make_representative_dataset(cfg, input_name, layout)
    else:
        raise ValueError("quant must be one of: float32, fp16, int8")

    log(f"[tflite] converting quant={quant} -> {tflite_path}")
    model_bytes = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(model_bytes)
    log(f"[tflite] saved: {tflite_path}")
    meta["ok"] = True
    return meta


def load_tflite_interpreter(tflite_path: Path):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    return interpreter


def quantize_tensor_if_needed(x_float: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if dtype == np.float32 or scale in (0.0, None):
        return x_float.astype(np.float32)
    q = np.round(x_float / float(scale) + float(zero_point))
    if dtype == np.int8:
        return np.clip(q, -128, 127).astype(np.int8)
    if dtype == np.uint8:
        return np.clip(q, 0, 255).astype(np.uint8)
    return q.astype(dtype)


def dequantize_tensor_if_needed(y: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if dtype == np.float32 or scale in (0.0, None):
        return y.astype(np.float32)
    return (y.astype(np.float32) - float(zero_point)) * float(scale)


def run_tflite(tflite_path: Path, x_nchw: np.ndarray) -> np.ndarray:
    interpreter = load_tflite_interpreter(tflite_path)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    layout = infer_layout_from_shape(list(input_detail["shape"]))
    x_float = nchw_to_layout(x_nchw, layout)
    x_feed = quantize_tensor_if_needed(x_float, input_detail)
    interpreter.set_tensor(input_detail["index"], x_feed)
    interpreter.invoke()
    y = interpreter.get_tensor(output_detail["index"])
    return dequantize_tensor_if_needed(y, output_detail).astype(np.float32)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def run_pytorch(model: nn.Module, x_nchw: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(x_nchw).to(device=device, dtype=torch.float32)
        y = model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if isinstance(y, dict):
            y = next(iter(y.values()))
        return y.detach().cpu().numpy().astype(np.float32)


def argmax_int(logits: np.ndarray) -> int:
    return int(np.argmax(np.asarray(logits).reshape(-1)))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.reshape(-1).astype(np.float32) - b.reshape(-1).astype(np.float32))))


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.reshape(-1).astype(np.float32) - b.reshape(-1).astype(np.float32))))


def verify_exports(
    model: nn.Module,
    onnx_path: Path,
    tflite_path: Optional[Path],
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    dataset = FER2013Hybrid(
        str(cfg["verify_csv"]),
        None if is_none_like(cfg.get("img_base")) else str(cfg.get("img_base")),
        str(cfg.get("verify_split", "test")),
        img_size=int(cfg.get("img_size", IMG_SIZE)),
        include_label=True,
        strict=True,
    )
    limit = min(int(cfg.get("verify_limit", 300)), len(dataset))
    stride = max(1, int(cfg.get("verify_stride", 1)))
    indices = list(range(0, len(dataset), stride))[:limit]

    records: List[Dict[str, Any]] = []
    for out_idx, ds_idx in enumerate(indices):
        x_tensor, label_tensor = dataset[ds_idx]
        label = int(label_tensor.item())
        x_nchw = x_tensor.unsqueeze(0).cpu().numpy().astype(np.float32)

        y_pt = run_pytorch(model, x_nchw, device)
        y_onnx = run_onnx(onnx_path, x_nchw)
        y_tflite = run_tflite(tflite_path, x_nchw) if tflite_path is not None else None

        rec: Dict[str, Any] = {
            "verify_index": out_idx,
            "dataset_index": ds_idx,
            "true_label_id": label,
            "true_label": LABELS[label] if 0 <= label < len(LABELS) else str(label),
            "pytorch_pred_id": argmax_int(y_pt),
            "onnx_pred_id": argmax_int(y_onnx),
            "pytorch_onnx_max_abs_diff": max_abs_diff(y_pt, y_onnx),
            "pytorch_onnx_mean_abs_diff": mean_abs_diff(y_pt, y_onnx),
        }
        rec["pytorch_pred"] = LABELS[rec["pytorch_pred_id"]]
        rec["onnx_pred"] = LABELS[rec["onnx_pred_id"]]
        rec["pytorch_correct"] = int(rec["pytorch_pred_id"] == label)
        rec["onnx_correct"] = int(rec["onnx_pred_id"] == label)
        rec["pytorch_onnx_pred_match"] = int(rec["pytorch_pred_id"] == rec["onnx_pred_id"])

        if y_tflite is not None:
            tflite_pred_id = argmax_int(y_tflite)
            rec.update({
                "tflite_pred_id": tflite_pred_id,
                "tflite_pred": LABELS[tflite_pred_id],
                "tflite_correct": int(tflite_pred_id == label),
                "pytorch_tflite_pred_match": int(argmax_int(y_pt) == tflite_pred_id),
                "onnx_tflite_pred_match": int(argmax_int(y_onnx) == tflite_pred_id),
                "pytorch_tflite_max_abs_diff": max_abs_diff(y_pt, y_tflite),
                "pytorch_tflite_mean_abs_diff": mean_abs_diff(y_pt, y_tflite),
                "onnx_tflite_max_abs_diff": max_abs_diff(y_onnx, y_tflite),
                "onnx_tflite_mean_abs_diff": mean_abs_diff(y_onnx, y_tflite),
            })

        records.append(rec)

    def avg(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in records if key in r]
        return float(np.mean(vals)) if vals else None

    def maxv(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in records if key in r]
        return float(np.max(vals)) if vals else None

    summary: Dict[str, Any] = {
        "verify_csv": str(cfg["verify_csv"]),
        "verify_split": str(cfg.get("verify_split", "test")),
        "num_samples": len(records),
        "pytorch_acc": avg("pytorch_correct"),
        "onnx_acc": avg("onnx_correct"),
        "pytorch_onnx_pred_match_rate": avg("pytorch_onnx_pred_match"),
        "pytorch_onnx_max_abs_diff_max": maxv("pytorch_onnx_max_abs_diff"),
        "pytorch_onnx_max_abs_diff_avg": avg("pytorch_onnx_max_abs_diff"),
        "pytorch_onnx_mean_abs_diff_avg": avg("pytorch_onnx_mean_abs_diff"),
    }

    if tflite_path is not None:
        summary.update({
            "tflite_acc": avg("tflite_correct"),
            "pytorch_tflite_pred_match_rate": avg("pytorch_tflite_pred_match"),
            "onnx_tflite_pred_match_rate": avg("onnx_tflite_pred_match"),
            "pytorch_tflite_max_abs_diff_max": maxv("pytorch_tflite_max_abs_diff"),
            "pytorch_tflite_max_abs_diff_avg": avg("pytorch_tflite_max_abs_diff"),
            "pytorch_tflite_mean_abs_diff_avg": avg("pytorch_tflite_mean_abs_diff"),
            "onnx_tflite_max_abs_diff_max": maxv("onnx_tflite_max_abs_diff"),
            "onnx_tflite_max_abs_diff_avg": avg("onnx_tflite_max_abs_diff"),
        })

    return {"summary": summary, "records": records}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def preflight(cfg: Mapping[str, Any], ckpt_path: Path) -> None:
    verify_csv = Path(str(cfg["verify_csv"]))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not verify_csv.exists():
        raise FileNotFoundError(f"verify_csv not found: {verify_csv}")
    if int(cfg.get("num_classes", NUM_CLASSES)) != NUM_CLASSES:
        raise ValueError(f"num_classes mismatch: config={cfg.get('num_classes')} project={NUM_CLASSES}")

    lower = str(ckpt_path).lower()
    if "stage3" in lower:
        log("[warning] checkpoint path contains 'stage3'. Final selected model should normally be Stage2 historical best.")
    if "clean_v2" in lower or "weighted" in lower:
        log("[warning] checkpoint path contains clean_v2/weighted. Confirm this is intentional.")


def build_paths(cfg: Mapping[str, Any]) -> Dict[str, Path]:
    outdir = Path(str(cfg["outdir"]))
    stem = str(cfg.get("artifact_stem", "fer_mbv3_stage2_final"))
    quant = str(cfg.get("quant", "fp16")).lower()
    return {
        "outdir": outdir,
        "onnx": outdir / f"{stem}.onnx",
        "saved_model": outdir / f"{stem}_saved_model",
        "tflite": outdir / f"{stem}_{quant}.tflite",
        "manifest": outdir / f"{stem}_export_manifest.json",
        "verify_report": outdir / f"{stem}_verification_report.json",
        "verify_csv": outdir / f"{stem}_verification_records.csv",
    }


def export_pipeline(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = resolve_config(cfg)
    ckpt_path = choose_checkpoint(cfg)
    paths = build_paths(cfg)
    preflight(cfg, ckpt_path)

    outdir = paths["outdir"]
    if outdir.exists() and bool(cfg.get("overwrite", True)):
        safe_remove(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=== FER final model export ===")
    log(f"project_root: {cfg['project_root']}")
    log(f"checkpoint  : {ckpt_path}")
    log(f"outdir      : {outdir}")
    log(f"quant       : {cfg.get('quant')}")

    model = build_and_load_model(cfg, ckpt_path)
    model.eval()

    export_meta = export_onnx(model, paths["onnx"], cfg)
    check_onnx_model(paths["onnx"])

    final_onnx = paths["onnx"]
    simplify_meta = {"enabled": False}
    if bool(cfg.get("onnx_simplify", True)):
        final_onnx, simplify_meta = simplify_onnx_model(paths["onnx"])
        check_onnx_model(final_onnx)

    tf_meta: Dict[str, Any] = {"enabled": False}
    tflite_path: Optional[Path] = None
    if bool(cfg.get("convert_tf", True)):
        tf_meta = convert_onnx_to_saved_model(final_onnx, paths["saved_model"], cfg)
        tflite_meta = convert_saved_model_to_tflite(paths["saved_model"], paths["tflite"], cfg)
        tf_meta["tflite"] = tflite_meta
        tflite_path = paths["tflite"]

    verification = verify_exports(model, final_onnx, tflite_path, cfg)
    write_json(paths["verify_report"], verification)

    if verification["records"]:
        fields = list(verification["records"][0].keys())
        write_csv_rows(paths["verify_csv"], verification["records"], fields)

    summary = verification["summary"]
    onnx_max = summary.get("pytorch_onnx_max_abs_diff_max")
    tflite_max = summary.get("pytorch_tflite_max_abs_diff_max")

    warnings_list: List[str] = []
    if onnx_max is not None and float(onnx_max) > float(cfg.get("max_allowed_onnx_diff", 1e-4)):
        warnings_list.append(f"ONNX max diff is high: {onnx_max}")
    if tflite_max is not None and float(tflite_max) > float(cfg.get("max_allowed_tflite_diff", 5e-2)):
        warnings_list.append(f"TFLite max diff is high: {tflite_max}")

    manifest = {
        "status": "finished_with_warnings" if warnings_list else "finished",
        "created_at": now_iso(),
        "project_root": cfg["project_root"],
        "final_model_decision": "Use Stage2 balanced clean historical best. Stage3 final did not outperform it.",
        "checkpoint": str(ckpt_path),
        "labels": list(LABELS),
        "config": cfg,
        "artifacts": {
            "onnx_original": file_info(paths["onnx"]),
            "onnx_final": file_info(final_onnx),
            "saved_model": file_info(paths["saved_model"]),
            "tflite": file_info(paths["tflite"]) if tflite_path is not None else {"exists": False},
            "verification_report": file_info(paths["verify_report"]),
            "verification_records_csv": file_info(paths["verify_csv"]),
        },
        "export": export_meta,
        "onnx_simplify": simplify_meta,
        "tensorflow_tflite": tf_meta,
        "verification_summary": summary,
        "warnings": warnings_list,
    }
    write_json(paths["manifest"], manifest)

    log("=== export complete ===")
    log(json.dumps({
        "status": manifest["status"],
        "onnx": manifest["artifacts"]["onnx_final"],
        "tflite": manifest["artifacts"]["tflite"],
        "verification_summary": summary,
        "manifest": str(paths["manifest"]),
    }, indent=2, ensure_ascii=False))

    if warnings_list and bool(cfg.get("fail_on_large_diff", False)):
        raise RuntimeError("Export verification failed safety gates: " + " | ".join(warnings_list))

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final FER MobileNetV3 model to ONNX / TFLite.")
    parser.add_argument("--project-root", type=str, default=None, help=r"Project root. Default: D:\fer_pi or FER_PROJECT_ROOT")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path, relative to project root or absolute")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory, relative to project root or absolute")
    parser.add_argument("--verify-csv", type=str, default=None, help="CSV used for export verification, default data/csv/test.csv")
    parser.add_argument("--verify-limit", type=int, default=None)
    parser.add_argument("--quant", type=str, choices=["float32", "fp16", "int8"], default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--no-dynamo", action="store_true", help="Disable torch.onnx.export(dynamo=True) attempt")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--skip-tf", action="store_true", help="Only export ONNX; skip SavedModel/TFLite")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--fail-on-large-diff", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if args.project_root is not None:
        cfg["project_root"] = args.project_root
    if args.ckpt is not None:
        cfg["ckpt"] = args.ckpt
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if args.verify_csv is not None:
        cfg["verify_csv"] = args.verify_csv
    if args.verify_limit is not None:
        cfg["verify_limit"] = args.verify_limit
    if args.quant is not None:
        cfg["quant"] = args.quant
    if args.device is not None:
        cfg["device"] = args.device
    if args.opset is not None:
        cfg["opset"] = args.opset
    if args.no_dynamo:
        cfg["try_dynamo_export"] = False
    if args.no_simplify:
        cfg["onnx_simplify"] = False
    if args.skip_tf:
        cfg["convert_tf"] = False
    if args.no_overwrite:
        cfg["overwrite"] = False
    if args.fail_on_large_diff:
        cfg["fail_on_large_diff"] = True
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    cfg = config_from_args(args)
    return export_pipeline(cfg)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()