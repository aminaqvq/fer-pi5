#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_onnx_tf.py

将本项目的人脸表情识别模型从 PyTorch 导出为：
  1) ONNX
  2) TensorFlow SavedModel（通过 onnx2tf）
  3) TensorFlow Lite（可选：FP16 / INT8）

面向 Raspberry Pi 5 的默认推荐：
  - INT8（权重+激活）量化 + float32 输入/输出（部署端仍按训练时做 normalize，最省事，通常也很快）
  - 如追求极致吞吐/最低功耗，可切到 int8/uint8 I/O（但推理端需要按量化参数喂数据）

依赖（建议在导出机器上安装）：
  pip install torch onnx onnxsim onnxruntime pillow numpy
  pip install onnx2tf tensorflow==2.15.* tf-keras==2.15.*

注意：
  - onnx2tf 更新很快，如遇到转换失败，优先尝试换一个 onnx2tf 版本（新/旧各试一个）。
"""

import os
import sys
import json
import argparse
import pathlib
from typing import Optional, Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn

from model_mbv3 import get_model  # 项目内模型

# dataset.py 不是必需：只有在你想用 dataloader 做 INT8 校准且没有 rep_dir 时才用到
try:
    from dataset import get_dataloaders_hybrid  # type: ignore
except Exception:
    get_dataloaders_hybrid = None  # type: ignore


# ===================== 默认配置（可按需修改） =====================
CONFIG = {
    "ckpt": r"./checkpoints/best_model_stage2.pth",
    "variant": "large",
    "img_size": 224,
    "num_classes": 7,
    "outdir": r"./export",
    "opset": 17,

    # 导出控制
    "only_onnx": False,
    "onnx_simplify": True,
    "check_with_ort": False,

    # TFLite 量化： "none" / "fp16" / "int8"
    "quant": "int8",

    # TFLite I/O dtype（仅 quant=int8 时生效）： "float32" / "int8" / "uint8"
    # 推荐默认 float32：推理端仍按训练时 normalize，几乎不改推理代码。
    "tflite_io": "float32",

    # onnx2tf 输出目录名
    "tf_out_name": "tf_model",

    # 代表性数据（INT8 校准）
    # 优先使用 rep_dir 下的 jpg/png（更可控；建议放 200~1000 张“真实场景”图）
    "rep_dir": "",
    "rep_limit": 400,            # rep_dir 最多取多少张
    "calib_split": "val",        # 若 rep_dir 为空才尝试 dataloader（可选 train/val/test）
    "calib_samples": 400,        # dataloader 取多少个样本
    "rep_apply_normalize": True, # rep_dir 图像是否按训练 mean/std 做 normalize（一般要 True）

    # 与 dataset.py 一致的均值方差（ImageNet）
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),

    # 输入布局（导出的 ONNX 输入布局）
    # "nchw": PyTorch 默认；"nhwc": 外部输入为 NHWC，脚本会自动包一层 transpose
    # 推荐 "nhwc"：更贴近 TFLite（onnx2tf 也更擅长把 NCHW 转 NHWC）
    "input_layout": "nhwc",

    # 自动安装 TF/onnx2tf 依赖（生产环境建议关掉）
    "auto_install_tf_deps": False,
}


# ===================== 工具函数 =====================
def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_ckpt_into(model: nn.Module, ckpt_path: str, device: str = "cpu") -> None:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    # 兼容：state_dict / 直接 dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # 兼容：带 module. 前缀
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            new_state[nk] = v
        state = new_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log(f"[ckpt] missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        log(f"[ckpt] unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    log(f"[ckpt] loaded: {ckpt_path}")


class NHWCWrapper(nn.Module):
    """让模型接受 NHWC 输入（B,H,W,C），内部转成 NCHW 再喂给原模型。"""
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,H,W,C -> B,C,H,W
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.base(x)


def export_onnx(
    model: nn.Module,
    onnx_path: str,
    img_size: int,
    opset: int,
    input_layout: str,
) -> None:
    model.eval()
    if input_layout.lower() == "nhwc":
        dummy = torch.randn(1, img_size, img_size, 3, dtype=torch.float32)
        input_name = "input"
        dynamic_axes = {input_name: {0: "batch"}, "output": {0: "batch"}}
    else:
        dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
        input_name = "input"
        dynamic_axes = {input_name: {0: "batch"}, "output": {0: "batch"}}

    log(f"[onnx] exporting to: {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=[input_name],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    log("[onnx] export done.")


def simplify_onnx(onnx_path: str, sim_path: str) -> str:
    try:
        import onnx  # noqa
        from onnxsim import simplify  # noqa

        log("[onnxsim] simplifying ...")
        onnx_model = onnx.load(onnx_path)
        model_simplified, check = simplify(onnx_model)
        if check:
            onnx.save(model_simplified, sim_path)
            log(f"[onnxsim] saved: {sim_path}")
            return sim_path
        log("[onnxsim] check failed, keep original onnx.")
    except Exception as e:
        log(f"[onnxsim] failed: {e!r}")
    return onnx_path


def check_with_onnxruntime(onnx_path: str, img_size: int, input_layout: str) -> None:
    try:
        import onnxruntime as ort  # noqa
    except Exception as e:
        log(f"[ort] onnxruntime missing: {e!r}")
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    if input_layout.lower() == "nhwc":
        x = np.random.randn(1, img_size, img_size, 3).astype(np.float32)
    else:
        x = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    y = sess.run(None, {inp.name: x})[0]
    log(f"[ort] ok. output shape: {y.shape}, dtype: {y.dtype}")


def auto_install_tf_deps(need: List[str]) -> None:
    """
    仅用于你本地快速试验。受控环境/离线环境建议关闭。
    """
    try:
        pkgs: List[str] = []
        if "tensorflow" in need:
            pkgs += ["tensorflow==2.15.*", "tf-keras==2.15.*"]
        if "onnx" in need:
            pkgs += ["onnx"]
        if "onnx2tf" in need:
            pkgs += ["onnx2tf"]
        if pkgs:
            log(f"[deps] auto install: {pkgs}")
            os.system(f"{sys.executable} -m pip install " + " ".join(pkgs))
    except Exception as e:
        log(f"[deps] auto install failed: {e!r}")


# ===================== 代表性数据（INT8 校准） =====================
def _normalize_hwc01(a: np.ndarray, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> np.ndarray:
    """
    a: HWC float32 in [0,1]
    return: HWC float32 normalized (x-mean)/std
    """
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    return (a - mean_arr) / std_arr


def make_rep_dataset_from_dir(
    rep_dir: str,
    img_size: int,
    input_layout: str,
    limit: int,
    apply_normalize: bool,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> Optional[Iterable[np.ndarray]]:
    """
    从目录收集 jpg/png 作为代表性数据：
      - 读入 -> RGB -> resize -> HWC float32 [0,1]
      - 若 apply_normalize=True：再做 (x-mean)/std（应与训练一致）
      - 最终按 input_layout 返回 HWC 或 CHW
    """
    try:
        from PIL import Image
    except Exception:
        log("[rep] PIL not installed; pip install pillow")
        return None

    rep_dir = rep_dir.strip()
    if not rep_dir:
        return None

    paths: List[pathlib.Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths += list(pathlib.Path(rep_dir).rglob(ext))
    paths = sorted(paths)
    if not paths:
        log(f"[rep] no images found in {rep_dir}")
        return None

    if limit > 0:
        paths = paths[:limit]

    def gen():
        for p in paths:
            img = Image.open(str(p)).convert("RGB").resize((img_size, img_size))
            a = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
            if apply_normalize:
                a = _normalize_hwc01(a, mean, std)
            if input_layout.lower() == "nchw":
                a = np.transpose(a, (2, 0, 1))  # CHW
            yield a.astype(np.float32)

    log(f"[rep] using rep_dir: {rep_dir} (n={len(paths)}), apply_normalize={apply_normalize}")
    return gen()


def make_rep_dataset_from_loader(
    split: str,
    samples: int,
    input_layout: str,
) -> Optional[Iterable[np.ndarray]]:
    """
    用项目 dataloader 取代表性数据（若 dataset.py 可用）。
    注意：dataset.py 里已经做了 Normalize(mean,std)，所以 xb 通常已经是 “训练输入分布”。
    """
    if get_dataloaders_hybrid is None:
        return None

    try:
        train_loader, val_loader, test_loader, _ = get_dataloaders_hybrid(
            img_size=CONFIG["img_size"],
            batch_size=32,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            dynamic_sampling=False,
            per_class=5000,
            include_unlabeled=False,
            unlabeled_two_views=False,
            prefetch_factor=2,
        )
    except Exception as e:
        log(f"[rep] build dataloader failed: {e!r}")
        return None

    if split == "train":
        loader = train_loader
    elif split == "test":
        loader = test_loader
    else:
        loader = val_loader

    def gen():
        n = 0
        for xb, _ in loader:
            # xb: Bx3xHxW (通常已经 normalize 到训练分布)
            arr = xb.detach().cpu().numpy().astype(np.float32)
            for i in range(arr.shape[0]):
                a = arr[i]  # CHW
                if input_layout.lower() == "nhwc":
                    a = np.transpose(a, (1, 2, 0))  # HWC
                yield a
                n += 1
                if n >= samples:
                    return

    log(f"[rep] using dataloader split={split}, samples={samples}, input_layout={input_layout}")
    return gen()


# ===================== ONNX -> TF -> TFLite =====================
def onnx_to_tf_and_tflite(
    onnx_path: str,
    outdir: str,
    tf_out_name: str,
    quant: str,
    tflite_io: str,
    rep_ds: Optional[Iterable[np.ndarray]],
) -> None:
    """
    quant:
      - "none": 仅导出 float32 tflite
      - "fp16": 导出 fp16 tflite（CPU 可能不更快，但模型更小）
      - "int8": 导出 INT8（权重+激活）；需要 rep_ds
    tflite_io（仅 quant=int8 时生效）:
      - "float32" / "int8" / "uint8"
    """
    tf_dir = os.path.join(outdir, tf_out_name)
    ensure_dir(outdir)

    # 先转 TF SavedModel
    try:
        import onnx  # noqa: F401
        import onnx2tf
    except Exception as e:
        log(f"[tf] onnx2tf missing: {e!r}")
        return

    log("[tf] converting ONNX -> SavedModel ...")
    try:
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=tf_dir,
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True,
        )
        log(f"[tf] SavedModel: {tf_dir}")
    except Exception as e:
        log(f"[tf] onnx2tf.convert failed: {e!r}")
        return

    # 再转 TFLite
    try:
        import tensorflow as tf
    except Exception as e:
        log(f"[tflite] tensorflow missing: {e!r}")
        return

    # 1) float32 baseline
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
        tflite_f32 = converter.convert()
        f32_path = os.path.join(outdir, "model_float32.tflite")
        with open(f32_path, "wb") as f:
            f.write(tflite_f32)
        log(f"[tflite] float32 saved: {f32_path}")
    except Exception as e:
        log(f"[tflite] float32 failed: {e!r}")

    quant = (quant or "none").lower()

    # 2) fp16
    if quant == "fp16":
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_fp16 = converter.convert()
            fp16_path = os.path.join(outdir, "model_fp16.tflite")
            with open(fp16_path, "wb") as f:
                f.write(tflite_fp16)
            log(f"[tflite] fp16 saved: {fp16_path}")
        except Exception as e:
            log(f"[tflite] fp16 failed: {e!r}")

    # 3) int8
    if quant == "int8":
        if rep_ds is None:
            log("[tflite] INT8 requires representative dataset. Provide --rep-dir or ensure dataset.py works.")
            return

        def rep_data_gen():
            for arr in rep_ds:
                a = np.asarray(arr, dtype=np.float32)
                yield [np.expand_dims(a, 0)]

        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = rep_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

            io = (tflite_io or "float32").lower()
            if io == "int8":
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            elif io == "uint8":
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            # io == float32: 不设置，保持 float32 I/O（部署最省事）

            tflite_int8 = converter.convert()
            name = f"model_int8_{io}.tflite"
            i8_path = os.path.join(outdir, name)
            with open(i8_path, "wb") as f:
                f.write(tflite_int8)
            log(f"[tflite] INT8 saved: {i8_path}")
        except Exception as e:
            log(f"[tflite] INT8 failed: {e!r}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, default=CONFIG["ckpt"])
    ap.add_argument("--variant", type=str, default=CONFIG["variant"])
    ap.add_argument("--img-size", type=int, default=CONFIG["img_size"])
    ap.add_argument("--num-classes", type=int, default=CONFIG["num_classes"])
    ap.add_argument("--outdir", type=str, default=CONFIG["outdir"])
    ap.add_argument("--opset", type=int, default=CONFIG["opset"])

    ap.add_argument("--input-layout", type=str, default=CONFIG["input_layout"], choices=["nchw", "nhwc"])

    ap.add_argument("--only-onnx", action="store_true", default=CONFIG["only_onnx"])
    ap.add_argument("--onnx-simplify", action="store_true", default=CONFIG["onnx_simplify"])
    ap.add_argument("--check-with-ort", action="store_true", default=CONFIG["check_with_ort"])

    ap.add_argument("--quant", type=str, default=CONFIG["quant"], choices=["none", "fp16", "int8"])
    ap.add_argument("--tflite-io", type=str, default=CONFIG["tflite_io"], choices=["float32", "int8", "uint8"])
    ap.add_argument("--tf-out-name", type=str, default=CONFIG["tf_out_name"])

    ap.add_argument("--rep-dir", type=str, default=CONFIG["rep_dir"])
    ap.add_argument("--rep-limit", type=int, default=CONFIG["rep_limit"])
    ap.add_argument("--rep-apply-normalize", action="store_true", default=CONFIG["rep_apply_normalize"])

    ap.add_argument("--calib-split", type=str, default=CONFIG["calib_split"], choices=["train", "val", "test"])
    ap.add_argument("--calib-samples", type=int, default=CONFIG["calib_samples"])

    ap.add_argument("--auto-install-tf-deps", action="store_true", default=CONFIG["auto_install_tf_deps"])

    args = ap.parse_args()

    ensure_dir(args.outdir)

    # 1) 构建模型并加载权重
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = get_model(args.variant, num_classes=args.num_classes, pretrained=False, device=device, verbose=True)
    load_ckpt_into(base_model, args.ckpt, device=device)
    base_model.to(device).eval()

    # 2) 包装输入布局
    if args.input_layout.lower() == "nhwc":
        model = NHWCWrapper(base_model).to(device).eval()
    else:
        model = base_model

    # 3) 导出 ONNX
    onnx_path = os.path.join(args.outdir, "model.onnx")
    export_onnx(model, onnx_path, args.img_size, args.opset, args.input_layout)

    # 4) ONNX simplify（可选）
    if args.onnx_simplify:
        onnx_sim_path = os.path.join(args.outdir, "model_simplified.onnx")
        onnx_path = simplify_onnx(onnx_path, onnx_sim_path)

    # 5) ORT 快速检查（可选）
    if args.check_with_ort:
        check_with_onnxruntime(onnx_path, args.img_size, args.input_layout)

    if args.only_onnx:
        log("[done] only-onnx enabled, stop here.")
        return

    # 6) 准备代表性数据（仅 INT8 需要）
    rep_ds: Optional[Iterable[np.ndarray]] = None
    if args.quant.lower() == "int8":
        mean = CONFIG["mean"]
        std = CONFIG["std"]
        rep_ds = make_rep_dataset_from_dir(
            rep_dir=args.rep_dir,
            img_size=args.img_size,
            input_layout=args.input_layout,
            limit=args.rep_limit,
            apply_normalize=bool(args.rep_apply_normalize),
            mean=mean,
            std=std,
        )
        if rep_ds is None:
            rep_ds = make_rep_dataset_from_loader(
                split=args.calib_split,
                samples=args.calib_samples,
                input_layout=args.input_layout,
            )

    # 7) ONNX -> TF SavedModel -> TFLite
    if args.auto_install_tf_deps:
        auto_install_tf_deps(["tensorflow", "onnx", "onnx2tf"])

    onnx_to_tf_and_tflite(
        onnx_path=onnx_path,
        outdir=args.outdir,
        tf_out_name=args.tf_out_name,
        quant=args.quant,
        tflite_io=args.tflite_io,
        rep_ds=rep_ds,
    )

    log("[done] export finished.")


if __name__ == "__main__":
    main()