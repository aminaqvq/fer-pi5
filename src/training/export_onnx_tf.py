import os
import sys
import json
import time
import math
import argparse
import pathlib
from typing import Optional, Iterable, Tuple

import torch
import torch.nn as nn
import numpy as np

# ==== 用户可直接修改的默认配置 ====
CONFIG = {
    "ckpt": r"D:\fer-pi5\checkpoints\best_model_stage2.pth",  # 训练出的权重
    "variant": "large",           # model_mbv3.get_model 的 variant
    "img_size": 224,              # 与训练一致
    "num_classes": 7,
    "outdir": r"D:\fer-pi5\checkpoints\exported",
    "opset": 13,                  # ONNX opset
    "only_onnx": False,            # 只导 ONNX（跳过 TF/TFLite）
    "onnx_simplify": True,        # 若安装 onnxsim 则简化
    "check_with_ort": True,       # 若安装 onnxruntime 则做数值校验

    # ====== TF / TFLite（可选）======
    "quant": "int8",                # None / "fp16" / "int8"
    "tf_out_name": "tf_model",    # SavedModel 输出目录名
    "auto_install_tf_deps": True,# 缺包时尝试自动安装（离线请设 False）

    # INT8 代表性数据（二选一）
    "calib_split": "val",         # 使用项目 dataloader 的哪一集：train/val/test
    "calib_samples": 3500,         # 取多少张做校准
    "rep_dir": r"D:\fer-pi5\data\Rafdb",              # 或者：给一段图片目录（jpg/png），优先级更高

    # 归一化（和训练一致）
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],

    # dataset 用（仅当需要 INT8 校准且没 rep_dir）
    "csv_base": r"D:\fer-pi5\data\csv",
    "img_base": None,
    "batch_size": 64,
    "num_workers": 0,
}

# ====== 依赖工程里的模型/数据构造 ======
# 你的工程结构已在同一路径下
from model_mbv3 import get_model  # noqa: E402
try:
    # 只有需要 INT8 校准而且 rep_dir 为空时用到
    from dataset import get_dataloaders_hybrid  # noqa: E402
except Exception:
    get_dataloaders_hybrid = None

def log(msg: str):
    print(msg, flush=True)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_ckpt_into(model: nn.Module, ckpt_path: str, device: str = "cpu"):
    log(f"[load] ckpt: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    # 支持 weights_only=True 保存的纯 state_dict；或包含 'state_dict'
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    # 兼容 DataParallel: module.*
    new_state = {}
    for k, v in state.items():
        nk = k
        if k.startswith("module."):
            nk = k[len("module."):]
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        log(f"[load] missing keys: {len(missing)} -> {missing[:6]} ...")
    if unexpected:
        log(f"[load] unexpected keys: {len(unexpected)} -> {unexpected[:6]} ...")

def dummy_input(bs: int, img: int, device: str) -> torch.Tensor:
    return torch.randn(bs, 3, img, img, device=device)

def export_onnx(
    model: nn.Module,
    outdir: str,
    img_size: int,
    opset: int,
    onnx_simplify: bool = True,
    check_with_ort: bool = True,
):
    ensure_dir(outdir)
    onnx_path = os.path.join(outdir, "model.onnx")
    sim_path = os.path.join(outdir, "model.sim.onnx")

    model.eval()
    device = next(model.parameters()).device
    x = dummy_input(1, img_size, device)

    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}
    log(f"[onnx] exporting -> {onnx_path}")
    torch.onnx.export(
        model, x, onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    log("[onnx] done.")

    # 简化
    if onnx_simplify:
        try:
            import onnx
            from onnxsim import simplify
            log("[onnxsim] simplifying ...")
            onnx_model = onnx.load(onnx_path)
            model_simplified, check = simplify(onnx_model)
            if check:
                onnx.save(model_simplified, sim_path)
                log(f"[onnxsim] saved: {sim_path}")
            else:
                log("[onnxsim] check failed, keep original onnx.")
        except Exception as e:
            log(f"[onnxsim] skipped: {e!r}")

    # onnxruntime 数值校验
    if check_with_ort:
        try:
            import onnxruntime as ort
            arr = x.detach().cpu().numpy().astype(np.float32)
            sess = ort.InferenceSession(sim_path if os.path.exists(sim_path) else onnx_path,
                                        providers=["CPUExecutionProvider"])
            ort_out = sess.run(None, {"input": arr})[0]
            with torch.no_grad():
                torch_out = model(x).detach().cpu().numpy()
            atol, rtol = 1e-3, 1e-2
            ok = np.allclose(ort_out, torch_out, atol=atol, rtol=rtol)
            log(f"[ort] check close={ok} (rtol={rtol}, atol={atol}) "
                f"max_abs={np.max(np.abs(ort_out - torch_out)):.4e}")
        except Exception as e:
            log(f"[ort] skipped: {e!r}")

def maybe_install_tf_deps(auto: bool):
    if not auto:
        return
    try:
        import importlib
        def need(name):  # noqa
            try:
                importlib.import_module(name)
                return False
            except Exception:
                return True
        pkgs = []
        if need("tensorflow"):
            pkgs += ["tensorflow==2.15.*", "tf-keras==2.15.*"]
        if need("onnx"):
            pkgs += ["onnx"]
        if need("onnx2tf"):
            pkgs += ["onnx2tf"]
        if pkgs:
            log(f"[deps] auto install: {pkgs}")
            os.system(f"{sys.executable} -m pip install " + " ".join(pkgs))
    except Exception as e:
        log(f"[deps] auto install failed: {e!r}")

def onnx_to_tf_and_tflite(
    onnx_path: str,
    outdir: str,
    quant: Optional[str],
    rep_ds: Optional[Iterable[np.ndarray]],
):
    """
    quant: None / "fp16" / "int8"
    rep_ds: generator yielding fp32 CHW or HWC [0,1] arrays for INT8 calibration
    """
    # 先转 TF SavedModel
    tf_dir = os.path.join(outdir, "tf_model")
    try:
        import onnx
        import onnx2tf
    except Exception as e:
        log(f"[tf] onnx2tf missing: {e!r}")
        return

    log("[tf] converting ONNX -> SavedModel ...")
    # onnx2tf 命令式接口
    try:
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=tf_dir,
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True,
        )
        log(f"[tf] SavedModel: {tf_dir}")
    except Exception as e:
        log(f"[tf] convert failed: {e!r}")
        return

    # 转 TFLite
    try:
        import tensorflow as tf
    except Exception as e:
        log(f"[tflite] tensorflow missing: {e!r}")
        return

    # FP32
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
        converter.experimental_new_converter = True
        tflite_fp32 = converter.convert()
        fp32_path = os.path.join(outdir, "model_fp32.tflite")
        open(fp32_path, "wb").write(tflite_fp32)
        log(f"[tflite] FP32 saved: {fp32_path}")
    except Exception as e:
        log(f"[tflite] FP32 failed: {e!r}")

    # FP16
    if quant == "fp16":
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_fp16 = converter.convert()
            fp16_path = os.path.join(outdir, "model_fp16.tflite")
            open(fp16_path, "wb").write(tflite_fp16)
            log(f"[tflite] FP16 saved: {fp16_path}")
        except Exception as e:
            log(f"[tflite] FP16 failed: {e!r}")

    # INT8
    if quant == "int8":
        if rep_ds is None:
            log("[tflite] INT8 needs representative dataset, but None provided.")
            return
        try:
            def rep_data_gen():
                for arr in rep_ds:
                    # arr shape allow CHW/HWC, convert to HWC float32 in [0,1]
                    a = np.asarray(arr, dtype=np.float32)
                    if a.ndim == 3 and a.shape[0] == 3:
                        a = np.transpose(a, (1, 2, 0))
                    yield [np.expand_dims(a, 0)]
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = rep_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            tflite_int8 = converter.convert()
            i8_path = os.path.join(outdir, "model_int8.tflite")
            open(i8_path, "wb").write(tflite_int8)
            log(f"[tflite] INT8 saved: {i8_path}")
        except Exception as e:
            log(f"[tflite] INT8 failed: {e!r}")

def make_rep_dataset_from_loader(
    split: str,
    samples: int,
    img_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> Optional[Iterable[np.ndarray]]:
    """
    用项目 dataloader 取代表性数据（若 dataset 可用）。
    返回生成器（yield CHW [0,1] float32）
    """
    if get_dataloaders_hybrid is None:
        return None
    # 尽量单进程
    try:
        train_loader, val_loader, test_loader, _ = get_dataloaders_hybrid(
            csv_base=str(CONFIG["csv_base"]),
            img_base=(None if CONFIG["img_base"] in (None, "None", "") else str(CONFIG["img_base"])),
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            pin_memory=True,
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
        for xb, yb in loader:
            # xb: Bx3xHxW already normalized to 0..1 (assumed by your dataset)
            # 如果你的 dataset 已做 mean/std 归一化到 imagenet 均值，则需反归一化到 [0,1]
            # 这里假设 DataLoader 输出是 0..1 归一化（若不是，请按你的数据流程调整）
            arr = xb.detach().cpu().numpy()
            for i in range(arr.shape[0]):
                a = arr[i]
                # 若你的 dataset 做了 (x-mean)/std，请反归一化：
                # a = (a * np.array(std)[:,None,None]) + np.array(mean)[:,None,None]
                a = np.clip(a, 0.0, 1.0)
                yield a.astype(np.float32)
                n += 1
                if n >= samples:
                    return
    return gen()

def make_rep_dataset_from_dir(rep_dir: str, img_size: int) -> Optional[Iterable[np.ndarray]]:
    """
    从目录收集 jpg/png 作为代表性数据（转换为 HWC [0,1]）
    """
    try:
        from PIL import Image
    except Exception:
        log("[rep] PIL not installed; pip install pillow")
        return None

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths += list(pathlib.Path(rep_dir).rglob(ext))
    if not paths:
        log(f"[rep] no images found in {rep_dir}")
        return None

    def gen():
        for p in paths:
            img = Image.open(str(p)).convert("RGB").resize((img_size, img_size))
            a = np.asarray(img, dtype=np.float32) / 255.0  # HWC
            yield a
    return gen()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=CONFIG["ckpt"])
    ap.add_argument("--variant", type=str, default=CONFIG["variant"])
    ap.add_argument("--img-size", type=int, default=CONFIG["img_size"])
    ap.add_argument("--num-classes", type=int, default=CONFIG["num_classes"])
    ap.add_argument("--outdir", type=str, default=CONFIG["outdir"])
    ap.add_argument("--opset", type=int, default=CONFIG["opset"])
    ap.add_argument("--only-onnx", action="store_true" if CONFIG["only_onnx"] else "store_false",
                    default=CONFIG["only_onnx"])
    ap.add_argument("--onnx-simplify", action="store_true" if CONFIG["onnx_simplify"] else "store_false",
                    default=CONFIG["onnx_simplify"])
    ap.add_argument("--check-with-ort", action="store_true" if CONFIG["check_with_ort"] else "store_false",
                    default=CONFIG["check_with_ort"])
    ap.add_argument("--quant", type=str, default=str(CONFIG["quant"]) if CONFIG["quant"] else None,
                    choices=[None, "fp16", "int8"])
    ap.add_argument("--tf-out-name", type=str, default=CONFIG["tf_out_name"])
    ap.add_argument("--auto-install-tf-deps", action="store_true" if CONFIG["auto_install_tf_deps"] else "store_false",
                    default=CONFIG["auto_install_tf_deps"])
    ap.add_argument("--calib-split", type=str, default=CONFIG["calib_split"], choices=["train", "val", "test"])
    ap.add_argument("--calib-samples", type=int, default=CONFIG["calib_samples"])
    ap.add_argument("--rep-dir", type=str, default=CONFIG["rep_dir"])
    args = ap.parse_args([]) if "__file__" not in globals() else ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.outdir)

    # 1) 构建模型并加载权重
    model = get_model(args.variant, num_classes=args.num_classes, pretrained=False, device=device, verbose=True)
    load_ckpt_into(model, args.ckpt, device=device)
    model.to(device).eval()

    # 2) 导出 ONNX（含可选 onnxsim + onnxruntime 校验）
    export_onnx(
        model=model,
        outdir=args.outdir,
        img_size=args.img_size,
        opset=args.opset,
        onnx_simplify=args.onnx_simplify,
        check_with_ort=args.check_with_ort,
    )

    if args.only_onnx:
        log("[done] ONNX export finished (only_onnx=True).")
        return

    # 3) （可选）TF/TFLite
    maybe_install_tf_deps(args.auto_install_tf_deps)

    onnx_path = os.path.join(args.outdir, "model.sim.onnx")
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(args.outdir, "model.onnx")
    if not os.path.exists(onnx_path):
        log("[tf] ONNX not found; abort TF/TFLite.")
        return

    rep_ds = None
    if args.quant == "int8":
        if args.rep_dir:
            rep_ds = make_rep_dataset_from_dir(args.rep_dir, args.img_size)
        else:
            rep_ds = make_rep_dataset_from_loader(
                split=args.calib_split,
                samples=args.calib_samples,
                img_size=args.img_size,
                mean=tuple(CONFIG["mean"]),
                std=tuple(CONFIG["std"]),
            )

    onnx_to_tf_and_tflite(
        onnx_path=onnx_path,
        outdir=args.outdir,
        quant=args.quant,
        rep_ds=rep_ds,
    )
    log("[done] Export completed.")

if __name__ == "__main__":
    main()