"""Model discovery, metadata and inference helpers for JetSeg.

Public API:
- scan_model_store() -> dict
- get_model_info(path) -> dict
- generate_markdown_table(models_info, device_profile=None) -> str
- run_inference_on_models(models, images_dir, out_root, no_quantize=True) -> dict
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("jetseg.model_utils")


def scan_model_store(root: Optional[Path] = None) -> dict:
    if root is None:
        root = Path(__file__).resolve().parent / "model_store"
    out = {}
    if not root.exists():
        return out
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        out[task] = {}
        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            variants = []
            for f in sorted(model_dir.iterdir()):
                if f.suffix.lower() in (".onnx", ".pt", ".pth"):
                    variants.append(str(f.name))
            out[task][model_dir.name] = variants
    return out


def _try_import_onnx():
    try:
        import onnx
        return onnx
    except Exception:
        return None


def _try_import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def get_model_info(model_path: str) -> dict:
    """Return metadata for a single model file.

    Fields: path, variant (fp32/fp16/int8/pth), input_shape, output_shape,
    params (int), est_memory_mb (float).
    """
    p = Path(model_path)
    info = {
        "path": str(p),
        "name": p.name,
        "variant": None,
        "input_shape": None,
        "output_shape": None,
        "params": None,
        "est_memory_mb": None,
    }

    stem = p.stem.lower()
    if "fp16" in stem:
        info["variant"] = "fp16"
    elif "int8" in stem or "quant" in stem:
        info["variant"] = "int8"
    elif p.suffix.lower() in (".pt", ".pth"):
        info["variant"] = "pth"
    else:
        info["variant"] = "fp32"

    # ONNX: try to read shapes and parameter count
    if p.suffix.lower() == ".onnx":
        onnx = _try_import_onnx()
        if onnx is not None:
            try:
                model = onnx.load(str(p))
                # inputs
                if model.graph.input:
                    inp = model.graph.input[0]
                    shape = []
                    try:
                        for d in inp.type.tensor_type.shape.dim:
                            if d.dim_value:
                                shape.append(int(d.dim_value))
                            else:
                                shape.append(None)
                        info["input_shape"] = tuple(shape)
                    except Exception:
                        info["input_shape"] = None
                # outputs
                if model.graph.output:
                    outp = model.graph.output[0]
                    shape = []
                    try:
                        for d in outp.type.tensor_type.shape.dim:
                            if d.dim_value:
                                shape.append(int(d.dim_value))
                            else:
                                shape.append(None)
                        info["output_shape"] = tuple(shape)
                    except Exception:
                        info["output_shape"] = None

                # params: sum initializers sizes
                params = 0
                for init in model.graph.initializer:
                    # dims product
                    prod = 1
                    for dim in init.dims:
                        prod *= int(dim)
                    params += prod
                info["params"] = int(params)
                if params:
                    # assume float32 unless variant fp16/int8
                    dtype_bytes = 2 if info["variant"] == "fp16" else (1 if info["variant"] == "int8" else 4)
                    info["est_memory_mb"] = float(params * dtype_bytes) / (1024 * 1024)
            except Exception as e:
                logger.debug("Failed to parse ONNX model %s: %s", p, e)

    elif p.suffix.lower() in (".pt", ".pth"):
        torch = _try_import_torch()
        if torch is not None:
            try:
                sd = torch.load(str(p), map_location="cpu")
                # sd may be state_dict or a dict with 'state_dict'
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                total = 0
                if isinstance(sd, dict):
                    for v in sd.values():
                        try:
                            total += getattr(v, "numel", lambda: 0)()
                        except Exception:
                            pass
                info["params"] = int(total)
                if total:
                    info["est_memory_mb"] = float(total * 4) / (1024 * 1024)
            except Exception as e:
                logger.debug("Failed to inspect torch model %s: %s", p, e)

    # best-effort defaults
    if info["params"] is None:
        info["params"] = 0
    if info["est_memory_mb"] is None:
        info["est_memory_mb"] = 0.0

    return info


def estimate_recommendation(info: dict, device_profile: Optional[str] = None) -> str:
    """Return a textual recommendation based on params and device_profile.

    device_profile: 'jetson', 'pi', 'desktop' or None
    """
    params = info.get("params", 0)
    mem_mb = info.get("est_memory_mb", 0.0)
    variant = info.get("variant", "fp32")

    # heuristics
    if device_profile is None:
        device_profile = "auto"

    if device_profile == "jetson":
        if variant == "fp16" or mem_mb < 100:
            return "Recommended (Jetson): good fit (FP16 preferred)"
        return "May be heavy on Jetson; consider FP16 or tiny model"
    if device_profile == "pi":
        if variant == "int8" or mem_mb < 50:
            return "Recommended (Pi): use INT8/tiny model"
        return "Too large for Pi; consider quantization or tiny variant"
    if device_profile == "desktop":
        return "Recommended (Desktop): OK"

    # auto heuristic
    if mem_mb < 50 or "tiny" in info.get("path", ""):
        return "Suitable for low-resource devices (Pi/ARM)"
    if mem_mb < 200:
        return "Suitable for edge GPU (Jetson)"
    return "Prefer desktop/server GPU"


def generate_markdown_table(models_info: dict, device_profile: Optional[str] = None) -> str:
    """Return a Markdown table representing models_info.

    models_info: dict mapping task -> model_name -> list of variant paths
    """
    def _fmt_shape(shape):
        if not shape:
            return "-"
        try:
            s = tuple(shape)
            if len(s) == 4:
                # prefer human-friendly HxWxC with layout hint
                if s[1] == 3:
                    return f"{s[2]}x{s[3]}x3 (NCHW)"
                if s[3] == 3:
                    return f"{s[1]}x{s[2]}x3 (NHWC)"
            # fallback: join dims
            return "x".join(str(x) if x is not None else "?" for x in s)
        except Exception:
            return str(shape)

    def _fmt_params(p):
        try:
            p = int(p)
        except Exception:
            return str(p)
        if p >= 1_000_000:
            return f"{p/1_000_000:.2f}M"
        if p >= 1000:
            return f"{p/1000:.1f}k"
        return str(p)

    def _short_rec(rec: str) -> str:
        if not rec:
            return ""
        r = rec.lower()
        if "pi" in r or "low-resource" in r or "int8" in r:
            return "Low"
        if "jetson" in r or "fp16" in r:
            return "Edge"
        if "desktop" in r or "server" in r:
            return "Desktop"
        return rec

    rows = []
    header = ["Task", "Model", "Variant", "Input", "Output", "Params", "Mem(MB)", "Rec"]
    rows.append("| " + " | ".join(header) + " |")
    rows.append("|" + " --- |" * len(header))
    for task, models in models_info.items():
        for mname, variants in models.items():
            for v in variants:
                p = Path(__file__).resolve().parent / "model_store" / task / mname / v
                info = get_model_info(str(p))
                rec = estimate_recommendation(info, device_profile=device_profile)
                inp = _fmt_shape(info.get("input_shape"))
                out = _fmt_shape(info.get("output_shape"))
                params = _fmt_params(info.get("params"))
                mem = f"{info.get('est_memory_mb'):.1f}"
                rows.append(f"| {task} | {mname} | {info.get('variant')} | {inp} | {out} | {params} | {mem} | {_short_rec(rec)} |")
    return "\n".join(rows)


def run_inference_on_models(models_info: dict, images_dir: str, out_root: str, no_quantize: bool = True, use_fp16: bool = False, composite_mode: str = "side_by_side") -> dict:
    """Run inference for each model/variant on images and save outputs.

    Returns mapping model -> list of generated image paths.
    """
    from jetseg import inference as inference_mod
    from jetseg.engine import HumanSeg

    images_dir = Path(images_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    results = {}
    for task, models in models_info.items():
        for mname, variants in models.items():
            for v in variants:
                model_path = Path(__file__).resolve().parent / "model_store" / task / mname / v
                if not model_path.exists():
                    continue
                key = f"{task}/{mname}/{v}"
                results[key] = []
                try:
                    # choose backend based on file suffix
                    suffix = model_path.suffix.lower()
                    if suffix in (".pt", ".pth"):
                        # prefer torch backend if available
                        hs = HumanSeg(use_fp16=use_fp16, cache_dir=None, backend="torch", torch_model_path=str(model_path), input_size=None, torch_device=None, auto_quantize=False, no_quantize=True)
                    else:
                        hs = HumanSeg(use_fp16=use_fp16, cache_dir=None, backend="onnx", model_path=str(model_path), auto_quantize=not no_quantize, no_quantize=no_quantize)
                except Exception as e:
                    logger.warning("Failed to init HumanSeg for %s: %s", model_path, e)
                    continue

                out_dir = out_root / task / mname / Path(v).stem
                out_dir.mkdir(parents=True, exist_ok=True)
                for img in images:
                    img_np = __import__("cv2").imread(str(img))
                    if img_np is None:
                        continue
                    pred = hs.predict(img_np)
                    if pred is None:
                        continue
                    gt = __import__("numpy").zeros_like(pred)
                    out_path = out_dir / f"{img.stem}_comp.png"
                    inference_mod.save_composite(img_np, gt, pred, str(out_path), mode=composite_mode)
                    results[key].append(str(out_path))
    return results
