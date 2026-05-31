"""On-device quantization helpers for JetSeg.

Provides lightweight helpers to compute a model checksum, locate cached
quantized variants in a cache directory, and perform best-effort
dynamic INT8 or FP16 conversions. The module is robust when optional
dependencies are missing: in non-interactive mode it will simply return
the original model path if quantization tooling is not available.
"""
from __future__ import annotations

import hashlib
import json
import time
import shutil
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("jetseg.quantize")


def compute_model_key(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(cache_dir: str) -> Path:
    return Path(cache_dir) / "quantize_manifest.json"


def load_manifest(cache_dir: str) -> dict:
    p = _manifest_path(cache_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_manifest(cache_dir: str, manifest: dict) -> None:
    p = _manifest_path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))


def find_cached_variant(model_path: str, variant: str, cache_dir: str) -> Optional[Path]:
    """Return a Path to a cached quantized variant if available, else None."""
    try:
        sha = compute_model_key(model_path)
    except FileNotFoundError:
        return None

    manifest = load_manifest(cache_dir)
    entry = manifest.get(sha)
    if entry and variant in entry:
        fname = entry[variant].get("filename")
        if fname:
            fp = Path(cache_dir) / fname
            if fp.exists():
                return fp

    # fallback: search for files matching a conventional pattern
    p = Path(cache_dir)
    if not p.exists():
        return None
    stem = Path(model_path).stem
    for candidate in p.glob(f"{stem}_{variant}_*.onnx"):
        return candidate
    return None


def quantize_dynamic(input_path: str, out_path: str) -> Path:
    """Perform dynamic quantization (weights) using ONNX Runtime quantization.

    Raises RuntimeError if the quantization API is not available.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as e:
        raise RuntimeError("onnxruntime.quantization not available; install onnxruntime with quantization support") from e

    quantize_dynamic(input_path, out_path, weight_type=QuantType.QInt8)
    return Path(out_path)


def convert_to_fp16(input_path: str, out_path: str) -> Path:
    """Best-effort FP16 conversion. Falls back to copying the file if
    a converter isn't installed.
    """
    try:
        import onnx
        from onnxconverter_common import float16
    except Exception:
        logger.warning("FP16 converter not available; copying FP32 model as placeholder")
        shutil.copy(input_path, out_path)
        return Path(out_path)

    model = onnx.load(input_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, out_path)
    return Path(out_path)


def ensure_or_prompt_quantized(model_path: str, prefer_fp16: bool = False, cache_dir: Optional[str] = None, interactive: bool = True, method: Optional[str] = None) -> Path:
    """Ensure a quantized variant exists in `cache_dir`.

    If a cached variant exists it is returned. In non-interactive mode,
    if no variant is found the original model path is returned. In
    interactive mode the user is prompted to choose a method and the
    resulting artifact is written into `cache_dir` and recorded in the
    manifest.
    """
    if cache_dir is None:
        from os.path import expanduser

        cache_dir = str(Path(expanduser("~")) / ".cache" / "jetseg")

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    try:
        sha = compute_model_key(model_path)
    except FileNotFoundError:
        raise

    # pick a desired variant
    if method:
        chosen = method
    else:
        chosen = "fp16" if prefer_fp16 else "dynamic"

    # map method -> variant string
    method_map = {"dynamic": "int8", "int8": "int8", "fp16": "fp16", "skip": "skip"}
    variant = method_map.get(chosen, None)

    if variant == "skip":
        return Path(model_path)

    if variant is None:
        # fallback behavior: non-interactive -> return original
        if not interactive:
            return Path(model_path)

    # check cache
    if variant is not None:
        found = find_cached_variant(model_path, variant, cache_dir)
        if found:
            return found

    # non-interactive and no cached variant -> return original
    if not interactive:
        return Path(model_path)

    # Interactive prompt
    print(f"No cached quantized variant found for {model_path}")
    print("Choose quantization method:")
    print("1) Dynamic INT8 (fast, no calibration)")
    print("2) Static INT8 (requires calibration) - not implemented here")
    print("3) Convert to FP16 (suitable for GPU/TensorRT)")
    print("4) Skip - use FP32 now")
    choice = input("Select [1-4]: ").strip()

    if choice == "1":
        variant = "int8"
        out_name = f"{Path(model_path).stem}_{variant}_{sha[:8]}.onnx"
        out_path = Path(cache_dir) / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        quantize_dynamic(str(model_path), str(out_path))
    elif choice == "3":
        variant = "fp16"
        out_name = f"{Path(model_path).stem}_{variant}_{sha[:8]}.onnx"
        out_path = Path(cache_dir) / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        convert_to_fp16(str(model_path), str(out_path))
    else:
        # skip or unsupported choice
        return Path(model_path)

    # update manifest
    manifest = load_manifest(cache_dir)
    manifest.setdefault(sha, {})[variant] = {"filename": out_name, "created_at": int(time.time())}
    save_manifest(cache_dir, manifest)

    return out_path
