#!/usr/bin/env python3
"""Convert PyTorch backup checkpoints into ONNX artifacts optimized for different runtimes.

This script will:
 - Copy provided checkpoint(s) into `jetseg/jetseg/backups/`
 - Load model state into the UNet architecture from the UNET project (fallback to optimized_unet)
 - Export a float32 ONNX with output in [0,1] (applies Sigmoid if needed)
 - Attempt dynamic quantization to INT8 using onnxruntime.quantization (if available)
 - Optionally attempt FP16 conversion if helper libs are present

Usage:
  python3 convert_backups.py --src /home/bigkatoan/UNET/backup --out-dir jetseg/jetseg/backups --image-size 480

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import shutil
import importlib.util
import torch
import torch.nn as nn
import onnx
import os


def load_unet_from_path(unet_py_path: Path):
    spec = importlib.util.spec_from_file_location("unet_models", str(unet_py_path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SigmoidWrapper(nn.Module):
    def __init__(self, model: nn.Module, probs_already: bool = False):
        super().__init__()
        self.model = model
        self.probs_already = bool(probs_already)

    def forward(self, x):
        out = self.model(x)
        if self.probs_already:
            return out
        return torch.sigmoid(out)


def export_to_onnx(model: nn.Module, image_size: int, out_path: Path, opset: int = 11):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )


def try_quantize_dynamic(fp32_path: Path, out_path: Path):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8)
        return True
    except Exception as e:
        print("Dynamic quantization not available or failed:", e)
        return False


def try_convert_fp16(fp32_path: Path, out_path: Path):
    # try common converter(s)
    try:
        # onnxconverter_common provides convert_float_to_float16
        from onnxconverter_common import convert_float_to_float16
        model = onnx.load(str(fp32_path))
        model_fp16 = convert_float_to_float16(model)
        onnx.save(model_fp16, str(out_path))
        return True
    except Exception as e:
        print("FP16 conversion not available or failed:", e)
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(Path.home() / "UNET" / "backup"), help="source checkpoint file or directory")
    p.add_argument("--out-dir", default="jetseg/jetseg/backups", help="output directory inside jetseg package")
    p.add_argument("--image-size", type=int, default=480, help="export ONNX with this input size (square)")
    p.add_argument("--opset", type=int, default=11)
    args = p.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # gather checkpoint files
    cks = []
    if src.is_dir():
        for pth in src.iterdir():
            if pth.suffix in (".pt", ".pth"):
                cks.append(pth)
    elif src.is_file():
        cks = [src]
    else:
        print("Source not found:", src)
        return

    if not cks:
        print("No checkpoints found in", src)
        return

    # attempt to load UNET model definition from user's UNET folder
    unet_py = Path.home() / "UNET" / "models.py"
    unet_mod = None
    if unet_py.exists():
        try:
            unet_mod = load_unet_from_path(unet_py)
            print("Loaded UNET module from:", unet_py)
        except Exception as e:
            print("Failed to import UNET models.py:", e)

    for ck in cks:
        print("Processing checkpoint:", ck)
        dest_ck = out_dir / ck.name
        shutil.copy2(ck, dest_ck)
        print("Copied checkpoint to", dest_ck)

        # load checkpoint
        data = torch.load(ck, map_location="cpu")
        if isinstance(data, dict) and "model_state_dict" in data:
            state = data["model_state_dict"]
        elif isinstance(data, dict) and any(k.startswith("layer") or k in ("state_dict",) for k in data.keys()):
            state = data.get("state_dict", data)
        else:
            state = data

        model = None
        # try user's UNet first
        if unet_mod is not None and hasattr(unet_mod, "UNet"):
            try:
                ModelClass = getattr(unet_mod, "UNet")
                model = ModelClass(in_channels=3, out_channels=1, final_sigmoid=True)
                model.load_state_dict(state)
                print("Loaded checkpoint into UNet class")
            except Exception as e:
                print("Failed to load state into UNet class:", e)
                model = None

        # fallback to optimized_unet inside jetseg
        if model is None:
            try:
                from jetseg.models.optimized_unet import UNetOptimized

                model = UNetOptimized(in_channels=3, out_channels=1)
                # try loading; if fails, ignore
                try:
                    model.load_state_dict(state)
                    print("Loaded checkpoint into UNetOptimized (state matched)")
                except Exception:
                    print("Checkpoint keys did not match UNetOptimized; proceeding with new initialized model")
            except Exception as e:
                print("Failed to import UNetOptimized:", e)
                model = None

        if model is None:
            print("Could not instantiate any compatible model for", ck)
            continue

        # check whether model already outputs probabilities by running a dummy
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, args.image_size, args.image_size)
            try:
                out = model(dummy)
                out_np = out.numpy() if isinstance(out, torch.Tensor) else None
                probs_already = False
                if isinstance(out, torch.Tensor):
                    if out.min() >= 0.0 and out.max() <= 1.0:
                        probs_already = True
                else:
                    probs_already = False
            except Exception:
                # try wrapping with sigmoid if forward fails (shape/divisibility issues)
                probs_already = False

        wrapper = SigmoidWrapper(model, probs_already=probs_already)

        # export FP32 ONNX
        base_name = ck.stem
        out_fp32 = out_dir / f"{base_name}_fp32.onnx"
        print("Exporting ONNX (FP32) to", out_fp32)
        try:
            export_to_onnx(wrapper, args.image_size, out_fp32, opset=args.opset)
            print("Exported fp32 ONNX ->", out_fp32)
        except Exception as e:
            print("Failed to export ONNX:", e)
            continue

        # try FP16 conversion
        out_fp16 = out_dir / f"{base_name}_fp16.onnx"
        if try_convert_fp16(out_fp32, out_fp16):
            print("Saved FP16 ONNX ->", out_fp16)
        else:
            print("FP16 conversion not available; skip")

        # try dynamic quantization to INT8 (for CPU)
        out_int8 = out_dir / f"{base_name}_int8.onnx"
        if try_quantize_dynamic(out_fp32, out_int8):
            print("Saved dynamic-quantized INT8 ONNX ->", out_int8)
        else:
            print("Dynamic quantization unavailable or failed; skip")

    # create a simple manifest
    manifest = {"models": []}
    for p in sorted(out_dir.iterdir()):
        if p.suffix == ".onnx" or p.suffix in (".pt", ".pth"):
            manifest["models"].append(str(p.name))

    try:
        import json

        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print("Wrote manifest.json")
    except Exception:
        pass


if __name__ == "__main__":
    main()
