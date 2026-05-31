#!/usr/bin/env python3
"""Run inference across multiple models/variants and save composite outputs.

This script locates models under `jetseg/jetseg/model_store/<task>/<model>/` and
runs `HumanSeg` on images in `--input`. Results are saved per-model/variant
under `--output`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np


def find_model_store() -> Path:
    here = Path(__file__).resolve().parents[1]
    return here / "jetseg" / "model_store"


def pick_model_file(model_dir: Path) -> Path | None:
    # Prefer explicit fp32, then human_seg.onnx, then any .onnx
    candidates = list(sorted(model_dir.glob("*_fp32.onnx")))
    if candidates:
        return candidates[0]
    # common default name
    t = model_dir / "human_seg.onnx"
    if t.exists():
        return t
    # fallback to any onnx
    any_onnx = list(sorted(model_dir.glob("*.onnx")))
    if any_onnx:
        return any_onnx[0]
    # fallback to torch
    any_pt = list(sorted(model_dir.glob("*.pt"))) + list(sorted(model_dir.glob("*.pth")))
    if any_pt:
        return any_pt[0]
    return None


def run_for_model(model_path: Path, images: list[Path], out_dir: Path, args):
    from jetseg.engine import HumanSeg
    from jetseg import inference as inference_mod

    model_name = model_path.parent.name
    variant = model_path.stem
    out_model_dir = out_dir / model_name / variant
    out_model_dir.mkdir(parents=True, exist_ok=True)

    # initialize engine (may trigger quantize prompt unless --no-quantize)
    hs = HumanSeg(use_fp16=args.use_fp16, cache_dir=args.cache_dir, backend=args.backend, model_path=str(model_path), auto_quantize=not args.no_quantize, no_quantize=args.no_quantize)

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue
        pred = hs.predict(img)
        if pred is None:
            print(f"Prediction None for {img_path}")
            continue
        gt_mask = np.zeros_like(pred)
        out_path = out_model_dir / f"{img_path.stem}_comp.png"
        inference_mod.save_composite(img, gt_mask, pred, str(out_path))
        print(f"Saved: {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="run_inference_multi_model")
    parser.add_argument("--task", default="humanseg", help="Task name under model_store")
    parser.add_argument("--models", default=None, help="Comma-separated model names to run (defaults to all)")
    parser.add_argument("--input", default="images/", help="Input images folder")
    parser.add_argument("--output", default="images/results_multi/", help="Output root folder")
    parser.add_argument("--backend", default="onnx", choices=["onnx", "torch"], help="Backend to use")
    parser.add_argument("--no-quantize", action="store_true", help="Do not attempt on-device quantization")
    parser.add_argument("--use-fp16", action="store_true", help="Prefer FP16 variants when available")
    parser.add_argument("--cache-dir", default=None, help="Cache dir for quantized variants")
    args = parser.parse_args(argv)

    model_store = find_model_store()
    task_dir = model_store / args.task
    if not task_dir.exists():
        print(f"Task folder not found: {task_dir}")
        sys.exit(2)

    selected = None
    if args.models:
        selected = [x.strip() for x in args.models.split(",") if x.strip()]

    models = [d for d in sorted(task_dir.iterdir()) if d.is_dir()]
    if selected:
        models = [m for m in models if m.name in selected]

    images = []
    input_dir = Path(args.input)
    if input_dir.exists():
        for p in sorted(input_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                images.append(p)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images:
        print("No images found in input; nothing to do")
        sys.exit(0)

    for model_dir in models:
        model_file = pick_model_file(model_dir)
        if model_file is None:
            print(f"No model file found in {model_dir}; skipping")
            continue
        print(f"Running model {model_dir.name} -> {model_file.name}")
        run_for_model(model_file, images, out_dir, args)


if __name__ == "__main__":
    main()
