#!/usr/bin/env python3
"""Simple CLI to create a quantized variant for a model and store it in cache.

This script is non-interactive by default and will use the chosen method
to generate a cached artifact under the cache directory (or the default
`~/.cache/jetseg`).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from jetseg import quantize
except Exception:
    print("Failed to import jetseg.quantize. Are you running inside the repo?")
    raise


def main(argv=None):
    parser = argparse.ArgumentParser(prog="quantize_on_device")
    parser.add_argument("--input", "-i", required=True, help="Input ONNX model path")
    parser.add_argument("--method", choices=["dynamic", "fp16", "skip"], default="dynamic", help="Quantization method")
    parser.add_argument("--out-dir", default=None, help="Cache/output directory (defaults to ~/.cache/jetseg)")
    parser.add_argument("--force", action="store_true", help="Force re-quantize even if cached variant exists")
    args = parser.parse_args(argv)

    model = Path(args.input)
    if not model.exists():
        print(f"Input model not found: {model}")
        sys.exit(2)

    cache_dir = args.out_dir
    try:
        qpath = quantize.ensure_or_prompt_quantized(str(model), prefer_fp16=(args.method == "fp16"), cache_dir=cache_dir, interactive=False, method=args.method)
        print(f"Quantized model available at: {qpath}")
    except Exception as e:
        print("Quantization failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
