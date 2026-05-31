#!/usr/bin/env python3
"""Generate README examples: run inference across models and output a markdown table.

Saves produced composite images under `images/readme_examples/` and
prints a markdown table suitable for embedding in README.md.
"""
from __future__ import annotations

from pathlib import Path
import argparse
from jetseg import model_utils


def main(argv=None):
    parser = argparse.ArgumentParser(prog="generate_readme_examples")
    parser.add_argument("--device", choices=["auto", "jetson", "pi", "desktop"], default="auto")
    parser.add_argument("--images", default="images/", help="Source images folder")
    parser.add_argument("--out", default="images/readme_examples/", help="Output folder for examples")
    parser.add_argument("--no-quantize", action="store_true", help="Do not perform on-device quantization during generation")
    args = parser.parse_args(argv)

    models = model_utils.scan_model_store()
    out = Path(args.out)

    print("Found models:")
    print(models)

    print("Running inference for README examples (this may take some time)...")
    # generate overlay composites (single image per example) for README
    results = model_utils.run_inference_on_models(models, args.images, str(out), no_quantize=args.no_quantize, composite_mode="overlay")

    print("Generation complete. Markdown table:")
    md = model_utils.generate_markdown_table(models, device_profile=args.device if args.device != "auto" else None)
    print(md)
    print('\nExample outputs saved under:', out)


if __name__ == "__main__":
    main()
