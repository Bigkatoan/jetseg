#!/usr/bin/env python3
"""List models and variants stored under jetseg/jetseg/model_store/.

This helper is intended to be run from the repository root. It prints a
human-readable list and can emit JSON for scripting.
"""
from __future__ import annotations

import json
from pathlib import Path
import argparse


def find_model_store() -> Path:
    here = Path(__file__).resolve().parents[1]
    ms = here / "jetseg" / "model_store"
    return ms


def scan_models(model_store: Path) -> dict:
    out = {}
    if not model_store.exists():
        return out
    for task_dir in sorted(model_store.iterdir()):
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
                    variants.append(f.name)
            out[task][model_dir.name] = variants
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog="list_models")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)

    ms = find_model_store()
    models = scan_models(ms)
    if args.json:
        print(json.dumps(models, indent=2))
        return

    if not models:
        print("No models found in:", ms)
        return

    for task, entries in models.items():
        print(f"Task: {task}")
        for mname, variants in entries.items():
            print(f"  - {mname}")
            for v in variants:
                print(f"      {v}")


if __name__ == "__main__":
    main()
