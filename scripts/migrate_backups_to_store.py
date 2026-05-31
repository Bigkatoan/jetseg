#!/usr/bin/env python3
"""Migrate old `jetseg/jetseg/backups` artifacts into the new `model_store` layout.

This script is non-destructive by default: it will COPY files into the new store.
Use `--move` to perform a destructive move instead.

It will also optionally update `jetseg/jetseg/model_registry.json` when `--register` is passed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backups", default="jetseg/jetseg/backups", help="existing backups folder")
    p.add_argument("--store", default="jetseg/jetseg/model_store", help="destination model_store root")
    p.add_argument("--task", default="humanseg", help="task name to register under")
    p.add_argument("--move", action="store_true", help="move files instead of copying (destructive)")
    p.add_argument("--register", action="store_true", help="update model_registry.json with migrated models")
    args = p.parse_args()

    backups = Path(args.backups)
    store = Path(args.store)
    pkg_root = Path(__file__).parent.parent / "jetseg"
    registry_path = pkg_root / "model_registry.json"

    if not backups.exists():
        print("Backups folder not found:", backups)
        return

    # collect candidate model base names (prefix before first underscore or full stem for .pt/.pth)
    files = [p for p in backups.iterdir() if p.is_file()]
    bases = set()
    for f in files:
        name = f.name
        if "_fp32" in name or "_fp16" in name or "_int8" in name:
            base = name.split("_")[0]
        else:
            base = Path(name).stem
        bases.add(base)

    migrated = []
    for base in sorted(bases):
        model_files = [f for f in files if f.name.startswith(base)]
        if not model_files:
            continue
        dest = store / args.task / base
        dest.mkdir(parents=True, exist_ok=True)
        for f in model_files:
            dest_path = dest / f.name
            if args.move:
                shutil.move(str(f), str(dest_path))
                print(f"Moved {f} -> {dest_path}")
            else:
                shutil.copy2(str(f), str(dest_path))
                print(f"Copied {f} -> {dest_path}")
        # write manifest + metadata
        manifest = {"models": [str(p.name) for p in sorted(dest.iterdir()) if p.suffix in (".onnx", ".pt", ".pth")]}
        with open(dest / "manifest.json", "w") as mf:
            json.dump(manifest, mf, indent=2)
        metadata = {"task": args.task, "model_name": base, "artifacts": manifest["models"]}
        with open(dest / "metadata.json", "w") as md:
            json.dump(metadata, md, indent=2)
        migrated.append(base)

        # optionally register
        if args.register and registry_path.exists():
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    reg = json.load(f)
            except Exception:
                reg = {}
            if args.task not in reg:
                reg[args.task] = {"default": base, "models": {}}
            models = reg[args.task].setdefault("models", {})
            # build variants
            variants = {}
            for art in manifest["models"]:
                if art.endswith("_fp32.onnx"):
                    variants["fp32"] = f"model_store/{args.task}/{base}/{art}"
                elif art.endswith("_fp16.onnx"):
                    variants["fp16"] = f"model_store/{args.task}/{base}/{art}"
                elif art.endswith("_int8.onnx"):
                    variants["int8"] = f"model_store/{args.task}/{base}/{art}"
                elif art.endswith(".pt") or art.endswith(".pth"):
                    variants["pth"] = f"model_store/{args.task}/{base}/{art}"
            models[base] = {"description": f"Migrated from backups", "input_size": None, "variants": variants, "version": "v1"}
            if not reg[args.task].get("default"):
                reg[args.task]["default"] = base
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(reg, f, indent=2)
            print(f"Registered {base} in registry")

    print("Migration complete. Migrated models:", migrated)


if __name__ == "__main__":
    main()
