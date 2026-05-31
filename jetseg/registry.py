"""Simple model registry helpers for JetSeg.

This module reads `model_registry.json` shipped with the package and
provides convenience functions to list tasks, models, and resolve
artifact paths for ONNX/PyTorch variants.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

_REGISTRY_PATH = Path(__file__).parent / "model_registry.json"


def load_registry() -> Dict[str, Any]:
    with _REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_tasks():
    return list(load_registry().keys())


def list_models(task: str):
    reg = load_registry()
    if task not in reg:
        raise KeyError(f"Unknown task: {task}")
    return list(reg[task].get("models", {}).keys())


def get_model_entry(task: str, model_name: str | None = None) -> Dict[str, Any]:
    reg = load_registry()
    if task not in reg:
        raise KeyError(f"Unknown task: {task}")
    task_entry = reg[task]
    if model_name is None:
        model_name = task_entry.get("default")
    models = task_entry.get("models", {})
    if model_name not in models:
        raise KeyError(f"Unknown model '{model_name}' for task '{task}'")
    return models[model_name]


def get_model_path(task: str, model_name: str, variant: str) -> Path:
    entry = get_model_entry(task, model_name)
    variants = entry.get("variants", {})
    if variant not in variants:
        raise KeyError(f"Variant '{variant}' not found for model '{model_name}'")
    rel = variants[variant]
    # Resolve relative to the package directory
    abs_path = (Path(__file__).parent / rel).resolve()
    return abs_path


def get_default_variant(task: str, model_name: str) -> str:
    """Return a sensible default variant name for a model (fp16->fp32->int8->pth)."""
    entry = get_model_entry(task, model_name)
    v = entry.get("variants", {})
    for prefer in ("fp16", "fp32", "int8", "pth"):
        if prefer in v:
            return prefer
    # fallback to first available
    if v:
        return next(iter(v.keys()))
    raise KeyError(f"No variants available for model '{model_name}'")
