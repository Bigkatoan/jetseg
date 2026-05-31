#!/usr/bin/env python3
"""Generate a markdown list of models from the package registry.

Writes output to `jetseg/docs/models.md` (created if missing).
"""
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
registry_path = root / "jetseg" / "model_registry.json"
out_dir = root / "jetseg" / "docs"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "models.md"

if not registry_path.exists():
    print("Registry not found:", registry_path)
    raise SystemExit(1)

reg = json.loads(registry_path.read_text(encoding="utf-8"))

package_dir = root / "jetseg"

lines = ["# Available Models\n", "This file is generated from `jetseg/model_registry.json`.\n"]

for task, tentry in reg.items():
    lines.append(f"## Task: {task}\n")
    default = tentry.get("default")
    lines.append(f"**Default:** {default}\n")
    models = tentry.get("models", {})
    for name, meta in models.items():
        lines.append(f"### {name}  \n- Description: {meta.get('description','')}\n- Input size: {meta.get('input_size', 'unknown')}\n- Version: {meta.get('version','')}")
        variants = meta.get("variants", {})
        if variants:
            lines.append("- Variants:")
            for vname, path in variants.items():
                # try to check existence (resolve relative to package directory)
                p = (package_dir / path).resolve() if not Path(path).is_absolute() else Path(path)
                exists = p.exists()
                lines.append(f"  - {vname}: {path} (exists: {exists})")
        lines.append("\n")

out_file.write_text('\n'.join(lines), encoding='utf-8')
print("Wrote", out_file)
