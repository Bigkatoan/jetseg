"""Export an optimized PyTorch UNet to ONNX for JetSeg inference.

Usage examples:
  python3 export_onnx.py --out jetseg/human_seg.onnx --image-size 224

If you have a trained PyTorch checkpoint, pass `--ckpt path/to.ckpt`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="path to PyTorch checkpoint (state_dict or full ckpt)")
    p.add_argument("--out", default=None, help="output ONNX path (defaults to jetseg/jetseg/human_seg.onnx)")
    p.add_argument("--image-size", type=int, default=224, help="input image size (square)")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    default_out = repo_root / "jetseg" / "human_seg.onnx"
    out_path = Path(args.out) if args.out else default_out

    # import model local to repo
    from jetseg.models.optimized_unet import UNetOptimized

    model = UNetOptimized(in_channels=3, out_channels=1)
    model.eval()

    if args.ckpt:
        ckpt_p = Path(args.ckpt)
        if ckpt_p.exists():
            data = torch.load(ckpt_p, map_location="cpu")
            # accept either full ckpt dict or state_dict
            if isinstance(data, dict) and "model_state_dict" in data:
                state = data["model_state_dict"]
            elif isinstance(data, dict) and any(k.startswith("layer") or k in ("state_dict",) for k in data.keys()):
                # try common keys
                state = data.get("state_dict", data)
            else:
                state = data
            try:
                model.load_state_dict(state)
                print(f"Loaded weights from {ckpt_p}")
            except Exception as e:
                print(f"Warning: failed to load checkpoint: {e}")
        else:
            print(f"Checkpoint not found: {ckpt_p}")

    dummy = torch.randn(1, 3, args.image_size, args.image_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            do_constant_folding=True,
            verbose=False,
        )
        print(f"Exported ONNX model to {out_path}")
    except ModuleNotFoundError as e:
        # torch.onnx may require onnxscript in newer PyTorch versions
        print("Failed to export ONNX: ", e)
        print("Hint: please install the 'onnxscript' package in your Python environment:")
        print("  pip install onnxscript")
        print("Or activate your project's venv that has torch+onnx support.")


if __name__ == "__main__":
    main()
