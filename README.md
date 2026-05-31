# JetSeg — Human Segmentation for Edge Devices

JetSeg is a lightweight Python library for real-time human segmentation, with utilities to run and manage multiple model variants (ONNX / PyTorch) and device-optimized builds for Edge GPUs (Jetson) and CPUs (Raspberry Pi / ARM).

Key goals:
- Easy-to-use API for quick inference
- Support for multiple model variants (FP32 / FP16 / INT8)
- Helpers for on-device quantization and model management

---

## Table Of Contents

- [Quickstart](#quickstart)
- [Usage Examples](#usage-examples)
- [Generating README Examples](#generating-readme-examples)
- [Model Store & Git LFS](#model-store--git-lfs)
- [Configuration & Troubleshooting](#configuration--troubleshooting)
- [Contributing & License](#contributing--license)

---

## Quickstart

Recommended: install from the built wheel (or `pip install -e .` for development).

On Jetson devices you must install a Jetson-compatible ONNX Runtime wheel first (example in `libs/`):

```bash
# (optional) install Jetson-specific ORT wheel (adjust filename)
pip install libs/onnxruntime_gpu-*.whl

# install the package (editable for development)
pip install -e .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/Bigkatoan/jetseg
```

Note: ONNX Runtime wheels are platform-specific — do not publish platform wheels inside the PyPI sdist.

---

## Usage Examples

Minimal image example:

```python
import cv2
from jetseg import HumanSeg

seg = HumanSeg(use_fp16=True)
img = cv2.imread('images/readme_examples/humanseg/human_seg_large/human_seg/image_comp.png')
mask = seg.predict(img)
cv2.imwrite('mask.png', mask)
```

Quick webcam demo:

```python
import cv2
from jetseg import HumanSeg

seg = HumanSeg()
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    mask = seg.predict(frame)
    out = seg.remove_background(frame, mask, bg_color=(0,255,0))
    cv2.imshow('JetSeg', out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

## Generating README Examples

The repository includes a generator script that runs example images through available models and saves overlay visualizations under `images/readme_examples/`.

Run locally (fast, non-quantized):

```bash
PYTHONPATH=. python3 scripts/generate_readme_examples.py --no-quantize
```

This prints a compact markdown table and saves overlay images that are suitable for embedding in the README.

Notes:
- INT8 variants may fail on CPU since some providers do not implement quantized ops (e.g., `ConvInteger`).
- Some `.pt` checkpoints may not load into the packaged UNet architecture; the generator will log and skip those.

---

## Model Store & Git LFS

Models are stored under `jetseg/jetseg/model_store/<task>/<model_name>/` and should be tracked with Git LFS.

After cloning, pull LFS-tracked artifacts:

```bash
git lfs install
git lfs pull --include="jetseg/jetseg/model_store/**"
```

When adding models as a maintainer:

```bash
git lfs track "jetseg/jetseg/model_store/**"
git add .gitattributes
git add <model-files>
git commit -m "Add model files (tracked by LFS)"
```

Registry & layout:

- `jetseg/jetseg/model_store/<task>/<model_name>/` — artifacts and manifest
- `jetseg/jetseg/model_registry.json` — registry mapping used by `HumanSeg.from_registry()`

---

## Configuration & Troubleshooting

- First run on Jetson may take 1–3 minutes while TensorRT compiles an engine — engines are cached in `~/.cache/jetseg/`.
- If you see `Could not find implementation for ConvInteger` when loading INT8 models, run FP32/FP16 variants on CPU or use a device that supports quantized ops.
- If a `.pt` file fails to load due to state_dict mismatch, either export a matching `.pt` (scripted or full model) or use the provided ONNX artifacts instead.

If you encounter problems, collect logs and open an issue with: Python version, ONNX Runtime provider list (`python -c "import onnxruntime as ort; print(ort.get_available_providers())"`), and the model path.

---

## Contributing & License

Contributions welcome — see `CONTRIBUTING.md` for guidelines.

Licensed under MIT.


## README Examples

The repository includes a small set of generated README examples showing overlay visualizations and a compact model table. Example images are saved to `images/readme_examples/` and are produced by `scripts/generate_readme_examples.py`.

To (re)generate the examples locally:

```bash
PYTHONPATH=. python3 scripts/generate_readme_examples.py --no-quantize
```

Notes:
- The script will produce overlay composites (single-image visualizations) by default and print a compact markdown table summarizing input/output shapes, parameter counts and a short recommendation.
- INT8 variants may fail on CPU because not all providers implement quantized ops (e.g., ConvInteger). The generator will skip non-runnable variants and log warnings.
- PyTorch checkpoint (`.pt`/`.pth`) variants may not always load into the bundled `UNetOptimized` architecture; those are skipped with a warning.

Generated model table (example):

| Task | Model | Variant | Input | Output | Params | Mem(MB) | Rec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| humanseg | human_seg_large | fp32 | 224x224x3 (NHWC) | 1x224x224x1 | 7.76M | 29.6 | Low |
| humanseg | human_seg_tiny | pth | - | - | 0 | 0.0 | Low |
| humanseg | human_seg_tiny | fp16 | 480x480x3 (NCHW) | 1x1x480x480 | 248.7k | 0.5 | Low |
| humanseg | human_seg_tiny | fp32 | 480x480x3 (NCHW) | 1x1x480x480 | 248.7k | 0.9 | Low |
| humanseg | human_seg_tiny | int8 | 480x480x3 (NCHW) | 1x1x480x480 | 248.8k | 0.2 | Low |

See `images/readme_examples/` for the generated overlays used in the README.

