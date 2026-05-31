# JetSeg — Human Segmentation Library

JetSeg is a compact Python library for on-device human segmentation. It exposes a small, well-documented API to run inference with ONNX and (optionally) PyTorch models, select device-optimized variants, and integrate segmentation into applications.

---

## Quickstart

Install (development mode):

```bash
pip install -r requirements.txt
pip install -e .
```

Notes:
- On Jetson/embedded platforms you may need a platform-specific ONNX Runtime wheel (not distributed in this repo).
- If you only want to use the library API (no example generation), installing the package is sufficient.

### Minimal example

```python
import cv2
from jetseg import HumanSeg

# create engine (disable on-device quantize if you don't want prompts)
# You can reference a bundled model by name (convenience shorthand):
# HumanSeg("human_seg_tiny") will resolve to the artifact under model_store if available.
seg = HumanSeg("human_seg_tiny")  # or: HumanSeg(model_name='human_seg_tiny')

img = cv2.imread('input.jpg')
mask = seg.predict(img)            # uint8 mask (0 or 255)
cv2.imwrite('mask.png', mask)

# optional: create a green-background overlay
overlay = seg.remove_background(img, mask, bg_color=(0, 255, 0))
cv2.imwrite('overlay.png', overlay)
```

---

## API Reference (summary)

- `HumanSeg(**kwargs)` — main engine. Important kwargs:
    - `use_fp16` (bool, default True): prefer FP16 variants when supported by provider.
    - `cache_dir` (str|None): engine/quantize cache directory.
    - `backend` ("onnx"|"torch", default "onnx"): runtime backend.
    - `torch_model_path` (str|None): path to `.pt`/`.pth` when using `backend='torch'`.
    - `input_size` (tuple|None): override model input resize (W,H).
    - `model_path` (str|None): explicit path to an ONNX or PyTorch artifact.
    - `auto_quantize` (bool, default True): attempt on-device quantization when appropriate.
    - `no_quantize` (bool, default False): skip quantization entirely.

- `HumanSeg.from_registry(task, model_name, variant, ...)` — construct engine using the package registry.

- `predict(image, threshold=0.5)` — run inference; returns an 8-bit mask (H,W) with values 0 or 255.

- `remove_background(image, mask, bg_color=(0,255,0))` — simple compositor that replaces background with `bg_color`.

Examples above show the common usage. See `jetseg/jetseg/engine.py` for full parameter list and behavior.

---

## Model selection and variants

- Use `model_path` to point to an explicit artifact (e.g., `.../human_seg.onnx` or `.../unet_last_fp32.onnx`).
- `HumanSeg.from_registry(...)` looks up registered models and preferred variants.
- At runtime the engine will try to select device-optimized variants (FP16 on TensorRT if available, INT8 when supported). To force a particular file, pass `model_path`.

Model artifacts should live under `jetseg/jetseg/model_store/<task>/<model_name>/` when bundled with the package.

---

## Quantization & device notes

- The library can attempt on-device quantization (dynamic/int8/fp16) to create optimized artifacts. This is controlled with `auto_quantize` and `no_quantize`.
- INT8 ONNX artifacts may fail on CPU with errors like `ConvInteger` not implemented — prefer FP32/FP16 for CPU, or run INT8 on devices/providers that support quantized ops.
- FP16 models expect float16 inputs; the engine will cast inputs automatically when necessary.

If you want to avoid any quantization step, construct `HumanSeg(..., auto_quantize=False, no_quantize=True)`.

---

## Troubleshooting (common issues)

- "Could not find implementation for ConvInteger": INT8 model not supported by the CPU provider. Use FP32/FP16 or an appropriate execution provider.
- PyTorch `.pt` load errors (state_dict mismatch): try using ONNX artifacts or provide a scripted module compatible with the bundled loader.
- Incorrect/empty masks: the engine tries several preprocessing candidates (BGR / RGB / ImageNet norm) and handles both NCHW/NHWC layouts, but custom models should match expected input layout and dtype.

If reporting an issue, include: Python version, ONNX Runtime available providers (run `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`), and the model path.

---

## Development

- Run unit tests:

```bash
pytest -q
```

- Install in editable mode for local development:

```bash
pip install -e .
```

---

## Contributing & License

- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
- Licensed under the terms in [LICENSE](LICENSE).


