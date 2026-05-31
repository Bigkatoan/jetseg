# JetSeg 🚀

**Optimized Human Segmentation Library for NVIDIA Jetson Orin Nano**

JetSeg is a lightweight, high-performance Python library designed specifically for **real-time human segmentation**. It is powered by **TensorRT (via ONNX Runtime)** to leverage the DLA/GPU capabilities of NVIDIA Jetson devices, achieving significantly lower latency compared to standard CPU inference.

> **Note:** This package is specialized for **Human Segmentation** tasks, optimized for edge devices like Jetson Orin Nano.

## 🖼️ Visualization

See JetSeg in action. The library takes a raw input image and produces a precise binary mask or a background-removed result.

| **Raw Input** | **Prediction Result** |
|:---:|:---:|
| ![Raw Input](image.jpg) | ![Prediction Result](predict.jpg) |

## ✨ Features

* **Batteries Included:** The segmentation model (`human_seg.onnx`) is bundled within the library. No external downloads required.
* **Hardware Acceleration:** Uses **TensorRT (FP16)** provider by default for maximum FPS on Jetson Orin Nano.
* **Auto Caching:** Automatically builds and caches TensorRT engines in `~/.cache/jetseg/` to speed up subsequent startups.
* **Easy API:** minimal boilerplate code. Just `import`, `init`, and `predict`.
* **Utilities:** Built-in background removal and replacement tools.

## 🛠️ Prerequisites

Before installing `jetseg`, ensure your Jetson environment is set up:

1. **Hardware:** NVIDIA Jetson Nano / TX2 / Xavier / Orin Nano / Orin AGX.
2. **JetPack:** 5.x or 6.x recommended.
3. **Dependencies:**
   * Python 3.8+
   * **onnxruntime-gpu**: This must be installed specifically for Jetson (JetPack version). Current wheel on libs folder.

   ```bash
   # Install onnxruntime-gpu from Jetson Zoo (if not already installed)
   pip install onnxruntime-gpu --extra-index-url
   ```

   *Note: Standard `pip install onnxruntime-gpu` usually pulls the x86 version which won't utilize Jetson's GPU correctly.*

## 🔐 Model storage & Git LFS

This repository stores large model artifacts (ONNX / PyTorch files) inside `jetseg/jetseg/model_store/` and uses Git LFS to keep the repository lightweight while tracking them in Git. After cloning, run:

```bash
git lfs install
git lfs pull --include="jetseg/jetseg/model_store/**"
```

If you are a maintainer and you add new model files, track them with:

```bash
git lfs track "jetseg/jetseg/model_store/**"
git add .gitattributes
git add <your-model-files>
git commit -m "Add model files (tracked by LFS)"
```

### Platform-specific ONNX Runtime (Jetson)

We include a Jetson-specific `onnxruntime_gpu` wheel in `libs/` for convenience. Do NOT include platform wheels in Python package releases — they are intentionally excluded from the sdist/wheel. To install the Jetson wheel locally:

```bash
# example (adjust filename to match file in libs/)
pip install libs/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
```

See the "On-device quantization" section below for how to prepare device-optimized variants.

## 📦 Installation

### Option 1: Install from Wheel (Recommended)

```bash
cd dist
pip install jetseg-1.0.0-py3-none-any.whl --force-reinstall
```
### Option 2: Install from git

```bash
pip install git+https://github.com/Bigkatoan/jetseg
```

### Option 3: Install from Source (For Developers and Other Platforms)

Clone the repository and install in editable mode:

```bash
cd jetseg_project
pip install -e .
```

## 🚀 Usage

### 1. Basic Inference (Webcam)

```python
import cv2
from jetseg import HumanSeg

# Initialize model (First run takes ~2 mins to build TensorRT engine)
# use_fp16=True is recommended for Jetson Orin Nano
seg = HumanSeg(use_fp16=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Predict mask
    mask = seg.predict(frame)

    # Remove background (Replace with Green Screen)
    result = seg.remove_background(frame, mask, bg_color=(0, 255, 0))

    cv2.imshow("JetSeg", result)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
```

### 2. Inference on Image

```python
import cv2
from jetseg import HumanSeg

seg = HumanSeg()
image = cv2.imread("test.jpg")

# Get binary mask (0 or 255)
mask = seg.predict(image, threshold=0.5)

# Save mask
cv2.imwrite("mask_output.jpg", mask)
print("Done!")
```

## 🎯 Project targets and supported model types

JetSeg aims to support multiple deployment targets and model variants:

- Edge GPU (NVIDIA Jetson family — Orin, AGX, Nano): targeted FP16/TensorRT engines
- Edge CPU (Raspberry Pi 5, ARM64): targeted INT8 dynamic/static quantized variants
- Desktop/server (x86_64): FP32 or FP16 as appropriate

Models are typically provided in three flavors: `*_fp32.onnx`, `*_fp16.onnx`, and `*_int8.onnx`. The runtime will pick the best available variant based on the device and available ONNX Runtime providers. Use the quantize helpers to generate device-optimized variants locally when needed.

## Inference scripts (multi-model)

Two high-level scripts are provided to make multi-model inference and experimentation convenient:

- `scripts/list_models.py` — enumerate models and available variants under `jetseg/jetseg/model_store/`.
- `scripts/run_inference_multi_model.py` — run inference across multiple models/variants and save per-model outputs under `images/results_multi/`.

Examples

List all models (humanseg task):

```bash
python scripts/list_models.py
```

Run all `humanseg` models on images in `images/` and store outputs:

```bash
python scripts/run_inference_multi_model.py --task humanseg --input images/ --output images/results_multi/
```

Run a specific model (e.g., `human_seg_tiny`) and skip on-device quantization:

```bash
python scripts/run_inference_multi_model.py --task humanseg --models human_seg_tiny --no-quantize
```

Notes:

- The demo scripts assume you run them from the repository root (or that `PYTHONPATH=.` is set).
- `run_inference_multi_model.py` will instantiate `HumanSeg` per model; the engine will attempt to pick device-optimized variants automatically and — unless `--no-quantize` is specified — will prompt to quantize when a cached variant is missing (interactive TTY only).


## ⚙️ Configuration & Performance

### First Run Delay

When you initialize `HumanSeg()` for the very first time (or after clearing cache), TensorRT needs to compile the ONNX model into an engine optimized for your specific GPU.

* **Time:** 1-3 minutes.
* **Action:** Do not interrupt the process.
* **Cache Location:** `~/.cache/jetseg/`

Subsequent runs will load instantly (< 1s).

### FP16 vs FP32

You can toggle precision during initialization:

```python
# Faster, slightly less precise (Recommended for Orin)
seg = HumanSeg(use_fp16=True) 

# Slower, maximum precision
seg = HumanSeg(use_fp16=False) 
```

## ⚠️ Troubleshooting

## ✍️ Adding / Exporting Optimized Models (PyTorch -> ONNX)

You can train or modify a PyTorch model and export it to an ONNX file that JetSeg will load. A lightweight optimized UNet implementation is included at `jetseg/models/optimized_unet.py`.

To export a PyTorch model (or a fresh initialized model) to the bundled ONNX path run:

```bash
python3 export_onnx.py --ckpt path/to/checkpoint.pth --image-size 224
```

This will write the ONNX model to `jetseg/human_seg.onnx` (used by the runtime). Use `--out` to change the output path.

The exported ONNX uses opset 11 and is compatible with ONNX Runtime + TensorRT on Jetson devices.


**1. "TensorRT Provider not found" or "Shared object file not found"**

Make sure CUDA libraries are in your path. Add this to your `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```

Then run `source ~/.bashrc`.

**2. Permission Denied for Cache**
If the library cannot create `~/.cache/jetseg`, try running with appropriate user permissions or check ownership of your home directory.

## 📝 License

[MIT](https://choosealicense.com/licenses/mit/)

## Model Store / How to add or update models

JetSeg now includes a simple model store and registry to manage multiple converted artifacts per task.

Layout (inside the package):

- `jetseg/jetseg/model_store/<task>/<model_name>/` — contains ONNX/PTH artifacts, `manifest.json` and `metadata.json`.
- `jetseg/jetseg/model_registry.json` — registry mapping tasks -> models -> variant paths (relative to `jetseg/jetseg`).

Quick workflow to add a new model (recommended):

1. Train and produce a checkpoint `my_model.pt`.
2. Convert and place artifacts into the model store using the helper:

```bash
python3 convert_backups.py --src /path/to/my_model.pt --out-dir jetseg/jetseg/model_store --task humanseg --model-name my_model --image-size 480 --register
```

3. The script will produce `model_store/humanseg/my_model/` with `*_fp32.onnx`, optional `*_fp16.onnx` and `*_int8.onnx`, plus `manifest.json` and `metadata.json`.

4. If `--register` was used, `jetseg/jetseg/model_registry.json` is updated with relative paths pointing into the package. Otherwise, edit the registry manually.

5. Use the registry loader in code:

```python
from jetseg import HumanSeg
# load default variant from registry
seg = HumanSeg.from_registry('humanseg', 'my_model', backend='onnx')
```

Notes:

- The registry stores relative paths from the package folder, e.g. `model_store/humanseg/my_model/my_model_fp32.onnx`.
- `HumanSeg.from_registry()` will pick a sensible default variant (prefer `fp16` -> `fp32` -> `int8` -> `pth`) but you can force a variant.
- The migration script `scripts/migrate_backups_to_store.py` can migrate existing `jetseg/jetseg/backups/` content into the new store layout.

## Available Models

A generated list of models is available at [jetseg/docs/models.md](jetseg/docs/models.md). Run `tools/generate_model_list.py` to refresh.

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

