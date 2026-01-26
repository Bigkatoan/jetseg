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
   pip install onnxruntime-gpu --extra-index-url [https://pypi.jetson.ai](https://pypi.jetson.ai)
   ```

   *Note: Standard `pip install onnxruntime-gpu` usually pulls the x86 version which won't utilize Jetson's GPU correctly.*

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
