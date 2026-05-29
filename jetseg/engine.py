import logging
import onnxruntime as ort
import numpy as np
import cv2
import os
from pathlib import Path

try:
    import torch
except Exception:
    torch = None

# package logger
logger = logging.getLogger("jetseg")


class HumanSeg:
    def __init__(self, use_fp16=True, cache_dir=None, backend: str = "onnx", torch_model_path: str | None = None, input_size: tuple | None = None, torch_device: str | None = None):
        """
        Initialize JetSeg HumanSeg engine.
        :param use_fp16: enable FP16 for providers that support it (useful on Jetson/TensorRT)
        :param cache_dir: custom cache directory; defaults to ~/.cache/jetseg
        :param backend: 'onnx' (default) to use ONNX Runtime, or 'torch' to use local PyTorch model
        :param torch_model_path: path to a PyTorch checkpoint (.pth/.pt) or scripted module when using backend='torch'
        :param input_size: tuple (W,H) to override model input size
        :param torch_device: explicit torch device name (e.g., 'cuda' or 'cpu') if using torch backend
        """
        # 1. locate bundled ONNX model inside the package
        current_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(current_dir, "human_seg.onnx")

        # allow overriding expected input size
        if input_size is not None:
            self.input_size = tuple(input_size)
        else:
            self.input_size = (224, 224)

        # 2. Cache configuration
        if cache_dir is None:
            # Lấy đường dẫn Home của User (ví dụ: /home/orin/)
            home_dir = os.path.expanduser("~")
            # Tạo đường dẫn chuẩn: /home/orin/.cache/jetseg
            cache_dir = os.path.join(home_dir, ".cache", "jetseg")

        # create directory if missing; log at DEBUG level to avoid noisy startup prints
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir, exist_ok=True)
                logger.debug("Created cache directory: %s", cache_dir)
            except Exception:
                logger.warning("Unable to create cache at %s; using /tmp/jetseg_cache", cache_dir)
                cache_dir = "/tmp/jetseg_cache"
                os.makedirs(cache_dir, exist_ok=True)
        else:
            logger.debug("Using cache at: %s", cache_dir)

        # 3. Cấu hình TensorRT Provider
        trt_options = {
            'trt_fp16_enable': use_fp16,
            'trt_int8_enable': False,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': cache_dir,  # Trỏ về cache tập trung
            'trt_max_workspace_size': 2147483648,  # 2GB RAM build engine
        }

        # ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # backend selection: 'onnx' or 'torch'
        self.backend = str(backend).lower()
        self.torch_model = None
        self.torch_device = None

        if self.backend == "onnx":
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"CRITICAL: ONNX model not found at {self.model_path}")

            # choose available providers
            providers = ort.get_available_providers()
            trt_provider_name = next((p for p in providers if 'Tensorrt' in p or 'TensorRT' in p), None)

            logger.debug("Loading ONNX model: %s", self.model_path)

            if trt_provider_name:
                logger.info("Using TensorRT provider (FP16=%s)", use_fp16)
                # if cache empty, inform user (at INFO level)
                if not os.listdir(cache_dir):
                    logger.info("First run: building TensorRT engine (may take 1-2 minutes).")

                self.session = ort.InferenceSession(self.model_path, providers=[(trt_provider_name, trt_options)], sess_options=sess_options)
            elif 'CUDAExecutionProvider' in providers:
                logger.info("Using CUDAExecutionProvider for ONNX Runtime")
                self.session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'], sess_options=sess_options)
            else:
                logger.info("Falling back to CPUExecutionProvider for ONNX Runtime")
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'], sess_options=sess_options)

            self.input_name = self.session.get_inputs()[0].name

        elif self.backend == "torch":
            if torch is None:
                raise RuntimeError("PyTorch not available in this environment for backend='torch'")

            # device selection
            self.torch_device = torch_device or ("cuda" if torch.cuda.is_available() else "cpu")

            # load model if path provided
            if torch_model_path:
                tm = Path(torch_model_path)
                if not tm.exists():
                    raise FileNotFoundError(f"Torch model not found: {tm}")
                try:
                    # try to load scripted module first
                    try:
                        self.torch_model = torch.jit.load(str(tm), map_location=self.torch_device)
                    except Exception:
                        # load state_dict into provided UNetOptimized if available
                        from .models.optimized_unet import UNetOptimized
                        model = UNetOptimized(in_channels=3, out_channels=1)
                        data = torch.load(str(tm), map_location="cpu")
                        if isinstance(data, dict) and "model_state_dict" in data:
                            state = data["model_state_dict"]
                        elif isinstance(data, dict) and "state_dict" in data:
                            state = data["state_dict"]
                        else:
                            state = data
                        model.load_state_dict(state)
                        self.torch_model = model.to(self.torch_device).eval()
                except Exception as e:
                    raise RuntimeError(f"Failed to load torch model: {e}")
            else:
                # no model path provided: use default lightweight model
                try:
                    from .models.optimized_unet import UNetOptimized
                    self.torch_model = UNetOptimized(in_channels=3, out_channels=1).to(self.torch_device).eval()
                except Exception as e:
                    raise RuntimeError(f"Failed to instantiate default torch model: {e}")

            logger.info("Using PyTorch backend on device %s", self.torch_device)

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def predict(self, image, threshold=0.5):
        if image is None:
            return None
        h_orig, w_orig = image.shape[:2]

        if self.backend == "torch":
            # PyTorch inference path
            img_resized = cv2.resize(image, self.input_size)
            img_norm = img_resized.astype(np.float32) / 255.0
            # convert HWC (BGR) -> CHW (keep BGR order to match user's preprocessing expectations)
            tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.torch_device)
            with torch.no_grad():
                out = self.torch_model(tensor)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                # assume logits -> probabilities
                out_np = torch.sigmoid(out).cpu().numpy()
                pred_mask = out_np[0]
                # remove channel dim if present -> (C,H,W) or (H,W)
                if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
                    pred_mask = pred_mask[0]
                elif pred_mask.ndim == 3:
                    # take first channel if multiple
                    pred_mask = pred_mask[0]
            pred_mask = cv2.resize(pred_mask, (w_orig, h_orig))
            return (pred_mask > threshold).astype(np.uint8) * 255

        # ONNX Runtime path (existing behavior)
        img_resized = cv2.resize(image, self.input_size)
        img_norm = img_resized.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img_norm, axis=0)

        outputs = self.session.run(None, {self.input_name: input_tensor})

        pred_mask = outputs[0][0]
        pred_mask = cv2.resize(pred_mask, (w_orig, h_orig))

        return (pred_mask > threshold).astype(np.uint8) * 255

    def remove_background(self, image, mask, bg_color=(0, 255, 0)):
        green_bg = np.zeros_like(image)
        green_bg[:] = bg_color
        mask_3ch = np.expand_dims(mask > 0, axis=-1)
        return np.where(mask_3ch, image, green_bg)
