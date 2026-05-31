import logging
import onnxruntime as ort
import numpy as np
import cv2
import os
import sys
from pathlib import Path

try:
    import torch
except Exception:
    torch = None

# package logger
logger = logging.getLogger("jetseg")


class HumanSeg:
    def __init__(self, use_fp16=True, cache_dir=None, backend: str = "onnx", torch_model_path: str | None = None, input_size: tuple | None = None, torch_device: str | None = None, model_path: str | None = None, auto_quantize: bool = True, no_quantize: bool = False):
        """
        Initialize JetSeg HumanSeg engine.
        :param use_fp16: enable FP16 for providers that support it (useful on Jetson/TensorRT)
        :param cache_dir: custom cache directory; defaults to ~/.cache/jetseg
        :param backend: 'onnx' (default) to use ONNX Runtime, or 'torch' to use local PyTorch model
        :param torch_model_path: path to a PyTorch checkpoint (.pth/.pt) or scripted module when using backend='torch'
        :param input_size: tuple (W,H) to override model input size
        :param torch_device: explicit torch device name (e.g., 'cuda' or 'cpu') if using torch backend
        """
        # 1. locate bundled ONNX model inside the package (can be overridden)
        current_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(current_dir, "human_seg.onnx")
        # quantize flags
        self.auto_quantize = bool(auto_quantize)
        self.no_quantize = bool(no_quantize)
        if model_path is not None:
            # allow callers to pass an explicit ONNX/PTH path
            self.model_path = str(model_path)
        else:
            # Best-effort: prefer a registry-provided default model for the `humanseg` task
            # This keeps backward-compatibility if the registry is missing or resolution fails.
            try:
                from . import registry
                reg = registry.load_registry()
                model_name = reg.get("humanseg", {}).get("default")
                if model_name:
                    variant = registry.get_default_variant("humanseg", model_name)
                    mp = registry.get_model_path("humanseg", model_name, variant)
                    if mp.exists():
                        self.model_path = str(mp)
                        logger.debug("Using registry default model for humanseg: %s", self.model_path)
            except Exception:
                # keep bundled model if registry isn't available or resolution fails
                pass

        def _choose_onnx_variant(base_path: str, prefer_fp16: bool) -> str:
            """Look for device-optimized ONNX variants next to the base model.

            Priority (best-effort):
              - If TensorRT available and fp16 variant exists -> use fp16
              - If prefer_fp16 and fp16 exists -> use fp16
              - If CPU-only and int8 exists -> use int8
              - If fp32 variant exists -> use it
              - Else return base_path
            """
            from pathlib import Path

            base = Path(base_path)
            stem = base.stem
            dirp = base.parent

            fp16 = dirp / f"{stem}_fp16.onnx"
            fp32 = dirp / f"{stem}_fp32.onnx"
            int8 = dirp / f"{stem}_int8.onnx"
            legacy = base

            # check providers
            try:
                providers = ort.get_available_providers()
            except Exception:
                providers = []

            trt_provider = next((p for p in providers if 'Tensorrt' in p or 'TensorRT' in p), None)

            # TensorRT + fp16
            if trt_provider and fp16.exists():
                logger.info("Using model variant: %s (TensorRT/FP16)", fp16.name)
                return str(fp16)

            # prefer fp16 if requested
            if prefer_fp16 and fp16.exists():
                logger.info("Using model variant: %s (FP16)", fp16.name)
                return str(fp16)

            # CPU prefer int8
            if 'CPUExecutionProvider' in providers and int8.exists():
                logger.info("Using model variant: %s (INT8)", int8.name)
                return str(int8)

            if fp32.exists():
                logger.info("Using model variant: %s (FP32)", fp32.name)
                return str(fp32)

            # fallback to given base
            return str(legacy)


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
            # attempt auto-quantization if enabled (may replace self.model_path)
            if self.auto_quantize and not self.no_quantize:
                try:
                    from . import quantize as quantize_mod
                except Exception:
                    quantize_mod = None

                if quantize_mod is not None:
                    try:
                        qpath = quantize_mod.ensure_or_prompt_quantized(self.model_path, prefer_fp16=use_fp16, cache_dir=cache_dir, interactive=sys.stdin.isatty())
                        if qpath is not None and Path(str(qpath)).exists() and str(qpath) != self.model_path:
                            self.model_path = str(qpath)
                    except Exception as e:
                        logger.warning("Quantization step failed or skipped: %s", e)

            # allow selecting fp16/int8/fp32 variants if present
            self.model_path = _choose_onnx_variant(self.model_path, use_fp16)
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

            # record input metadata (name and dtype) for proper input casting
            try:
                input_meta = self.session.get_inputs()[0]
                self.input_name = input_meta.name
                # type is a string like 'tensor(float)'
                self.input_dtype = str(input_meta.type)
                # capture input shape info for layout/resize decisions
                try:
                    raw_shape = list(input_meta.shape)
                    # normalize to ints or None
                    self.session_input_shape = tuple(int(x) if isinstance(x, int) else None for x in raw_shape)
                except Exception:
                    self.session_input_shape = None
                # infer layout
                layout = None
                if self.session_input_shape and len(self.session_input_shape) == 4:
                    if self.session_input_shape[1] == 3:
                        layout = 'NCHW'
                    elif self.session_input_shape[3] == 3:
                        layout = 'NHWC'
                self.session_input_layout = layout
            except Exception:
                # fallback
                self.input_name = self.session.get_inputs()[0].name
                self.input_dtype = 'tensor(float)'
                self.session_input_shape = None
                self.session_input_layout = None

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

    @classmethod
    def from_registry(cls, task: str = "humanseg", model_name: str | None = None, variant: str | None = None, backend: str = "onnx", use_fp16: bool = True, cache_dir: str | None = None, torch_device: str | None = None, input_size: tuple | None = None):
        """Construct a HumanSeg instance from the package model registry.

        :param task: task name in the registry (e.g., 'humanseg')
        :param model_name: specific model name (if omitted uses registry default)
        :param variant: variant to use (fp32/fp16/int8/pth). If omitted the registry default is chosen.
        :param backend: 'onnx' or 'torch'
        """
        try:
            from . import registry
        except Exception as e:
            raise RuntimeError(f"Failed to import registry: {e}")

        # resolve model_name default
        reg = registry.load_registry()
        if task not in reg:
            raise KeyError(f"Unknown task in registry: {task}")
        task_entry = reg[task]
        if model_name is None:
            model_name = task_entry.get("default")

        # resolve variant
        if variant is None:
            variant = registry.get_default_variant(task, model_name)

        # get artifact path
        model_path = registry.get_model_path(task, model_name, variant)

        # instantiate
        if backend == "onnx":
            return cls(use_fp16=use_fp16, cache_dir=cache_dir, backend="onnx", input_size=input_size, torch_device=torch_device, model_path=str(model_path))
        elif backend == "torch":
            # prefer a pth variant for torch backend
            if model_path.suffix in (".pt", ".pth"):
                return cls(use_fp16=use_fp16, cache_dir=cache_dir, backend="torch", torch_model_path=str(model_path), input_size=input_size, torch_device=torch_device)
            else:
                # try to locate pth variant
                try:
                    pth_path = registry.get_model_path(task, model_name, "pth")
                    return cls(use_fp16=use_fp16, cache_dir=cache_dir, backend="torch", torch_model_path=str(pth_path), input_size=input_size, torch_device=torch_device)
                except Exception:
                    raise RuntimeError("Requested torch backend but no .pt/.pth variant available for the selected model")
        else:
            raise ValueError(f"Unsupported backend: {backend}")

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
        # determine target resize from session expected shape if available
        target_w, target_h = None, None
        try:
            if getattr(self, "session_input_shape", None) and getattr(self, "session_input_layout", None):
                s = self.session_input_shape
                if self.session_input_layout == 'NCHW':
                    # shape like (N, C, H, W)
                    _, _, h, w = s
                    target_w, target_h = (w, h)
                elif self.session_input_layout == 'NHWC':
                    # shape like (N, H, W, C)
                    _, h, w, _ = s
                    target_w, target_h = (w, h)
        except Exception:
            target_w, target_h = None, None

        if target_w and target_h:
            img_resized = cv2.resize(image, (int(target_w), int(target_h)))
        else:
            img_resized = cv2.resize(image, self.input_size)

        img_norm = img_resized.astype(np.float32) / 255.0

        # transpose to NCHW if needed
        if getattr(self, "session_input_layout", None) == 'NCHW':
            img_chw = img_norm.transpose(2, 0, 1)
            input_tensor = np.expand_dims(img_chw, axis=0)
        else:
            input_tensor = np.expand_dims(img_norm, axis=0)

        # cast input to expected dtype if session expects float16
        try:
            if hasattr(self, "input_dtype") and "float16" in str(self.input_dtype).lower():
                input_tensor = input_tensor.astype(np.float16)
        except Exception:
            pass

        outputs = self.session.run(None, {self.input_name: input_tensor})

        # robustly extract predicted mask from outputs
        res = outputs[0]
        pred_mask = None
        try:
            import numpy as _np

            if isinstance(res, _np.ndarray):
                if res.ndim == 4:
                    # (N,C,H,W)
                    arr = res[0]
                    if arr.ndim == 3:
                        # (C,H,W) -> take channel 0
                        pred_mask = arr[0]
                    else:
                        pred_mask = _np.squeeze(arr)
                elif res.ndim == 3:
                    # could be (N,H,W) or (C,H,W)
                    if res.shape[0] == 1:
                        pred_mask = res[0]
                    else:
                        pred_mask = res[0]
                elif res.ndim == 2:
                    pred_mask = res
                else:
                    pred_mask = _np.squeeze(res)
            else:
                pred_mask = _np.array(res)
        except Exception:
            pred_mask = None

        if pred_mask is None:
            raise RuntimeError("Unable to interpret model output for prediction")

        # ensure 2D and convert to float32 (opencv may not support float16)
        try:
            if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
                pred_mask = pred_mask[0]
            pred_mask = np.ascontiguousarray(pred_mask.astype(np.float32))
            # if outputs look like logits (outside [0,1]), apply sigmoid
            try:
                mx = float(np.nanmax(pred_mask))
                mn = float(np.nanmin(pred_mask))
                if mx > 1.01 or mn < -0.01:
                    pred_mask = 1.0 / (1.0 + np.exp(-pred_mask))
            except Exception:
                pass
            # final resize to original image size
            pred_mask = cv2.resize(pred_mask, (w_orig, h_orig))
        except Exception as e:
            raise RuntimeError(f"Failed to resize prediction map: {e}") from e

        return (pred_mask > threshold).astype(np.uint8) * 255

    def remove_background(self, image, mask, bg_color=(0, 255, 0)):
        green_bg = np.zeros_like(image)
        green_bg[:] = bg_color
        mask_3ch = np.expand_dims(mask > 0, axis=-1)
        return np.where(mask_3ch, image, green_bg)
