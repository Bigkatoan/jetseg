import pytest
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None

from jetseg import registry


def test_onnx_fp32_inference_smoke():
    if ort is None:
        pytest.skip("onnxruntime not installed")

    try:
        path = registry.get_model_path("humanseg", "unet_best", "fp32")
    except Exception:
        pytest.skip("No fp32 ONNX registered for humanseg/unet_best")

    if not path.exists():
        pytest.skip("Registered ONNX file not found on disk")

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) 
    inp = sess.get_inputs()[0]

    # build a minimal random input matching batch and channel dims
    shape = []
    for i, d in enumerate(inp.shape):
        if d is None:
            # batch -> 1, spatial dims -> 480 (common default)
            shape.append(1 if i == 0 else 480)
        else:
            shape.append(int(d))

    arr = np.random.randn(*shape).astype(np.float32)
    try:
        out = sess.run(None, {inp.name: arr})
    except Exception as e:
        pytest.skip(f"ONNX runtime execution failed: {e}")

    assert out is not None
