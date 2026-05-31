import json
from pathlib import Path


def test_manifest_and_find_cached_variant(tmp_path, monkeypatch):
    # write a small fake model file
    model = tmp_path / "fake_model.onnx"
    model.write_bytes(b"fake-model-contents")

    # import the module under test
    from jetseg import quantize

    sha = quantize.compute_model_key(str(model))

    cache = tmp_path / "cache"
    assert quantize.find_cached_variant(str(model), "int8", str(cache)) is None

    # create a fake cached file and manifest
    cached_file = cache / f"{model.stem}_int8_{sha[:8]}.onnx"
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    cached_file.write_bytes(b"fake-quant")

    manifest = {sha: {"int8": {"filename": cached_file.name, "created_at": 0}}}
    quantize.save_manifest(str(cache), manifest)

    found = quantize.find_cached_variant(str(model), "int8", str(cache))
    assert found is not None
    assert found.exists()
