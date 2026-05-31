import pytest

from jetseg import registry


def test_registry_has_humanseg():
    reg = registry.load_registry()
    assert "humanseg" in reg
    models = registry.list_models("humanseg")
    assert isinstance(models, list)
    # expect at least the converted models present
    assert any(m.startswith("unet_") for m in models)
