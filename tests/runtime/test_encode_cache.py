"""Cold-start loading must use cached assets without changing model preprocessing."""
from types import SimpleNamespace

import huggingface_hub
import open_clip
from open_clip import tokenizer as clip_tokenizer
import pytest

from src.search import encode


def test_cached_assets_skip_network_and_preserve_preprocessing(tmp_path, monkeypatch):
    weights = tmp_path / "open_clip_model.safetensors"
    weights.touch()
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name:
                        str(weights) if repo == "test/model" and name == weights.name else None)
    pretrained = {"hf_hub": "test/model/", "mean": (0.5,) * 3, "std": (0.5,) * 3,
                  "interpolation": "bicubic", "resize_mode": "squash"}
    monkeypatch.setattr(open_clip, "get_pretrained_cfg", lambda *args: pretrained)
    calls = []
    def create(name, *, pretrained, **kwargs):
        assert pretrained == str(weights)  # A tag would trigger an unnecessary remote check.
        calls.append(kwargs)
        return SimpleNamespace(eval=lambda: None), None, "preprocess"
    monkeypatch.setattr(open_clip, "create_model_and_transforms", create)
    sentinel = object()
    def tokenizer(name, **kwargs):
        assert kwargs == {"local_files_only": True}
        return sentinel
    monkeypatch.setattr(open_clip, "get_tokenizer", tokenizer)
    result = encode.load_clip("siglip384", tmp_path)
    assert result.tokenizer is sentinel
    assert calls == [{"image_mean": (0.5,) * 3, "image_std": (0.5,) * 3,
                      "image_interpolation": "bicubic", "image_resize_mode": "squash"}]


@pytest.mark.parametrize("model_name,cache_miss", [("ViT-B-16-SigLIP-384", True), ("ViT-B-32", False)])
def test_tokenizer_cache_miss_and_non_hf_tokenizer(monkeypatch, model_name, cache_miss):
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *args: None)
    calls = []
    sentinel = object()
    def tokenizer(name, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("not cached yet")
        return sentinel
    monkeypatch.setattr(open_clip, "get_tokenizer", tokenizer)
    assert encode.load_tokenizer(model_name) is sentinel
    assert calls == ([{"local_files_only": True}, {}] if cache_miss else [{}])


def test_cached_tokenizer_directory_keeps_siglip_configuration(tmp_path, monkeypatch):
    cached_config = tmp_path / "tokenizer_config.json"
    cached_config.touch()
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *args: str(cached_config))
    sentinel = object()
    def tokenizer(path, **kwargs):
        assert path == str(tmp_path)
        assert kwargs == {"context_length": 64, "tokenizer_mode": None,
                          "clean": "canonicalize", "local_files_only": True}
        return sentinel
    monkeypatch.setattr(clip_tokenizer, "HFTokenizer", tokenizer)
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda *args, **kwargs: pytest.fail("unexpected hub fallback"))
    assert encode.load_tokenizer("ViT-B-16-SigLIP-384") is sentinel
