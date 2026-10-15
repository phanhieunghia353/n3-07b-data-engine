"""Runtime backend selection, independent of index-building and training scripts."""


def set_backend(clip, backend: str, ov_cache, batch: int) -> None:
    """Keep model weights unchanged; only configure the image encoder's execution backend."""
    if backend == "torch_cuda":
        clip.to_cuda()
    elif backend.startswith("openvino"):
        parts = backend.split("_")
        clip.to_openvino(ov_cache, device=parts[1].upper(), batch=batch,
                         precision=parts[2] if len(parts) > 2 else None)
