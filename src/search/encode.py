"""G2 / G4: CLIP encoders. Image or text in -> L2-normalised vector out. Nothing else.

CLIP here only converts: it does not store, search or remember anything, and a vector
cannot be turned back into an image. Search is FAISS (G3) + the lookup table.

Backends for the image encoder:
  torch_cpu     open_clip model as is (reference)
  openvino_gpu  model.visual converted with ov.convert_model, compiled on the Intel GPU
                (IR cached under {index_root}/_ov_cache/). Fallback: openvino_cpu, then torch_cpu.
Text is encoded with torch on CPU (a few prompts per query, fast enough).

Every image goes through pad_square (OI-08) before the model's own preprocess, so the
model's resize/center-crop never drops the image's left and right edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.common.config import CONFIG_DIR, load_yaml
from src.search.tiling import pad_square

MODELS = load_yaml(CONFIG_DIR / "clip_models.yaml")["models"]


@dataclass
class Clip:
    model_id: str
    open_clip_name: str
    pretrained: str
    model: torch.nn.Module
    preprocess: object
    tokenizer: object
    input_res: int
    embed_dim: int
    backend: str = "torch_cpu"
    _ov_visual: object = field(default=None, repr=False)

    # ---------- image ----------
    def prep(self, imgs: list[Image.Image]) -> torch.Tensor:
        """PIL crops -> model input tensor (N, 3, R, R). pad_square first."""
        return torch.stack([self.preprocess(pad_square(im.convert("RGB"))) for im in imgs])

    @torch.no_grad()
    def encode_pixels(self, x: torch.Tensor) -> np.ndarray:
        if self._ov_visual is not None:
            v = torch.from_numpy(self._run_ov(x.numpy()))
        elif self.backend == "torch_cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                v = self.model.encode_image(x.cuda(non_blocking=True)).float().cpu()
        else:
            v = self.model.encode_image(x).float()
        return torch.nn.functional.normalize(v, dim=-1).numpy()

    def _run_ov(self, x: np.ndarray) -> np.ndarray:
        """Static-batch OpenVINO model: feed chunks of ov_batch, zero-pad the last one."""
        b = self._ov_batch
        outs = []
        for i in range(0, len(x), b):
            chunk = x[i:i + b]
            n = len(chunk)
            if n < b:
                chunk = np.concatenate([chunk, np.zeros((b - n, *chunk.shape[1:]), chunk.dtype)])
            outs.append(np.asarray(self._ov_visual(chunk)[0], dtype=np.float32)[:n])
        return np.concatenate(outs)

    def encode_images(self, imgs: list[Image.Image]) -> np.ndarray:
        return self.encode_pixels(self.prep(imgs))

    # ---------- text ----------
    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        dev = next(self.model.parameters()).device
        v = self.model.encode_text(self.tokenizer(texts).to(dev)).float().cpu()
        return torch.nn.functional.normalize(v, dim=-1).numpy()

    # ---------- backend ----------
    def to_cuda(self) -> str:
        """Colab path: model on the NVIDIA GPU, image encoder under fp16 autocast."""
        self.model.cuda()
        self._ov_visual = None
        self.backend = "torch_cuda"
        return self.backend

    def to_openvino(self, cache_dir: Path, device: str = "GPU", batch: int = 32,
                    precision: str | None = None) -> str:
        """Compile model.visual with OpenVINO on `device` (static batch). Returns the backend name.

        precision: OpenVINO INFERENCE_PRECISION_HINT ("f32"/"f16"); None = device default (f16 on GPU).
        SO400M crashes on the Intel GPU in f16 (access violation in the GPU plugin); f32 runs.
        """
        import openvino as ov
        cache_dir.mkdir(parents=True, exist_ok=True)
        xml = cache_dir / f"{self.model_id}_visual_b{batch}.xml"
        if not xml.exists():
            r = self.input_res
            # open_clip ViTs fail torch.jit.trace's sanity re-run ("graphs differed"); trace without
            # it. Correctness is checked instead by the torch-vs-OpenVINO cosine parity test.
            # trace with batch 1 (a batch-32 trace of SO400M at 384 px exhausts RAM), then set the batch in OV
            with torch.no_grad():
                traced = torch.jit.trace(self.model.visual, torch.zeros(1, 3, r, r), check_trace=False)
            ov_model = ov.convert_model(traced)
            del traced
            ov_model.reshape([batch, 3, r, r])
            ov.save_model(ov_model, str(xml))           # weights stored as fp16 by default
        cfg = {"INFERENCE_PRECISION_HINT": precision} if precision else {}
        self._ov_visual = ov.Core().compile_model(str(xml), device, cfg)
        self._ov_batch = batch
        self.backend = f"openvino_{device.lower()}" + (f"_{precision}" if precision else "")
        return self.backend

    def to_torch(self) -> None:
        self._ov_visual = None
        self.backend = "torch_cpu"


def local_weights(cfg: dict, weights_dir: Path | None) -> Path | None:
    """{clip_weights_dir}/{open_clip_name}__{pretrained}.safetensors if it exists, else None."""
    if weights_dir is None:
        return None
    p = Path(weights_dir) / f"{cfg['open_clip_name']}__{cfg['pretrained']}.safetensors"
    return p if p.exists() else None


def cached_weights(pretrained_cfg: dict) -> Path | None:
    """Reuse a complete HF cache entry without remote HEAD/version checks."""
    hub = pretrained_cfg.get("hf_hub", "")
    if not hub:
        return None
    from huggingface_hub import try_to_load_from_cache
    from open_clip.constants import HF_SAFE_WEIGHTS_NAME, HF_WEIGHTS_NAME
    repo, filename = hub.rsplit("/", 1) if "/" in hub else (hub, "")
    filename = filename or HF_WEIGHTS_NAME
    candidates = []
    if filename == HF_WEIGHTS_NAME:
        candidates.append(HF_SAFE_WEIGHTS_NAME)
    elif filename.endswith((".bin", ".pth")):
        candidates.append(filename[:-4] + ".safetensors")
    for name in [*candidates, filename]:
        cached = try_to_load_from_cache(repo, name)
        if isinstance(cached, str) and Path(cached).is_file():
            return Path(cached)
    return None


def load_tokenizer(model_name: str):
    """Prefer the cached HF tokenizer; retain normal download on a cache miss."""
    import open_clip
    cfg = open_clip.get_model_config(model_name) or {}
    text_cfg = cfg.get("text_cfg", {})
    if tokenizer_repo := text_cfg.get("hf_tokenizer_name"):
        from huggingface_hub import try_to_load_from_cache
        from open_clip.tokenizer import HFTokenizer
        cached = try_to_load_from_cache(tokenizer_repo, "tokenizer_config.json")
        if isinstance(cached, str):
            try:
                # A directory also prevents optional tokenizer probes from looking up the hub.
                return HFTokenizer(str(Path(cached).parent), context_length=text_cfg.get("context_length"),
                                   tokenizer_mode=text_cfg.get("tokenizer_mode"),
                                   **text_cfg.get("tokenizer_kwargs", {}), local_files_only=True)
            except OSError:
                pass
        try:
            return open_clip.get_tokenizer(model_name, local_files_only=True)
        except OSError:
            pass
    return open_clip.get_tokenizer(model_name)


def load_clip(model_id: str, weights_dir: Path | None = None) -> Clip:
    """Prefer explicit local weights, then the HF cache; download only on a cache miss.

    A file path makes open_clip forget the pretrained tag's preprocess (e.g. SigLIP uses
    mean/std 0.5 and 'squash' resize), so those are passed explicitly from the tag's config.
    """
    import open_clip
    cfg = MODELS[model_id]
    pc = open_clip.get_pretrained_cfg(cfg["open_clip_name"], cfg["pretrained"])
    local = local_weights(cfg, weights_dir) or cached_weights(pc)
    if local is None:
        model, _, preprocess = open_clip.create_model_and_transforms(cfg["open_clip_name"],
                                                                     pretrained=cfg["pretrained"])
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["open_clip_name"], pretrained=str(local), image_mean=pc.get("mean"), image_std=pc.get("std"),
            image_interpolation=pc.get("interpolation"), image_resize_mode=pc.get("resize_mode"))
    model.eval()
    return Clip(model_id, cfg["open_clip_name"], cfg["pretrained"], model, preprocess,
                load_tokenizer(cfg["open_clip_name"]), cfg["input_res"], cfg["embed_dim"])
