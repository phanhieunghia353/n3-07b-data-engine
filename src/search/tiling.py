"""G1: cut one 1280x720 image into crops, in memory (nothing written to disk).

In:  PIL image + a TILING_CONFIG entry from configs/tiling.yaml
Out: list of crops; crop k <-> tile_idx k. tile 0 = whole image (when include_whole),
     then the grid row by row, left to right.
     t3x2o20 -> 7 crops: whole + 6 tiles of ~492x400 px, neighbours overlap 20%.

pad_square() is the preprocess chosen in OI-08: the crop is centred on a square canvas
(fill = black) so the CLIP preprocess (resize shortest side + center crop) drops nothing.
"""
from __future__ import annotations

from PIL import Image

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels, x2/y2 exclusive


def _axis_spans(length: int, n: int, overlap: float) -> list[tuple[int, int]]:
    """n spans covering [0, length) where neighbours share `overlap` of a span's size."""
    size = length / (n - (n - 1) * overlap)
    stride = size * (1 - overlap)
    spans = []
    for i in range(n):
        a = round(i * stride)
        b = length if i == n - 1 else round(i * stride + size)
        spans.append((a, b))
    return spans


def tile_boxes(width: int, height: int, cfg: dict) -> list[Box]:
    """Pixel boxes for every tile_idx of one tiling config."""
    boxes: list[Box] = []
    if cfg["include_whole"]:
        boxes.append((0, 0, width, height))
    if cfg["grid_cols"] and cfg["grid_rows"]:
        xs = _axis_spans(width, cfg["grid_cols"], cfg["overlap"])
        ys = _axis_spans(height, cfg["grid_rows"], cfg["overlap"])
        boxes += [(x1, y1, x2, y2) for (y1, y2) in ys for (x1, x2) in xs]
    if len(boxes) != cfg["n_tiles"]:
        raise ValueError(f"tiling gives {len(boxes)} tiles, config says n_tiles={cfg['n_tiles']}")
    return boxes


def crops(img: Image.Image, cfg: dict) -> list[Image.Image]:
    return [img.crop(b) for b in tile_boxes(img.width, img.height, cfg)]


def pad_square(img: Image.Image, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Centre img on a square canvas of side max(w, h)."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img.convert("RGB"), ((side - w) // 2, (side - h) // 2))
    return canvas
