import numpy as np
import pytest
from PIL import Image

from src.common.config import CONFIG_DIR, load_yaml
from src.search.tiling import crops, pad_square, tile_boxes

TILINGS = load_yaml(CONFIG_DIR / "tiling.yaml")["tilings"]


def test_whole_is_single_full_box():
    assert tile_boxes(1280, 720, TILINGS["whole"]) == [(0, 0, 1280, 720)]


def test_t3x2o20_geometry():
    boxes = tile_boxes(1280, 720, TILINGS["t3x2o20"])
    assert len(boxes) == 7 and boxes[0] == (0, 0, 1280, 720)
    grid = boxes[1:]
    for x1, y1, x2, y2 in grid:
        assert 0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720
        assert abs((x2 - x1) - 1280 / 2.6) <= 1 and abs((y2 - y1) - 400) <= 1
    # row-major order: first row top, left to right
    assert [b[1] for b in grid] == [0, 0, 0, 320, 320, 320]
    assert [b[0] for b in grid[:3]] == sorted(b[0] for b in grid[:3])
    # neighbours overlap ~20% of tile width / height
    w = grid[0][2] - grid[0][0]
    assert abs((grid[0][2] - grid[1][0]) / w - 0.2) < 0.01
    assert abs((grid[0][3] - grid[3][1]) / 400 - 0.2) < 0.01


def test_grid_covers_every_pixel():
    cover = np.zeros((720, 1280), dtype=int)
    for x1, y1, x2, y2 in tile_boxes(1280, 720, TILINGS["t3x2o20"])[1:]:
        cover[y1:y2, x1:x2] += 1
    assert cover.min() >= 1


def test_n_tiles_mismatch_raises():
    bad = dict(TILINGS["t3x2o20"], n_tiles=6)
    with pytest.raises(ValueError):
        tile_boxes(1280, 720, bad)


def test_crops_and_pad_square():
    img = Image.new("RGB", (1280, 720), (200, 100, 50))
    cs = crops(img, TILINGS["t3x2o20"])
    assert [c.size for c in cs][0] == (1280, 720)
    sq = pad_square(cs[0])
    assert sq.size == (1280, 1280)
    a = np.asarray(sq)
    assert (a[:280] == 0).all() and (a[1000:] == 0).all()        # 280 px black bands top/bottom
    assert (a[280:1000] == [200, 100, 50]).all()                 # content kept, centred
    assert pad_square(Image.new("RGB", (400, 400))).size == (400, 400)
