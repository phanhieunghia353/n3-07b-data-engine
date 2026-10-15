import numpy as np
import pandas as pd
import pytest

import faiss

def assemble(vectors, lookup):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index

from src.search.search import Index, aggregate, search_vectors


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _toy_index(n_tiles=3):
    # 3 images x 3 tiles in 2-D. image "b" owns the single best crop (tile 2) for direction x.
    vecs = _unit([[0, 1], [0.2, 1], [0.1, 1],      # a
                  [0.5, 1], [0.3, 1], [1, 0.05],  # b
                  [0.6, 1], [0.6, 1], [0, 1]])    # c: tiles 0 and 1 identical
    lookup = pd.DataFrame({"row_id": np.arange(9), "image_id": np.repeat(["a", "b", "c"], n_tiles),
                           "tile_idx": np.tile(np.arange(n_tiles), 3)})
    meta = {"model_id": "toy", "n_tiles": n_tiles, "n_vectors": 9}
    return Index(None, meta, assemble(vecs, lookup), lookup)


def test_max_over_tiles_and_best_tile():
    idx = _toy_index()
    r = search_vectors(idx, _unit([1, 0]), top_n=None)
    assert list(r["image_id"]) == ["b", "c", "a"]
    assert r.loc[0, "best_tile_idx"] == 2
    assert list(r["rank"]) == [1, 2, 3]
    assert r["score"].is_monotonic_decreasing


def test_tie_breaks_lower_tile_then_image_id():
    # c tiles 0 and 1 have identical vectors -> best_tile_idx must be 0
    idx = _toy_index()
    r = search_vectors(idx, _unit([1, 0]), top_n=None).set_index("image_id")
    assert r.loc["c", "best_tile_idx"] == 0
    scores = np.array([[0.5, 0.5]]); rows = np.array([[5, 0]])       # b and a tie on score
    out = aggregate(scores, rows, idx.lookup)
    assert list(out["image_id"]) == ["a", "b"]                     # image_id asc


def test_max_over_prompts_records_best_prompt():
    idx = _toy_index()
    r = search_vectors(idx, _unit([[1, 0], [0, 1]]), top_n=None, query_ids=["px", "py"]).set_index("image_id")
    assert r.loc["b", "best_query_id"] == "px"                     # b wins on x
    assert r.loc["a", "best_query_id"] == "py"
    assert r.loc["a", "score"] == pytest.approx(1.0, abs=1e-6)


def test_topn_with_k_equal_n_tiles_times_n_is_exact():
    rng = np.random.default_rng(0)
    n_img, n_tiles = 200, 7
    vecs = _unit(rng.normal(size=(n_img * n_tiles, 16)))
    lookup = pd.DataFrame({"row_id": np.arange(n_img * n_tiles), "image_id": np.repeat([f"i{k:03d}" for k in
                           range(n_img)], n_tiles), "tile_idx": np.tile(np.arange(n_tiles), n_img)})
    idx = Index(None, {"model_id": "m", "n_tiles": n_tiles, "n_vectors": len(vecs)}, assemble(vecs, lookup), lookup)
    q = _unit(rng.normal(size=(3, 16)))
    full = search_vectors(idx, q, top_n=None).head(20)
    fast = search_vectors(idx, q, top_n=20)
    pd.testing.assert_frame_equal(full.reset_index(drop=True), fast.reset_index(drop=True))


def test_model_mismatch_raises():
    with pytest.raises(ValueError, match="query model"):
        _toy_index().check_model("other")
