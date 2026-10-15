"""G5: search an index with query vectors and aggregate crop hits into an image ranking.

Gear chain (every query):
  prompts (1-5 sentences) or an example image
   -> G4 CLIP encoder of the SAME model as the index -> query vectors (P x dim)
   -> FAISS index.search: for each query vector, the k nearest crop rows (row_id, score)
   -> lookup.parquet: row_id -> (image_id, tile_idx)
   -> image score = max over its tiles and over the prompts; keep best_tile_idx, best_query
   -> rank by score desc, ties by image_id asc
FAISS only finds the nearest vectors fast; it never sees images or text.

k_vectors: to get the exact top-N images, n_tiles * N vectors per prompt suffice (each image
owns at most n_tiles rows). tool_select uses k_vectors = ntotal (full scan).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


@dataclass
class Index:
    folder: Path
    meta: dict
    index: faiss.Index
    lookup: pd.DataFrame

    @classmethod
    def load(cls, folder: Path) -> "Index":
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        index = faiss.read_index(str(folder / "index.faiss"))
        lookup = pd.read_parquet(folder / "lookup.parquet")
        if not (index.ntotal == len(lookup) == meta["n_vectors"]):
            raise ValueError(f"{folder}: ntotal {index.ntotal}, lookup {len(lookup)}, meta {meta['n_vectors']}")
        return cls(folder, meta, index, lookup)

    @property
    def n_tiles(self) -> int:
        return int(self.meta["n_tiles"])

    def check_model(self, model_id: str) -> None:
        """Query and index vectors are only comparable when produced by the same model (ERD check 14)."""
        if model_id != self.meta["model_id"]:
            raise ValueError(f"query model {model_id!r} != index model {self.meta['model_id']!r}")


def aggregate(scores: np.ndarray, rows: np.ndarray, lookup: pd.DataFrame,
              query_ids: list[str] | None = None) -> pd.DataFrame:
    """FAISS output (P x k scores and row ids) -> image ranking.

    Returns columns: rank, image_id, score, best_tile_idx, best_query_id.
    """
    P, k = scores.shape
    query_ids = query_ids or [str(i) for i in range(P)]
    flat = pd.DataFrame({"row_id": rows.ravel(), "score": scores.ravel().astype(np.float64),
                         "best_query_id": np.repeat(query_ids, k)})
    flat = flat[flat["row_id"] >= 0]                                  # FAISS pads with -1
    flat = flat.join(lookup.set_index("row_id")[["image_id", "tile_idx"]], on="row_id")
    # max over tiles and prompts: keep the best row per image (ties -> lower tile, then first prompt)
    flat = flat.sort_values(["image_id", "score", "tile_idx"], ascending=[True, False, True], kind="mergesort")
    best = flat.drop_duplicates("image_id", keep="first")
    best = best.sort_values(["score", "image_id"], ascending=[False, True], kind="mergesort")
    out = best.rename(columns={"tile_idx": "best_tile_idx"})[["image_id", "score", "best_tile_idx", "best_query_id"]]
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.reset_index(drop=True)


def search_vectors(idx: Index, qv: np.ndarray, top_n: int | None = None, k_vectors: int | None = None,
                   query_ids: list[str] | None = None) -> pd.DataFrame:
    """Search with already-encoded query vectors (P x dim). top_n=None -> full ranking."""
    qv = np.ascontiguousarray(np.atleast_2d(qv), dtype=np.float32)
    if k_vectors is None:
        k_vectors = idx.index.ntotal if top_n is None else min(idx.index.ntotal, top_n * idx.n_tiles)
    scores, rows = idx.index.search(qv, k_vectors)
    ranking = aggregate(scores, rows, idx.lookup, query_ids)
    return ranking if top_n is None else ranking.head(top_n)


def search_text(idx: Index, clip, prompts: list[str], top_n: int | None = 50,
                query_ids: list[str] | None = None, k_vectors: int | None = None) -> pd.DataFrame:
    idx.check_model(clip.model_id)
    return search_vectors(idx, clip.encode_text(prompts), top_n, k_vectors, query_ids)
