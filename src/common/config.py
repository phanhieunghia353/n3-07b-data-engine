"""Load repo configs and resolve machine-local paths.

All paths come from configs/paths.yaml (not committed; copy paths.example.yaml).
Set env var N3_PATHS to point at another paths file.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DECISIONS_PATH = CONFIG_DIR / "decisions.yaml"
SPLITS_DIR = REPO_ROOT / "splits"


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_decisions(path: Path | None = None) -> dict:
    return load_yaml(Path(path) if path else DECISIONS_PATH)


def file_sha(path: Path) -> str:
    """sha256 of a text file with line endings normalised to LF (same on Windows and Colab)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def decisions_sha(path: Path | None = None) -> str:
    return file_sha(Path(path) if path else DECISIONS_PATH)


def list_sha(ids) -> str:
    """sha256 of image ids sorted ascending, joined by '\\n' (ERD convention)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def code_commit() -> str:
    """Current git HEAD, with '+dirty' if the work tree has uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass(frozen=True)
class Paths:
    data_root: Path
    images_dir: Path          # contains train/ and val/
    labels_json: dict          # {"train": Path, "val": Path}
    yolo_labels_dir: Path      # contains train/ and val/
    meta_dir: Path
    index_root: Path
    preds_dir: Path
    uncertainty_file: Path | None = None
    clip_weights_dir: Path | None = None   # optional local CLIP weights (HF download fallback)

    def image_path(self, bdd_split: str, image_id: str) -> Path:
        return self.images_dir / bdd_split / f"{image_id}.jpg"

    def yolo_label_path(self, bdd_split: str, image_id: str) -> Path:
        return self.yolo_labels_dir / bdd_split / f"{image_id}.txt"


def _resolve(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else base / q


def load_paths(path: Path | None = None) -> Paths:
    path = Path(path or os.environ.get("N3_PATHS") or CONFIG_DIR / "paths.yaml")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy configs/paths.example.yaml to configs/paths.yaml and edit it.")
    cfg = load_yaml(path)
    data_root = _resolve(REPO_ROOT, cfg["data_root"])
    return Paths(
        data_root=data_root,
        images_dir=_resolve(data_root, cfg.get("images_dir", "images/100k")),
        labels_json={k: _resolve(data_root, v) for k, v in cfg.get("labels_json", {}).items()},
        yolo_labels_dir=_resolve(data_root, cfg.get("yolo_labels_dir", "labels/100k")),
        meta_dir=_resolve(REPO_ROOT, cfg["meta_dir"]),
        index_root=_resolve(REPO_ROOT, cfg["index_root"]),
        preds_dir=_resolve(REPO_ROOT, cfg.get("preds_dir", "data/preds")),
        uncertainty_file=_resolve(REPO_ROOT, cfg["uncertainty_file"]) if cfg.get("uncertainty_file") else None,
        clip_weights_dir=_resolve(REPO_ROOT, cfg["clip_weights_dir"]) if cfg.get("clip_weights_dir") else None,
    )
