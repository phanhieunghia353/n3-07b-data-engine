"""Validate a local asset pack without downloading models or changing any data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_paths  # noqa: E402


def check(paths, all_images: bool = False) -> list[str]:
    import pandas as pd
    from src.search.search import Index

    issues = []
    metadata = paths.meta_dir / 'images.parquet'
    if not metadata.exists():
        return [f'Missing metadata: {metadata}']
    images = pd.read_parquet(metadata)
    required = {'image_id', 'image_path', 'weather', 'scene', 'timeofday'}
    if missing := required - set(images.columns):
        return [f'Metadata columns missing: {sorted(missing)}']
    if images.empty or images.image_id.duplicated().any() or images[list(required)].isna().any().any():
        issues.append('Metadata must be non-empty, with unique image_id and no missing required values')
    sample = images if all_images else images.head(5)
    missing_images = [name for name in sample.image_path if not (paths.data_root / name).is_file()]
    if missing_images:
        issues.append(f'{len(missing_images)} missing image(s) in checked rows; first: {missing_images[0]}')
    folders = sorted(paths.index_root.glob('*__*'))
    checked = 0
    image_ids = set(images.image_id)
    for folder in folders:
        if not folder.is_dir() or '_dbg' in folder.name:
            continue
        if not all((folder / name).is_file() for name in ['index.faiss', 'lookup.parquet', 'meta.json']):
            issues.append(f'Incomplete index directory: {folder.name}')
            continue
        try:
            index = Index.load(folder)
            expected = {'model_id', 'backend', 'n_tiles', 'n_vectors', 'open_clip_name', 'pretrained', 'preprocess'}
            if absent := expected - set(index.meta):
                issues.append(f'{folder.name}: missing meta fields {sorted(absent)}')
            elif index.meta['model_id'] != folder.name.split('__', 1)[0]:
                issues.append(f'{folder.name}: model_id does not match directory name')
            if not set(index.lookup.image_id) <= image_ids:
                issues.append(f'{folder.name}: lookup contains images absent from metadata')
            checked += 1
            del index
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            issues.append(f'{folder.name}: {exc}')
    if not checked:
        issues.append(f'No usable index found in {paths.index_root}')
    uncertainty = getattr(paths, 'uncertainty_file', None)
    if uncertainty:
        if not uncertainty.is_file():
            issues.append(f'Optional uncertainty file is configured but missing: {uncertainty}')
        elif not {'image_id', 'uncertainty'} <= set(pd.read_parquet(uncertainty).columns):
            issues.append('uncertainty_file needs image_id and uncertainty columns')
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-images', action='store_true', help='Check every image path instead of the first 5')
    args = parser.parse_args()
    try:
        issues = check(load_paths(), args.all_images)
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        issues = [str(exc)]
    if issues:
        print('Asset setup is not ready:')
        for issue in issues:
            print(f'- {issue}')
        return 1
    print('Local metadata, image paths and FAISS indices are ready.')
    print('Model/tokenizer cache is checked when the encoder first loads; no model was downloaded here.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
