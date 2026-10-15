"""Build a separate deployment pack from indexed images and cached model files.

No downloads, uploads, re-encoding, or re-indexing. The destination must be new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOKENIZER_FILES = {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'added_tokens.json', 'vocab.json', 'vocab.txt', 'merges.txt', 'spiece.model',
                   'sentencepiece.bpe.model', 'config.json'}


def image_source(data_root: Path, name: str) -> Path:
    """Metadata may only refer to images inside the configured data root."""
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name):
        raise ValueError(f'Unsafe image_path: {name}')
    source = data_root / name
    if not source.resolve().is_relative_to(data_root.resolve()) or not source.is_file():
        raise ValueError(f'Missing image or image outside data_root: {name}')
    return source


def verify(destination: Path) -> dict:
    """Check every transferred file without loading models or trusting manifest paths."""
    manifest = json.loads((destination / 'asset-manifest.json').read_text(encoding='utf-8'))
    records = manifest['files']
    if manifest.get('schema') != 1 or not records or len({r['path'] for r in records}) != len(records):
        raise ValueError('Invalid or duplicate asset manifest entries')
    for record in records:
        source = image_source(destination, record['path'])
        with source.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if source.stat().st_size != record['bytes'] or actual != record['sha256']:
            raise ValueError(f"Asset checksum mismatch: {record['path']}")
    return {'verified_files': len(records), 'images': manifest['images'], 'bytes': manifest['bytes']}


def prepare(paths, destination: Path, index_ids: list[str]) -> dict:
    import pandas as pd
    import open_clip
    from huggingface_hub.constants import HF_HUB_CACHE
    from src.search.encode import MODELS, cached_weights, local_weights
    from tools.check_assets import check
    from types import SimpleNamespace

    destination = destination.resolve()
    if destination.exists() or any(destination.is_relative_to(p.resolve()) for p in
                                   [paths.data_root, paths.index_root, paths.meta_dir]):
        raise ValueError('Choose a new destination outside the source assets')
    if not index_ids or len(set(index_ids)) != len(index_ids):
        raise ValueError('Choose at least one distinct index')
    jobs: dict[str, Path] = {}
    indexed_ids: set[str] = set()
    model_ids = set()
    for index_id in index_ids:
        if not re.fullmatch(r'[a-z0-9]+__[a-z0-9]+', index_id):
            raise ValueError(f'Unsafe index ID: {index_id}')
        folder = paths.index_root / index_id
        meta = json.loads((folder / 'meta.json').read_text(encoding='utf-8'))
        lookup = pd.read_parquet(folder / 'lookup.parquet')
        if len(lookup) != meta['n_vectors'] or meta['model_id'] != index_id.split('__', 1)[0]:
            raise ValueError(f'Inconsistent index: {index_id}')
        indexed_ids.update(lookup.image_id)
        model_ids.add(meta['model_id'])
        for name in ['index.faiss', 'lookup.parquet', 'meta.json']:
            jobs[f'indexes/{index_id}/{name}'] = folder / name
    images = pd.read_parquet(paths.meta_dir / 'images.parquet')
    images = images[images.image_id.isin(indexed_ids)].copy()
    if images.image_id.duplicated().any() or set(images.image_id) != indexed_ids:
        raise ValueError('Every indexed image must have exactly one metadata row')
    for name in images.image_path:
        jobs[f'data/{name}'] = image_source(paths.data_root, name)
    cache = Path(HF_HUB_CACHE)
    for model_id in sorted(model_ids):
        cfg = MODELS[model_id]
        pretrained = open_clip.get_pretrained_cfg(cfg['open_clip_name'], cfg['pretrained'])
        weights = local_weights(cfg, paths.clip_weights_dir) or cached_weights(pretrained)
        if weights is None:
            raise ValueError(f'Model {model_id} is not cached; prepare it locally first')
        if weights.suffix != '.safetensors':
            raise ValueError(f'Model {model_id} needs cached safetensors weights for this pack')
        jobs[f"weights/{cfg['open_clip_name']}__{cfg['pretrained']}.safetensors"] = weights
        # Explicit weight loading leaves only tokenizer files to resolve through the HF cache.
        tokenizer_repo = (open_clip.get_model_config(cfg['open_clip_name']) or {}).get(
            'text_cfg', {}).get('hf_tokenizer_name')
        if tokenizer_repo:
            repo = cache / ('models--' + tokenizer_repo.replace('/', '--'))
            files = [p for p in repo.rglob('*') if p.is_file() and
                     (p.name in TOKENIZER_FILES or 'refs' in p.relative_to(repo).parts)]
            if not any(p.name == 'tokenizer_config.json' for p in files):
                raise ValueError(f'Tokenizer {tokenizer_repo} is not cached')
            for source in files:
                jobs[f'tokenizer-cache/hub/{repo.name}/{source.relative_to(repo).as_posix()}'] = source
    if missing := [str(source) for source in jobs.values() if not source.is_file()]:
        raise ValueError(f'Missing source: {missing[0]}')
    destination.mkdir(parents=True)
    records = []
    for number, (relative, source) in enumerate(sorted(jobs.items()), 1):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        with source.open('rb') as stream:
            original = hashlib.file_digest(stream, 'sha256').hexdigest()
        with target.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != original:
            raise OSError(f'Copy verification failed: {relative}')
        records.append({'path': relative, 'bytes': target.stat().st_size, 'sha256': actual})
        if number % 5000 == 0:
            print(f'Copied and verified {number}/{len(jobs)} files', flush=True)
    (destination / 'meta').mkdir()
    metadata = destination / 'meta/images.parquet'
    images.to_parquet(metadata, index=False)
    with metadata.open('rb') as stream:
        records.append({'path': 'meta/images.parquet', 'bytes': metadata.stat().st_size,
                        'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()})
    issues = check(SimpleNamespace(data_root=destination / 'data', meta_dir=destination / 'meta',
                                   index_root=destination / 'indexes', uncertainty_file=None), all_images=True)
    if issues:
        raise ValueError('; '.join(issues))
    manifest = {'schema': 1, 'images': len(images), 'indexes': index_ids, 'models': sorted(model_ids),
                'bytes': sum(record['bytes'] for record in records), 'files': records}
    (destination / 'asset-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return {key: value for key, value in manifest.items() if key != 'files'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--paths', type=Path)
    parser.add_argument('--index', action='append', default=[], dest='indexes')
    parser.add_argument('--verify', action='store_true', help='Verify an existing pack without ML dependencies')
    args = parser.parse_args()
    if args.verify:
        result = verify(args.destination)
    else:
        from src.common.config import load_paths
        result = prepare(load_paths(args.paths), args.destination, args.indexes)
    print(json.dumps(result, indent=2))
