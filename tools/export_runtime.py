"""Export only explicitly approved runtime files. Never copy the working tree wholesale."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOTS = {'data', 'INDEX_ROOT', 'index_root', 'notebooks', 'report', 'results', 'splits',
                 'private', 'outputs', '.git', '.venv'}
PRIVATE_SUFFIXES = {'.docx', '.pdf', '.ipynb', '.parquet', '.faiss', '.npy', '.pt', '.safetensors'}


def approved_files(root: Path = REPO_ROOT) -> list[str]:
    """Fail closed if the manifest includes local configuration, assets, or unsafe paths."""
    files = [line.strip() for line in (root / 'public-files.txt').read_text(encoding='utf-8').splitlines()
             if line.strip() and not line.startswith('#')]
    if len(files) != len(set(files)):
        raise ValueError('Duplicate entries in public-files.txt')
    for name in files:
        path = PurePosixPath(name)
        if ('\\' in name or ':' in name or path.is_absolute() or '..' in path.parts
                or path.parts[0] in PRIVATE_ROOTS or path.suffix.lower() in PRIVATE_SUFFIXES
                or path.name.startswith('.env') or name == 'configs/paths.yaml'):
            raise ValueError(f'Private or unsafe export path: {name}')
        source = root / name
        if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Not a regular file inside the runtime repository: {name}')
    return files


def export(destination: Path, root: Path = REPO_ROOT) -> dict:
    """Create a new export; an existing directory is never overwritten or deleted."""
    files = approved_files(root)
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(root.resolve()):
        raise ValueError('Choose a new destination outside the source repository')
    destination.mkdir(parents=True)
    total = 0
    for name in files:
        source, target = root / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
            raise OSError(f'Export verification failed: {name}')
        total += target.stat().st_size
    return {'files': len(files), 'bytes': total, 'destination': str(destination)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.destination), ensure_ascii=False))
