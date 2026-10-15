"""Catch corrupt transfers and unsafe external manifests before starting the server."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.prepare_assets import verify


def test_verify_detects_corruption_after_transfer(tmp_path):
    sample = tmp_path / 'index.faiss'
    sample.write_bytes(b'complete asset')
    manifest = {'schema': 1, 'images': 1, 'bytes': sample.stat().st_size, 'files': [
        {'path': sample.name, 'bytes': sample.stat().st_size,
         'sha256': hashlib.sha256(sample.read_bytes()).hexdigest()}]}
    (tmp_path / 'asset-manifest.json').write_text(json.dumps(manifest))
    assert verify(tmp_path)['verified_files'] == 1
    sample.write_bytes(b'corrupt!')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify(tmp_path)


def test_verify_rejects_a_manifest_path_outside_pack(tmp_path):
    manifest = {'schema': 1, 'files': [{'path': '../private.txt'}]}
    (tmp_path / 'asset-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unsafe image_path'):
        verify(tmp_path)
