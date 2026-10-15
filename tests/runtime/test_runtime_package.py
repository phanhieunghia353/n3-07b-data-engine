"""Verify the public boundary and the first-run path, not private research files."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.common import config

ROOT = Path(__file__).resolve().parents[2]


def exporter():
    spec = spec_from_file_location('runtime_export', ROOT / 'tools/export_runtime.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_does_not_collect_private_neighbors(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'public-files.txt').write_text('public-files.txt\nREADME.md\n', encoding='utf-8')
    (root / 'README.md').write_text('Public runtime', encoding='utf-8')
    (root / 'report').mkdir()
    (root / 'report/paper.md').write_text('private', encoding='utf-8')
    (root / '.env').write_text('secret=private', encoding='utf-8')
    destination = tmp_path / 'export'
    result = exporter().export(destination, root)
    assert result['files'] == 2
    assert sorted(p.name for p in destination.iterdir()) == ['README.md', 'public-files.txt']
    with pytest.raises(ValueError, match='new destination'):
        exporter().export(destination, root)
    assert (root / 'report/paper.md').read_text() == 'private'


@pytest.mark.parametrize('name', ['../report.md', 'configs/paths.yaml', 'report/paper.md', '.env', 'data/x.txt'])
def test_export_rejects_private_or_unsafe_entries(tmp_path, name):
    (tmp_path / 'public-files.txt').write_text(name + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Private or unsafe'):
        exporter().approved_files(tmp_path)


def test_missing_local_config_shows_setup_instead_of_traceback(monkeypatch):
    def missing_paths():
        raise FileNotFoundError('local config missing')
    monkeypatch.setattr(config, 'load_paths', missing_paths)
    app = AppTest.from_file(str(ROOT / 'app/app.py')).run()
    assert not app.exception
    assert any('paths.example.yaml' in message.value for message in app.info)
    assert len(app.tabs) == 0
