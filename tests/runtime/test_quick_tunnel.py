"""Verify the local demo owns its children and stops them without touching other processes."""
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def loader(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / 'deploy/quick_tunnel.py'
    spec = spec_from_file_location('quick_demo', source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'STATE', tmp_path / '.demo-online')
    module.STATE.mkdir()
    powers = []
    monkeypatch.setattr(module.ctypes, 'windll', SimpleNamespace(kernel32=SimpleNamespace(
        SetThreadExecutionState=lambda flags: powers.append(flags) or 1)), raising=False)
    paths = tmp_path / 'paths.yaml'
    paths.write_text('{}', encoding='utf-8')
    return module, SimpleNamespace(paths=paths, port=8510, cloudflared=tmp_path / 'cloudflared.exe'), powers


def test_current_process_can_be_checked_without_signalling_it(tmp_path, monkeypatch):
    module, _, _ = loader(tmp_path, monkeypatch)
    assert module.alive(os.getpid())
    assert not module.alive(None)


def test_stop_releases_only_demo_children(tmp_path, monkeypatch):
    module, args, powers = loader(tmp_path, monkeypatch)
    (module.STATE / 'publish').touch()
    children = []
    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(children)
            self.command = command
            self.stopped = False
            children.append(self)
            if len(children) == 2:
                kwargs['stdout'].write('https://example-demo.trycloudflare.com\n')
                kwargs['stdout'].flush()
        def poll(self): return 0 if self.stopped else None
        def terminate(self): self.stopped = True
        def wait(self, timeout): return 0
    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    monkeypatch.setattr(module, 'healthy', lambda port: True)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: (module.STATE / 'stop').touch())
    module.run(args)
    assert len(children) == 2 and all(child.stopped for child in children)
    assert '--server.address=127.0.0.1' in children[0].command
    assert children[1].command[-1] == 'http://127.0.0.1:8510'
    assert module.status()['status'] == 'stopped'
    assert (module.STATE / 'link.txt').read_text().strip() == 'https://example-demo.trycloudflare.com'
    if os.name == 'nt':
        assert powers == [0x80000001, 0x80000000]


def test_failed_local_app_never_opens_tunnel(tmp_path, monkeypatch):
    module, args, _ = loader(tmp_path, monkeypatch)
    (module.STATE / 'publish').touch()
    started = []
    class Failed:
        pid = 100
        def __init__(self, command, **kwargs): started.append(command)
        def poll(self): return 1
    monkeypatch.setattr(module.subprocess, 'Popen', Failed)
    monkeypatch.setattr(module, 'healthy', lambda port: False)
    with pytest.raises(RuntimeError, match='did not start'):
        module.run(args)
    assert len(started) == 1
    assert module.status()['status'] == 'failed'
