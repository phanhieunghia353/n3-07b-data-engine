"""Run a temporary HTTPS demo from this PC and keep the PC awake while it runs."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.demo-online'
URL = re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com')


def save(**values):
    temporary = STATE / f'.state-{os.getpid()}.tmp'
    temporary.write_text(json.dumps(values, indent=2), encoding='utf-8')
    temporary.replace(STATE / 'state.json')


def status():
    path = STATE / 'state.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status': 'not started'}


def healthy(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=2) as reply:
            return reply.status == 200
    except OSError:
        return False


def alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == 'nt':
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def run(args):
    app = tunnel = None
    handles = []
    awake = False
    try:
        if os.name == 'nt':
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        env = os.environ.copy()
        env.update(N3_PATHS=str(args.paths.resolve()), N3_IMAGE_BACKEND='torch_cpu',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        import yaml
        cfg = yaml.safe_load(args.paths.read_text(encoding='utf-8'))
        weights = Path(cfg.get('clip_weights_dir') or '')
        cache = weights.parent / 'tokenizer-cache/hub'
        if weights.is_absolute() and cache.is_dir():
            env.update(HF_HUB_CACHE=str(cache), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        app_log = (STATE / 'app.log').open('w', encoding='utf-8')
        handles.append(app_log)
        app = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app/app.py'),
                                '--server.address=127.0.0.1', f'--server.port={args.port}',
                                '--server.headless=true', '--server.fileWatcherType=none',
                                '--server.maxUploadSize=15'], cwd=ROOT, env=env,
                               stdout=app_log, stderr=subprocess.STDOUT, creationflags=flags)
        base = {'worker_pid': os.getpid(), 'app_pid': app.pid, 'port': args.port,
                'keep_awake': awake, 'url': None}
        save(status='starting', **base)
        deadline = time.monotonic() + 120
        while not healthy(args.port):
            if app.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Streamlit did not start; see .demo-online/app.log')
            if (STATE / 'stop').exists():
                return
            time.sleep(0.5)
        save(status='preview', **base)
        while not (STATE / 'publish').exists():
            if (STATE / 'stop').exists():
                return
            if app.poll() is not None:
                raise RuntimeError('Local app stopped')
            time.sleep(0.5)
        tunnel_log = (STATE / 'tunnel.log').open('w', encoding='utf-8')
        handles.append(tunnel_log)
        tunnel = subprocess.Popen([str(args.cloudflared.resolve()), 'tunnel', '--no-autoupdate',
                                   '--protocol', 'http2', '--url', f'http://127.0.0.1:{args.port}'],
                                  cwd=ROOT, stdout=tunnel_log, stderr=subprocess.STDOUT,
                                  creationflags=flags)
        base['tunnel_pid'] = tunnel.pid
        save(status='connecting', **base)
        while not (STATE / 'stop').exists():
            if app.poll() is not None or tunnel.poll() is not None:
                raise RuntimeError('Demo process stopped; check local logs and start again')
            if base['url'] is None:
                log = (STATE / 'tunnel.log').read_text(encoding='utf-8', errors='replace')
                if match := URL.search(log):
                    base['url'] = match.group()
                    (STATE / 'link.txt').write_text(base['url'] + '\n', encoding='utf-8')
                    save(status='online', **base)
            time.sleep(1)
    except Exception as exc:
        save(status='failed', worker_pid=os.getpid(), error=str(exc), url=None)
        raise
    finally:
        for process in [tunnel, app]:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        for handle in handles:
            handle.close()
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if (STATE / 'stop').exists():
            save(status='stopped', url=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'run', 'publish', 'stop', 'status'])
    parser.add_argument('--paths', type=Path, default=(STATE / 'paths.yaml' if (STATE / 'paths.yaml').exists()
                                                    else ROOT / 'configs/paths.yaml'))
    parser.add_argument('--cloudflared', type=Path, default=STATE / 'cloudflared.exe')
    parser.add_argument('--port', type=int, default=8510)
    parser.add_argument('--preview', action='store_true', help='Check locally before publishing')
    args = parser.parse_args()
    STATE.mkdir(exist_ok=True)
    if args.action in {'publish', 'stop'}:
        (STATE / args.action).write_text('requested\n', encoding='utf-8')
        if args.action == 'stop':
            deadline = time.monotonic() + 15
            while status().get('status') != 'stopped' and time.monotonic() < deadline:
                if not alive(status().get('worker_pid')):
                    break
                time.sleep(0.2)
    elif args.action == 'start':
        old = status()
        if alive(old.get('worker_pid')) and old.get('status') in {'starting', 'preview', 'connecting', 'online'}:
            print(json.dumps(old, ensure_ascii=False))
            return
        if not args.paths.is_file() or not args.cloudflared.is_file():
            raise SystemExit('Prepare configs/paths.yaml and .demo-online/cloudflared.exe first')
        if healthy(args.port):
            raise SystemExit(f'Port {args.port} is already in use; choose another port')
        for name in ['stop', 'publish']:
            (STATE / name).unlink(missing_ok=True)
        if not args.preview:
            (STATE / 'publish').write_text('requested\n', encoding='utf-8')
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        with (STATE / 'worker.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'run',
                                        '--paths', str(args.paths.resolve()), '--cloudflared',
                                        str(args.cloudflared.resolve()), '--port', str(args.port)],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        save(status='starting', worker_pid=process.pid, url=None)
        print(f'Demo worker started: {process.pid}. Read .demo-online/link.txt when connected.')
    elif args.action == 'run':
        run(args)
        return
    print(json.dumps(status(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
