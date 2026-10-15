"""Validate mounted assets before starting the public CPU application."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_paths
from tools.check_assets import check


def main():
    try:
        paths = load_paths()
        issues = check(paths)
        if not issues:
            from src.search.encode import MODELS, local_weights, load_tokenizer
            models = {json.loads(meta.read_text(encoding='utf-8'))['model_id']
                      for meta in paths.index_root.glob('*__*/meta.json')}
            for model_id in models:
                cfg = MODELS[model_id]
                if local_weights(cfg, paths.clip_weights_dir) is None:
                    issues.append(f'Missing local weights for {model_id}; run tools/prepare_assets.py first')
                load_tokenizer(cfg['open_clip_name'])
    except (OSError, KeyError, ValueError) as exc:
        issues = [str(exc)]
    if issues:
        for issue in issues:
            print(f'Asset setup: {issue}', file=sys.stderr)
        return 1
    # Native configuration remains localhost-only; bind externally inside this container.
    os.execv(sys.executable, [sys.executable, '-m', 'streamlit', 'run', 'app/app.py',
             '--server.address=0.0.0.0', '--server.port=8501', '--server.fileWatcherType=none',
             '--server.maxUploadSize=15', '--server.enableXsrfProtection=true'])


if __name__ == '__main__':
    raise SystemExit(main())
