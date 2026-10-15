FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    N3_PATHS=/app/deploy/paths.container.yaml N3_IMAGE_BACKEND=torch_cpu \
    HF_HOME=/tmp/huggingface HF_HUB_CACHE=/assets/tokenizer-cache/hub \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY requirements.txt ./
# Install CPU wheels first; the matching requirements preserve these wheels.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0 \
    && python -m pip install -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/
COPY configs/clip_models.yaml configs/tiling.yaml ./configs/
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY tools/check_assets.py ./tools/check_assets.py
COPY deploy/entrypoint.py deploy/paths.container.yaml ./deploy/

USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
ENTRYPOINT ["python", "deploy/entrypoint.py"]
