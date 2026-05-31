FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/opt/cache/huggingface \
    APP_HOME=/opt/app

WORKDIR ${APP_HOME}

# Install essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt ${APP_HOME}/requirements.txt
RUN pip install --upgrade pip && \
    # Install CPU-specific PyTorch (much smaller, highly optimized for x86)
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r ${APP_HOME}/requirements.txt

COPY app ${APP_HOME}/app

RUN mkdir -p /opt/cache/arxiv /opt/cache/huggingface && \
    python ${APP_HOME}/app/prefetch.py \
      --model-id microsoft/harrier-oss-v1-0.6b \
      --cache-dir /opt/cache

WORKDIR ${APP_HOME}/app

CMD ["python", "speedtest.py", "--device", "cpu", "--dtype", "float32"]