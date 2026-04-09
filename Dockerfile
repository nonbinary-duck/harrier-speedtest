FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/opt/cache/huggingface \
    APP_HOME=/opt/app

WORKDIR ${APP_HOME}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt ${APP_HOME}/requirements.txt
RUN pip install --upgrade pip && pip install -r ${APP_HOME}/requirements.txt

COPY app ${APP_HOME}/app

RUN mkdir -p /opt/cache/arxiv /opt/cache/huggingface && \
    python ${APP_HOME}/app/prefetch.py \
      --model-id microsoft/harrier-oss-v1-0.6b \
      --cache-dir /opt/cache

WORKDIR ${APP_HOME}/app

CMD ["python", "speedtest.py"]