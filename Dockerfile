# VUI-FastAPI TTS • CUDA 12.1 • Python 3.12.3
FROM python:3.12.3-slim
ARG HUGGING_FACE_HUB_TOKEN

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/cache/huggingface \
    HF_HUB_CACHE=/opt/cache/huggingface

# Install system packages INCLUDING C compiler for torch.compile
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        build-essential \
        gcc \
        g++ && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache-dir -r /tmp/requirements.txt

# Clone and install VUI
RUN git clone --depth 1 https://github.com/fluxions-ai/vui.git /opt/vui
RUN uv pip install --system --no-cache-dir -e /opt/vui

# Pre-download VUI model weights
RUN --mount=type=cache,target=${HF_HOME} \
    python -c "from vui.model import Vui; print('Downloading VUI model...'); Vui.from_pretrained(); print('VUI model downloaded.')"

# Pre-download PyAnnote GATED models (VAD)
COPY download_pyannote.py /tmp/download_pyannote.py
RUN --mount=type=cache,target=${HF_HOME} \
    HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} \
    python /tmp/download_pyannote.py && \
    rm /tmp/download_pyannote.py

# Setup application
WORKDIR /app
COPY main.py /app/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]