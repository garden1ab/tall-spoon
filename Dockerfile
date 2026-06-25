FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/workspace/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace
ARG KREA2_REF=main
RUN git clone --depth=1 --branch ${KREA2_REF} https://github.com/krea-ai/krea-2.git /workspace/krea-2

COPY requirements-app.txt /workspace/requirements-app.txt

WORKDIR /workspace/krea-2
RUN uv sync && \
    uv pip install --python /workspace/krea-2/.venv/bin/python -r /workspace/requirements-app.txt && \
    /workspace/krea-2/.venv/bin/python - <<'PYVERIFY'
import gradio, huggingface_hub, psutil, accelerate
print("Krea-2 local UI app dependencies installed at build time.")
PYVERIFY

COPY app/ /workspace/app/
COPY scripts/ /workspace/scripts/
COPY scripts/start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh /workspace/scripts/*.sh

EXPOSE 7860
WORKDIR /workspace
CMD ["/workspace/start.sh"]
