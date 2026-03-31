FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        git \
        libgl1 \
        libglib2.0-0 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA 12.8 — 별도 레이어로 분리해 캐싱 효율 극대화
RUN uv pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# 나머지 의존성 설치 (torch 계열은 위에서 이미 설치)
COPY requirements.txt .
# `torchsde`처럼 `torch`로 시작하지만 다른 패키지는 제외하지 않도록
RUN grep -vE '^(torch($|[<>=!~])|torchvision($|[<>=!~])|torchaudio($|[<>=!~]))' requirements.txt | uv pip install -r /dev/stdin

# 소스 코드 복사
COPY . .

EXPOSE 7860

CMD ["python3", "main.py"]
