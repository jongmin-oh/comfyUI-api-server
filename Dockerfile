FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        libgl1 \
        libglib2.0-0 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA 12.4 — 별도 레이어로 분리해 캐싱 효율 극대화
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

# 나머지 의존성 설치 (torch 계열은 위에서 이미 설치)
COPY requirements.txt .
RUN grep -vE "^torch(vision|audio)?" requirements.txt | pip install -r /dev/stdin

# 소스 코드 복사
COPY . .

EXPOSE 7860

CMD ["python3", "main.py", "--gpu-only", "--cache-lru", "50"]
