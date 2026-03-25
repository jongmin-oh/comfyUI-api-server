<div align="center">

# ComfyUI API Server

**The fastest way to serve Stable Diffusion as a production API.**

ComfyUI's speed and optimization. SD WebUI's familiar REST interface. Zero compromises.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-0.17.0-blueviolet)](https://github.com/comfyanonymous/ComfyUI)
[![Docker](https://img.shields.io/badge/Docker-CUDA%2012.4-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-GPL--3.0-green)](LICENSE)

[Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Docker Setup](#-docker-setup) · [Examples](#-examples)

</div>

---

## Why This Exists

| | **SD WebUI** | **ComfyUI** | **This Project** |
|---|---|---|---|
| REST API (`/sdapi/v1`) | ✅ | ❌ | ✅ |
| Generation Speed | Slow | **Fast** | **Fast** |
| Latest Samplers & Optimizations | ❌ | ✅ | ✅ |
| LoRA Chaining | Limited | ✅ | ✅ |
| No Disk I/O for API calls | ❌ | ❌ | ✅ |
| Headless / Server-optimized | ❌ | ❌ | ✅ |
| Docker-ready | △ | △ | ✅ |

**ComfyUI is fast and powerful, but has no REST API. SD WebUI has a great API, but is slow and heavy.**

This project is a minimal FastAPI server built on top of ComfyUI's execution engine that exposes the `sdapi/v1` interface — so every tool, script, and app built for SD WebUI just works, while running at ComfyUI speed.

All UI-related code has been removed. This is a pure server.

---

## Features

- **SDAPI compatible** — Drop-in replacement for SD WebUI's `/sdapi/v1/txt2img` and `/sdapi/v1/img2img` endpoints
- **Pure in-memory I/O** — Images are passed as base64 strings; nothing is written to disk during inference
- **LoRA chaining** — Apply multiple LoRAs in a single request with independent `strength_model` and `strength_clip` controls
- **17+ samplers** — Euler, DPM++ 2M Karras, DDIM, UniPC, Heun, and more — all mapped from SD WebUI names
- **CLIP skip** — Full clip layer depth control
- **img2img with inpainting** — Mask-based inpainting with denoising strength control
- **LRU model caching** — Keep frequently used models in VRAM between requests
- **Docker + NVIDIA GPU** — One command to build and run

---

## Quick Start

### Option 1 — Docker (Recommended)

```bash
git clone https://github.com/jongmin-oh/comfyUI-api-server
cd comfyui-api-server

# Place your model checkpoints and LoRAs
mkdir -p models/checkpoints models/loras
cp your_model.safetensors models/checkpoints/

# Build and run
bash refresh.sh
```

The server starts on `http://localhost:7860`.

### Option 2 — Local Python

```bash
git clone https://github.com/jongmin-oh/comfyUI-api-server
cd comfyui-api-server

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py --gpu-only --cache-lru 50
```

---

## API Reference

The server exposes a subset of the SD WebUI SDAPI spec. Any client library that targets SD WebUI will work out of the box.

### `POST /sdapi/v1/txt2img`

Generate an image from a text prompt.

```json
{
  "prompt": "a photorealistic portrait of a woman, 8k, dramatic lighting",
  "negative_prompt": "blurry, low quality, watermark",
  "width": 1024,
  "height": 1024,
  "steps": 20,
  "cfg_scale": 7.0,
  "sampler_name": "DPM++ 2M Karras",
  "seed": -1,
  "batch_size": 1,
  "override_settings": {
    "sd_model_checkpoint": "your_model.safetensors"
  },
  "loras": [
    { "name": "your_lora.safetensors", "strength_model": 0.8, "strength_clip": 0.8 }
  ]
}
```

**Response:**
```json
{
  "images": ["<base64-encoded PNG>"],
  "parameters": { ... },
  "info": "..."
}
```

---

### `POST /sdapi/v1/img2img`

Modify an existing image using a prompt.

```json
{
  "init_images": ["<base64-encoded input image>"],
  "prompt": "same person, cyberpunk style, neon lighting",
  "negative_prompt": "blurry, low quality",
  "denoising_strength": 0.6,
  "width": 1024,
  "height": 1024,
  "steps": 20,
  "sampler_name": "Euler",
  "override_settings": {
    "sd_model_checkpoint": "your_model.safetensors"
  }
}
```

---

### `GET /sdapi/v1/sd-models`

List all available checkpoint models.

```json
[
  {
    "title": "your_model.safetensors",
    "model_name": "your_model",
    "filename": "/app/models/checkpoints/your_model.safetensors",
    "hash": "abc123",
    "sha256": "..."
  }
]
```

---

### `GET /sdapi/v1/samplers`

List all available samplers with their aliases.

```json
[
  { "name": "Euler", "aliases": ["euler"], "options": {} },
  { "name": "DPM++ 2M Karras", "aliases": ["dpmpp_2m_karras"], "options": {} },
  ...
]
```

---

### Supported Samplers

| SDAPI Name | ComfyUI Sampler | Scheduler |
|---|---|---|
| Euler | euler | normal |
| Euler a | euler_ancestral | normal |
| Heun | heun | normal |
| DPM2 | dpm_2 | normal |
| DPM2 a | dpm_2_ancestral | normal |
| DPM++ 2S a | dpmpp_2s_ancestral | normal |
| DPM++ 2M | dpmpp_2m | normal |
| DPM++ SDE | dpmpp_sde | normal |
| DPM++ 2M Karras | dpmpp_2m | karras |
| DPM++ 2M SDE Karras | dpmpp_2m_sde | karras |
| DPM++ 3M SDE Karras | dpmpp_3m_sde | karras |
| DDIM | ddim | normal |
| UniPC | uni_pc | normal |
| LMS | lms | normal |

---

## Docker Setup

### Dockerfile Overview

```
Base:    nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
Python:  3.11
Port:    7860
Command: python3 main.py --gpu-only --cache-lru 50
```

### `refresh.sh`

```bash
bash refresh.sh
```

This script:
1. Builds the Docker image tagged `comfyui-api`
2. Stops and removes any existing container
3. Runs a new container with:
   - NVIDIA GPU access (`--runtime=nvidia`)
   - Shared memory for multi-process PyTorch (`--ipc=host`)
   - Auto-restart on failure (`--restart=always`)
   - Local `models/` directory mounted into the container

### Manual Docker Run

```bash
docker build -t comfyui-api .

docker run -d \
  --name comfyui-api \
  --runtime=nvidia \
  --ipc=host \
  --restart=always \
  -p 7860:7860 \
  -v $(pwd)/models:/app/models \
  comfyui-api
```

---

## Model Setup

Place model files in the `models/` directory before starting the server:

```
models/
├── checkpoints/       # Main diffusion models (.safetensors, .ckpt)
│   └── your_model.safetensors
└── loras/             # LoRA models
    └── your_lora.safetensors
```

The server automatically detects all files in these directories at startup. No restart needed when using the `override_settings.sd_model_checkpoint` parameter to switch models per-request.

---

## Examples

### Python — txt2img

```python
import requests, base64, json
from PIL import Image
from io import BytesIO

response = requests.post("http://localhost:7860/sdapi/v1/txt2img", json={
    "prompt": "a beautiful landscape, golden hour, 8k",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "cfg_scale": 7.0,
    "sampler_name": "DPM++ 2M Karras",
    "seed": 42,
    "override_settings": {
        "sd_model_checkpoint": "your_model.safetensors"
    }
})

result = response.json()
image = Image.open(BytesIO(base64.b64decode(result["images"][0])))
image.save("output.png")
```

### Python — img2img

```python
import requests, base64, json
from PIL import Image
from io import BytesIO

# Load and encode the input image
with open("input.png", "rb") as f:
    input_b64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post("http://localhost:7860/sdapi/v1/img2img", json={
    "init_images": [input_b64],
    "prompt": "same scene, winter, snow",
    "denoising_strength": 0.6,
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "sampler_name": "Euler",
    "override_settings": {
        "sd_model_checkpoint": "your_model.safetensors"
    }
})

result = response.json()
image = Image.open(BytesIO(base64.b64decode(result["images"][0])))
image.save("output.png")
```

### cURL — txt2img

```bash
curl -s -X POST http://localhost:7860/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat sitting on a windowsill",
    "steps": 20,
    "width": 512,
    "height": 512,
    "sampler_name": "Euler",
    "override_settings": {"sd_model_checkpoint": "your_model.safetensors"}
  }' | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
open('output.png', 'wb').write(base64.b64decode(data['images'][0]))
print('Saved output.png')
"
```

---

## Architecture

```
HTTP Request
    │
    ▼
FastAPI Routes (/sdapi/routes.py)
    │  Parse request → Pydantic model validation
    │
    ▼
Workflow Builder (/sdapi/workflow_builder.py)
    │  Convert SDAPI params → ComfyUI node graph
    │  CheckpointLoader → LoRALoader(s) → CLIPEncode → KSampler → VAEDecode
    │
    ▼
SDAPI Executor (/sdapi/executor.py)
    │  Submit workflow to PromptQueue
    │  Async wait for completion (timeout: 300s)
    │
    ▼
ComfyUI Execution Engine (execution.py)
    │  Node-by-node graph execution
    │  GPU inference via PyTorch
    │  LRU model caching
    │
    ▼
MemoryImage Node (nodes.py)
    │  Tensor → PIL → BytesIO → base64
    │  No disk write
    │
    ▼
Serializer (/sdapi/serializer.py)
    │  History entry → SDAPI response format
    │
    ▼
HTTP Response { "images": ["<base64>"], ... }
```

---

## Configuration

### CLI Arguments

```bash
python main.py [options]

  --gpu-only              Use GPU only (no CPU fallback)
  --cache-lru N           LRU cache size for models in VRAM (default: 0)
  --cache-none            Disable model caching entirely
  --cache-classic         Use classic caching strategy
  --cache-ram-pressure    Evict models under RAM pressure
  --listen HOST           Bind address (default: 127.0.0.1)
  --port PORT             Port (default: 7860)
  --deterministic         Enable deterministic CUDA operations
```

### Extra Model Paths

To load models from external directories, create `extra_model_paths.yaml`:

```yaml
a1111:
  base_path: /path/to/stable-diffusion-webui/
  checkpoints: models/Stable-diffusion
  loras: models/Lora
  vae: models/VAE
```

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.4 support
- 8GB+ VRAM (16GB+ recommended for SDXL)
- Docker + NVIDIA Container Toolkit (for Docker deployment)

---

## Acknowledgements

This project is built on top of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by [@comfyanonymous](https://github.com/comfyanonymous).
The SDAPI interface is modeled after [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

---

## License

GPL-3.0. See [LICENSE](LICENSE).
