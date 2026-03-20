import logging
import random

import folder_paths

# SDAPI sampler name -> (ComfyUI sampler_name, ComfyUI scheduler)
SDAPI_SAMPLER_MAP = {
    "Euler":                ("euler",               "normal"),
    "Euler a":              ("euler_ancestral",      "normal"),
    "Heun":                 ("heun",                "normal"),
    "DPM2":                 ("dpm_2",               "normal"),
    "DPM2 a":               ("dpm_2_ancestral",     "normal"),
    "DPM++ 2S a":           ("dpmpp_2s_ancestral",  "normal"),
    "DPM++ 2M":             ("dpmpp_2m",            "normal"),
    "DPM++ SDE":            ("dpmpp_sde",           "normal"),
    "DPM++ 2M SDE":         ("dpmpp_2m_sde",        "normal"),
    "DPM++ 2M Karras":      ("dpmpp_2m",            "karras"),
    "DPM++ SDE Karras":     ("dpmpp_sde",           "karras"),
    "DPM++ 2M SDE Karras":  ("dpmpp_2m_sde",        "karras"),
    "DPM++ 2S a Karras":    ("dpmpp_2s_ancestral",  "karras"),
    "DPM fast":             ("dpm_fast",            "normal"),
    "DPM adaptive":         ("dpm_adaptive",        "normal"),
    "LMS":                  ("lms",                 "normal"),
    "LMS Karras":           ("lms",                 "karras"),
    "DDIM":                 ("ddim",                "ddim_uniform"),
    "UniPC":                ("uni_pc",              "normal"),
}

_DEFAULT_SAMPLER = "euler"
_DEFAULT_SCHEDULER = "normal"


def _parse_sampler(sdapi_sampler_name: str, scheduler_override: str | None = None) -> tuple:
    sampler, scheduler = SDAPI_SAMPLER_MAP.get(sdapi_sampler_name, (_DEFAULT_SAMPLER, _DEFAULT_SCHEDULER))
    if scheduler_override:
        scheduler = scheduler_override
    return sampler, scheduler


def _resolve_model(model_name: str | None) -> str:
    available = folder_paths.get_filename_list("checkpoints")
    if not available:
        raise ValueError("No checkpoint models found in models/checkpoints/")
    if model_name is None:
        return available[0]
    if model_name in available:
        return model_name
    lower = model_name.lower()
    for name in available:
        if lower in name.lower():
            return name
    logging.warning("[sdapi] Model '%s' not found, falling back to '%s'", model_name, available[0])
    return available[0]


def _random_seed():
    return random.randint(0, 0xFFFFFFFFFFFFFFFF)


class _WorkflowBuilder:
    """노드 ID를 순서대로 할당하며 workflow dict를 조립하는 헬퍼."""

    def __init__(self):
        self._nodes: dict = {}
        self._counter = 0

    def add(self, class_type: str, inputs: dict) -> str:
        self._counter += 1
        node_id = str(self._counter)
        self._nodes[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    def build(self) -> dict:
        return self._nodes


def _build_common_nodes(b: _WorkflowBuilder, params: dict) -> tuple[str, str, str]:
    """
    CheckpointLoaderSimple → LoraLoader(들) → CLIPSetLastLayer(선택)
    을 조립하고 (model_node_ref, clip_node_ref, vae_node_ref) 를 반환한다.
    각 ref 는 [node_id, slot_index] 형태의 리스트.
    """
    model_name = _resolve_model((params.get("override_settings") or {}).get("sd_model_checkpoint"))

    ckpt_id = b.add("CheckpointLoaderSimple", {"ckpt_name": model_name})
    model_ref = [ckpt_id, 0]
    clip_ref  = [ckpt_id, 1]
    vae_ref   = [ckpt_id, 2]

    # LoRA 체이닝 (workflow.json 의 LoraLoader 방식 그대로)
    for lora in params.get("loras") or []:
        lora_id = b.add("LoraLoader", {
            "model": model_ref,
            "clip":  clip_ref,
            "lora_name":       lora["name"],
            "strength_model":  lora["strength_model"],
            "strength_clip":   lora["strength_clip"],
        })
        model_ref = [lora_id, 0]
        clip_ref  = [lora_id, 1]

    # CLIPSetLastLayer (clip_skip >= 2 일 때만 적용, workflow.json 의 -2 = clip_skip 2)
    clip_skip = int(params.get("clip_skip", 1))
    if clip_skip >= 2:
        clip_set_id = b.add("CLIPSetLastLayer", {
            "clip":              clip_ref,
            "stop_at_clip_layer": -clip_skip,   # -2 → 마지막 레이어에서 2번째
        })
        clip_ref = [clip_set_id, 0]

    return model_ref, clip_ref, vae_ref


def build_txt2img_workflow(params: dict) -> dict:
    b = _WorkflowBuilder()

    positive_prompt = params.get("prompt", "")
    negative_prompt = params.get("negative_prompt", "")
    width      = int(params.get("width", 512))
    height     = int(params.get("height", 512))
    steps      = int(params.get("steps", 20))
    cfg        = float(params.get("cfg_scale", 7.0))
    batch_size = int(params.get("batch_size", 1))
    seed       = int(params.get("seed", -1))
    if seed == -1:
        seed = _random_seed()

    sampler_name, scheduler = _parse_sampler(params.get("sampler_name", "Euler"), params.get("scheduler"))

    model_ref, clip_ref, vae_ref = _build_common_nodes(b, params)

    pos_id = b.add("CLIPTextEncode", {"text": positive_prompt, "clip": clip_ref})
    neg_id = b.add("CLIPTextEncode", {"text": negative_prompt, "clip": clip_ref})
    lat_id = b.add("EmptyLatentImage", {"width": width, "height": height, "batch_size": batch_size})

    ks_id = b.add("KSampler", {
        "model":        model_ref,
        "positive":     [pos_id, 0],
        "negative":     [neg_id, 0],
        "latent_image": [lat_id, 0],
        "seed":         seed,
        "steps":        steps,
        "cfg":          cfg,
        "sampler_name": sampler_name,
        "scheduler":    scheduler,
        "denoise":      1.0,
    })

    dec_id  = b.add("VAEDecode",  {"samples": [ks_id, 0], "vae": vae_ref})
    b.add("MemoryImage", {"images": [dec_id, 0]})

    return b.build()


def build_img2img_workflow(params: dict, init_image_b64: str) -> dict:
    b = _WorkflowBuilder()

    positive_prompt = params.get("prompt", "")
    negative_prompt = params.get("negative_prompt", "")
    steps      = int(params.get("steps", 20))
    cfg        = float(params.get("cfg_scale", 7.0))
    batch_size = int(params.get("batch_size", 1))
    denoise    = float(params.get("denoising_strength", 0.75))
    seed       = int(params.get("seed", -1))
    if seed == -1:
        seed = _random_seed()

    sampler_name, scheduler = _parse_sampler(params.get("sampler_name", "Euler"), params.get("scheduler"))

    model_ref, clip_ref, vae_ref = _build_common_nodes(b, params)

    pos_id  = b.add("CLIPTextEncode", {"text": positive_prompt, "clip": clip_ref})
    neg_id  = b.add("CLIPTextEncode", {"text": negative_prompt, "clip": clip_ref})
    load_id = b.add("MemoryLoadImage", {"image_b64": init_image_b64})
    enc_id  = b.add("VAEEncode", {"pixels": [load_id, 0], "vae": vae_ref})

    ks_id = b.add("KSampler", {
        "model":        model_ref,
        "positive":     [pos_id, 0],
        "negative":     [neg_id, 0],
        "latent_image": [enc_id, 0],
        "seed":         seed,
        "steps":        steps,
        "cfg":          cfg,
        "sampler_name": sampler_name,
        "scheduler":    scheduler,
        "denoise":      denoise,
    })

    dec_id = b.add("VAEDecode",  {"samples": [ks_id, 0], "vae": vae_ref})
    b.add("MemoryImage", {"images": [dec_id, 0]})

    return b.build()
