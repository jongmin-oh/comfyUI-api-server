from typing import Any, Optional
from pydantic import BaseModel, Field


class LoraItem(BaseModel):
    name: str                       # 파일명 (확장자 포함, e.g. "handsome man 2.safetensors")
    strength_model: float = 1.0     # 모델에 적용되는 LoRA 강도
    strength_clip: float = 1.0      # CLIP에 적용되는 LoRA 강도


class Txt2ImgRequest(BaseModel):
    # --- Core ---
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = -1
    subseed: int = -1
    subseed_strength: float = 0.0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1

    # --- Sampling ---
    sampler_name: str = "Euler"
    scheduler: Optional[str] = None          # e.g. "karras", "normal" (overrides sampler preset)
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    batch_size: int = 1
    n_iter: int = 1                          # number of batches

    # --- LoRA ---
    loras: list[LoraItem] = Field(default_factory=list)  # 순서대로 체이닝 적용
    clip_skip: int = 1                       # CLIP 레이어 스킵 (workflow의 CLIPSetLastLayer, 1=없음, 2=마지막 1개 스킵)

    # --- Quality ---
    restore_faces: bool = False
    tiling: bool = False

    # --- Highres fix ---
    enable_hr: bool = False
    hr_scale: float = 2.0
    hr_upscaler: str = "Latent"
    hr_second_pass_steps: int = 0
    hr_resize_x: int = 0
    hr_resize_y: int = 0
    hr_checkpoint_name: Optional[str] = None
    hr_sampler_name: Optional[str] = None
    hr_scheduler: Optional[str] = None
    hr_prompt: str = ""
    hr_negative_prompt: str = ""
    denoising_strength: float = 0.7          # used when enable_hr=True

    # --- Model / override ---
    override_settings: dict[str, Any] = Field(default_factory=dict)
    override_settings_restore_afterwards: bool = True

    # --- Refiner (SDXL) ---
    refiner_checkpoint: Optional[str] = None
    refiner_switch_at: float = 0.8

    # --- LoRA / extra networks ---
    disable_extra_networks: bool = False

    # --- Output control ---
    send_images: bool = True
    save_images: bool = False
    do_not_save_samples: bool = False
    do_not_save_grid: bool = False

    # --- Sampler fine-tuning ---
    eta: Optional[float] = None
    s_min_uncond: float = 0.0
    s_churn: float = 0.0
    s_tmax: Optional[float] = None
    s_tmin: float = 0.0
    s_noise: float = 1.0

    # --- Script ---
    script_name: Optional[str] = None
    script_args: list[Any] = Field(default_factory=list)
    alwayson_scripts: dict[str, Any] = Field(default_factory=dict)

    # --- Misc ---
    comments: dict[str, Any] = Field(default_factory=dict)
    force_task_id: Optional[str] = None
    infotext: Optional[str] = None


class Img2ImgRequest(BaseModel):
    # --- Init image ---
    init_images: list[str] = Field(..., description="List of base64-encoded images")
    resize_mode: int = 0                     # 0=Just resize, 1=Crop, 2=Fill, 3=Just resize (latent upscale)
    image_cfg_scale: float = 1.5            # used by instruct-pix2pix

    # --- Inpainting ---
    mask: Optional[str] = None               # base64 mask image
    mask_blur_x: int = 4
    mask_blur_y: int = 4
    mask_blur: int = 4
    mask_round: bool = True
    inpainting_fill: int = 0                 # 0=fill, 1=original, 2=latent noise, 3=latent nothing
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 32
    inpainting_mask_invert: int = 0          # 0=mask painted area, 1=mask not painted area
    initial_noise_multiplier: float = 1.0

    # --- Core (same as txt2img) ---
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = -1
    subseed: int = -1
    subseed_strength: float = 0.0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1

    # --- Sampling ---
    sampler_name: str = "Euler"
    scheduler: Optional[str] = None
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    batch_size: int = 1
    n_iter: int = 1
    denoising_strength: float = 0.75

    # --- LoRA ---
    loras: list[LoraItem] = Field(default_factory=list)
    clip_skip: int = 1

    # --- Quality ---
    restore_faces: bool = False
    tiling: bool = False

    # --- Model / override ---
    override_settings: dict[str, Any] = Field(default_factory=dict)
    override_settings_restore_afterwards: bool = True

    # --- Refiner (SDXL) ---
    refiner_checkpoint: Optional[str] = None
    refiner_switch_at: float = 0.8

    # --- LoRA / extra networks ---
    disable_extra_networks: bool = False

    # --- Output control ---
    send_images: bool = True
    save_images: bool = False
    do_not_save_samples: bool = False
    do_not_save_grid: bool = False

    # --- Sampler fine-tuning ---
    eta: Optional[float] = None
    s_min_uncond: float = 0.0
    s_churn: float = 0.0
    s_tmax: Optional[float] = None
    s_tmin: float = 0.0
    s_noise: float = 1.0

    # --- Script ---
    script_name: Optional[str] = None
    script_args: list[Any] = Field(default_factory=list)
    alwayson_scripts: dict[str, Any] = Field(default_factory=dict)

    # --- Misc ---
    comments: dict[str, Any] = Field(default_factory=dict)
    force_task_id: Optional[str] = None
    infotext: Optional[str] = None
