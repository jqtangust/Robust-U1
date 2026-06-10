"""
加载 Robust-U1（BAGEL）推理栈：使用本仓库内的 ``modeling`` 包（与 Stage2 ``flow_grpo/bagel`` 同源）。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from PIL import Image

from ._paths import ensure_repo_on_path

_CHECKPOINT_CANDIDATES = (
    "model_bf16.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
)


def _resolve_checkpoint(model_path: Path, checkpoint_name: Optional[str]) -> Path:
    """Pick a single-file checkpoint under ``model_path``.

    If ``checkpoint_name`` is set, only that path is used (must exist).
    Otherwise the first existing file from ``_CHECKPOINT_CANDIDATES`` is used.
    """
    explicit = (checkpoint_name or "").strip()
    if explicit:
        ckpt = model_path / explicit
        if ckpt.is_file():
            return ckpt
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    for name in _CHECKPOINT_CANDIDATES:
        ckpt = model_path / name
        if ckpt.is_file():
            return ckpt
    tried = ", ".join(_CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(
        f"No checkpoint in {model_path}. Tried (auto): {tried}"
    )


def set_seed(seed: int) -> None:
    if seed <= 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_inferencer(
    model_path: str | Path,
    *,
    checkpoint_name: Optional[str] = None,
    max_mem_per_gpu: str = "80GiB",
) -> Tuple[Any, Any]:
    """
    返回 ``(inferencer, meta)``，其中 ``meta`` 含 ``pil_img2rgb``、``tokenizer``、``model_path``。

    ``checkpoint_name``: 模型目录下的权重文件名；为 ``None`` 或空字符串时在目录内按常见名称自动查找。
    """
    import os

    ensure_repo_on_path()

    from modeling.data.data_utils import add_special_tokens, pil_img2rgb
    from modeling.data.transforms import ImageTransform
    from modeling.inferencer import InterleaveInferencer
    from modeling.modeling.autoencoder import load_ae
    from modeling.modeling.bagel import (
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from modeling.modeling.qwen2 import Qwen2Tokenizer

    model_path = Path(model_path).resolve()

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device if k in device_map else "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    ckpt = _resolve_checkpoint(model_path, checkpoint_name)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(ckpt),
        device_map=device_map,
        offload_buffers=True,
        offload_folder="offload",
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()

    vae_device = device_map.get(same_device_modules[0], "cuda:0")
    vae_model = vae_model.to(device=vae_device, dtype=torch.bfloat16).eval()

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    meta = {
        "pil_img2rgb": pil_img2rgb,
        "tokenizer": tokenizer,
        "model_path": str(model_path),
    }
    return inferencer, meta


def run_image_edit(
    inferencer,
    *,
    pil_img2rgb,
    image: Image.Image,
    prompt: str,
    show_thinking: bool = False,
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 2.0,
    cfg_interval: float = 0.0,
    timestep_shift: float = 3.0,
    num_timesteps: int = 50,
    cfg_renorm_min: float = 1.0,
    cfg_renorm_type: str = "text_channel",
    max_think_token_n: int = 1024,
    do_sample: bool = False,
    text_temperature: float = 0.3,
    seed: int = 0,
) -> Tuple[Optional[Image.Image], str]:
    """单次编辑，返回 (最终图像, 累积文本)."""
    set_seed(seed)
    image = pil_img2rgb(image)
    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[cfg_interval, 1.0],
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )
    text_out = ""
    last_image: Optional[Image.Image] = None
    for chunk in inferencer(
        image=image,
        text=prompt,
        think=show_thinking,
        understanding_output=False,
        **inference_hyper,
    ):
        if isinstance(chunk, str):
            text_out += chunk
        else:
            last_image = chunk
    return last_image, text_out
