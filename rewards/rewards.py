# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""图像 restoration / 对齐训练用的 reward：结构 SSIM 与语义 CLIP 相似度。"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


def _to_float_image_tensor(image):
    """Convert a single image to a CHW float tensor in the [0, 1] range."""
    if isinstance(image, Image.Image):
        image = np.asarray(image)
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    image = image.detach().cpu()

    if image.ndim == 2:
        image = image.unsqueeze(0)
    elif image.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, but got shape {tuple(image.shape)}")

    if image.shape[0] not in (1, 3) and image.shape[-1] in (1, 3):
        image = image.permute(2, 0, 1)

    image = image.float()
    if image.numel() > 0 and image.max().item() > 1.0:
        image = image / 255.0

    return image.clamp(0.0, 1.0)


def _to_float_image_batch(images):
    """Convert image inputs to an NCHW float tensor in the [0, 1] range."""
    if isinstance(images, (list, tuple)):
        return torch.stack([_to_float_image_tensor(image) for image in images], dim=0)

    if isinstance(images, Image.Image):
        return _to_float_image_tensor(images).unsqueeze(0)

    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Unsupported image batch type: {type(images)!r}")

    images = images.detach().cpu()

    if images.ndim == 3:
        return _to_float_image_tensor(images).unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(f"Expected a 4D image batch, but got shape {tuple(images.shape)}")

    if images.shape[1] not in (1, 3) and images.shape[-1] in (1, 3):
        images = images.permute(0, 3, 1, 2)

    images = images.float()
    if images.numel() > 0 and images.max().item() > 1.0:
        images = images / 255.0

    return images.clamp(0.0, 1.0)


def structure_score(device):
    """Build an SSIM-based reward function for restoration tasks."""
    from pytorch_msssim import ssim

    def _fn(images, ref_images):
        generated = _to_float_image_batch(images)
        target = _to_float_image_batch(ref_images)

        if generated.shape[0] != target.shape[0]:
            raise ValueError("images and ref_images must contain the same number of items")

        if generated.shape[2:] != target.shape[2:]:
            target = torch.nn.functional.interpolate(
                target,
                size=generated.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        generated = generated.to(device)
        target = target.to(device)

        reward = ssim(generated, target, data_range=1.0, win_size=11, size_average=False)
        return reward.cpu().tolist(), {}

    return _fn


def _to_pil_image_list(images):
    """Convert image inputs to a list of PIL images."""
    images = _to_float_image_batch(images)
    images = (images * 255).round().clamp(0, 255).to(torch.uint8)

    pil_images = []
    for image in images:
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        pil_images.append(Image.fromarray(image.permute(1, 2, 0).numpy()))
    return pil_images


def _images_ref_to_pil_pair(images, ref_images):
    """Convert generated and reference images to aligned PIL lists."""
    gen_pil = _to_pil_image_list(images)
    ref_pil = _to_pil_image_list(ref_images)

    if len(gen_pil) != len(ref_pil):
        raise ValueError("images and ref_images must contain the same number of items")

    return gen_pil, ref_pil


def _resolve_tinyclip_local_weights() -> Tuple[Optional[str], Optional[str]]:
    """
    Optional local ViT-B-32 weights for open_clip (TinyCLIP / OpenAI CLIP).

    Set environment variable TINYCLIP_WEIGHTS_PATH to a .safetensors or .pth/.pt file.
    If unset or file missing, returns (None, None) and tinyclip_score uses pretrained='openai'.
    """
    path = os.environ.get("TINYCLIP_WEIGHTS_PATH", "").strip()
    if not path or not os.path.isfile(path):
        return None, None
    lower = path.lower()
    if lower.endswith(".safetensors"):
        return path, "safetensors"
    return path, "torch"


def tinyclip_score(device):
    """Build a TinyCLIP-based reward function for image similarity."""
    import open_clip
    import torch.nn.functional as F

    model_name = "ViT-B-32"
    alpha = float(os.environ.get("TINYCLIP_SIMILARITY_ALPHA", "5.0"))
    local_weights_path, weight_fmt = _resolve_tinyclip_local_weights()

    if local_weights_path:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None)
        if weight_fmt == "safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(local_weights_path)
        else:
            state_dict = torch.load(local_weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")

    model = model.to(device)
    model.eval()

    def _fn(images, ref_images):
        gen_pil, ref_pil = _images_ref_to_pil_pair(images, ref_images)
        gen_tensors = torch.stack([preprocess(img) for img in gen_pil], dim=0).to(device)
        ref_tensors = torch.stack([preprocess(img) for img in ref_pil], dim=0).to(device)

        with torch.inference_mode():
            gen_features = F.normalize(model.encode_image(gen_tensors), dim=-1)
            ref_features = F.normalize(model.encode_image(ref_tensors), dim=-1)
            similarity = (gen_features * ref_features).sum(dim=-1)
            similarity = similarity.clamp(0.0, 1.0)
            similarity = torch.exp(-alpha * (1.0 - similarity))

        return similarity.cpu().tolist(), {}

    return _fn
