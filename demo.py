#!/usr/bin/env python3
"""CLI demo for Robust-U1 interleaved reasoning."""

from __future__ import annotations

import argparse
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image

from inference.model_setup import load_inferencer, set_seed
from inference.reasoner import InterleaveReasoner, save_reasoning_results

_DEFAULT_IMAGE = "assets/degraded.png"
_DEFAULT_PROMPT = (
    "Question: Is the light green?\n"
    "Options:\n"
    "A. Yes\n"
    "B. No\n"
    "Please restore this corrupted image to its clean version.\n"
    "Based on what you observe in the restored image, please select the correct answer "
    "from the options above.\n"
)


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Robust-U1 interleaved reasoning (InterleaveReasoner).",
        epilog="If outputs look blurry, try cfg_renorm_type=global or lower cfg / cfg_renorm_min.",
    )
    p.add_argument("--model-path", type=str, default="", help="Local model directory; if omitted, download from --download-repo")
    p.add_argument("--download-repo", type=str, default="Jiaqi-hkust/Robust-U1")
    p.add_argument("--download-dir", type=str, default="./model_weights")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Weights file inside model-path; leave empty to auto-pick (model_bf16.safetensors, model.safetensors, ...)",
    )
    p.add_argument("--image", type=str, default=_DEFAULT_IMAGE, help="Input image path")
    p.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT, help="Task / instruction text")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory")
    p.add_argument(
        "--original-image",
        type=str,
        default="",
        help="Optional clean reference image for interleave prompts",
    )
    p.add_argument("--system-prompt", type=str, default="", help="Optional system prompt")
    p.add_argument("--max-mem-per-gpu", type=str, default="80GiB")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--verbose-iter", action="store_true")
    p.add_argument("--verbose-image", action="store_true")
    p.add_argument(
        "--do-sample",
        action=BooleanOptionalAction,
        default=True,
        help="Text decoding: sample (default) or greedy (--no-do-sample)",
    )
    p.add_argument("--text-temperature", type=float, default=0.3)
    p.add_argument("--cfg-text-scale", type=float, default=4.0)
    p.add_argument("--cfg-img-scale", type=float, default=2.0)
    p.add_argument(
        "--cfg-interval",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=("START", "END"),
        help="CFG denoising interval (fraction of steps)",
    )
    p.add_argument("--timestep-shift", type=float, default=3.0)
    p.add_argument("--num-timesteps", type=int, default=50)
    p.add_argument("--cfg-renorm-min", type=float, default=0.0)
    p.add_argument(
        "--cfg-renorm-type",
        type=str,
        default="text_channel",
        choices=["global", "channel", "text_channel"],
    )
    args = p.parse_args()

    model_path = args.model_path.strip()
    if not model_path:
        from huggingface_hub import snapshot_download

        save_dir = Path(args.download_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = save_dir / "cache"
        print(f"Downloading {args.download_repo} -> {save_dir}", flush=True)
        snapshot_download(
            cache_dir=str(cache_dir),
            local_dir=str(save_dir),
            repo_id=args.download_repo,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
        )
        model_path = str(save_dir)
    else:
        model_path = str(Path(model_path).resolve())

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print("Loading model", flush=True)
    inferencer, _meta = load_inferencer(
        model_path,
        checkpoint_name=args.checkpoint.strip() or None,
        max_mem_per_gpu=args.max_mem_per_gpu,
    )
    reasoner = InterleaveReasoner(inferencer)

    image = Image.open(_resolve_repo_path(args.image)).convert("RGB")
    original_image = None
    if args.original_image.strip():
        original_image = Image.open(_resolve_repo_path(args.original_image)).convert("RGB")

    inputs: list = [image, args.prompt]
    inference_hyper = dict(
        do_sample=args.do_sample,
        text_temperature=args.text_temperature,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=tuple(args.cfg_interval),
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
    )

    print("Running inference", flush=True)
    reasoning_steps = reasoner.reasoning_inference(
        inputs,
        system_prompt=args.system_prompt or None,
        max_iterations=args.max_iterations,
        verbose_iter=args.verbose_iter,
        verbose_image=args.verbose_image,
        original_image=original_image,
        **inference_hyper,
    )

    last_image, text = save_reasoning_results(reasoning_steps, out_dir)

    print(text if text else "(empty)", flush=True)
    if last_image is not None:
        print(f"Result image: {out_dir / 'result.png'}", flush=True)
    else:
        print("No result image.", flush=True)


if __name__ == "__main__":
    main()
