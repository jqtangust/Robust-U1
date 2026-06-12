"""
单卡版 InterleaveReasoner，逻辑对齐仓库根目录 ``interleave_reasoner.py``。
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from tqdm.auto import tqdm

from ._paths import ensure_repo_on_path

ensure_repo_on_path()

from modeling.data.data_utils import pil_img2rgb  # noqa: E402


class InterleaveReasoner:
    """多步文本-图像交错推理（无分布式依赖）。"""

    def __init__(self, inferencer):
        self.inferencer = inferencer
        self.tokenizer = inferencer.tokenizer
        self.new_token_ids = inferencer.new_token_ids
        self.action_token_map = {
            self.new_token_ids["start_of_image"]: "image",
            self.new_token_ids["bos_token_id"]: "text",
            self.new_token_ids["end_of_text"]: "end",
        }

    def generate_text_with_next_token_check(
        self,
        gen_context,
        max_length: int = 2048,
        do_sample: bool = True,
        temperature: float = 0.3,
    ):
        generated_ids = self.inferencer.gen_text(
            gen_context,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            return_ids=True,
        )
        next_action_token = generated_ids[-1]
        text_token_ids = generated_ids[1:-2]
        next_action = self.action_token_map.get(next_action_token, "undefined")
        decoded_text = self.tokenizer.decode(text_token_ids).strip()

        answer_tag = "<ANSWER>"
        max_answer_count = 3
        answer_count = decoded_text.count(answer_tag)
        if answer_count > max_answer_count:
            answer_positions = []
            start_pos = 0
            while True:
                pos = decoded_text.find(answer_tag, start_pos)
                if pos == -1:
                    break
                answer_positions.append(pos)
                start_pos = pos + len(answer_tag)
            if len(answer_positions) >= max_answer_count:
                cutoff_pos = answer_positions[max_answer_count - 1]
                decoded_text = decoded_text[:cutoff_pos]

        return decoded_text, next_action

    @staticmethod
    def _unwrap_gen_image(result):
        if isinstance(result, dict):
            return result.get("image", result)
        return result

    @torch.no_grad()
    def reasoning_inference(
        self,
        inputs: List[Union[str, Image.Image]],
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose_iter: bool = False,
        verbose_image: bool = False,
        original_image: Optional[Image.Image] = None,
        **inference_kwargs,
    ) -> List[Dict[str, Any]]:
        reasoning_steps: List[Dict[str, Any]] = []

        gen_context = self.inferencer.init_gen_context()
        cfg_text_context = self.inferencer.init_gen_context()
        cfg_img_context = self.inferencer.init_gen_context()
        image_shapes = (512, 512)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if system_prompt:
                gen_context = self.inferencer.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.inferencer.update_context_text(system_prompt, cfg_img_context)

            for input_item in inputs:
                if isinstance(input_item, str):
                    gen_context = self.inferencer.update_context_text(input_item, gen_context)
                    cfg_img_context = self.inferencer.update_context_text(input_item, cfg_img_context)
                elif isinstance(input_item, Image.Image):
                    processed_image = self.inferencer.vae_transform.resize_transform(
                        pil_img2rgb(input_item)
                    )
                    image_shapes = processed_image.size[::-1]
                    gen_context = self.inferencer.update_context_image(
                        processed_image, gen_context, vae=True, vit=True
                    )
                    cfg_text_context = self.inferencer.update_context_image(
                        processed_image, cfg_text_context, vae=True, vit=True
                    )

            if original_image is not None:
                original_text = "Here is the original image:"
                gen_context = self.inferencer.update_context_text(original_text, gen_context)
                cfg_img_context = self.inferencer.update_context_text(original_text, cfg_img_context)
                reasoning_steps.append({"type": "text", "content": original_text, "iteration": 0})
                processed_original = self.inferencer.vae_transform.resize_transform(
                    pil_img2rgb(original_image)
                )
                gen_context = self.inferencer.update_context_image(
                    processed_original, gen_context, vae=True, vit=True
                )
                cfg_text_context = self.inferencer.update_context_image(
                    processed_original, cfg_text_context, vae=True, vit=True
                )
                reasoning_steps.append(
                    {"type": "image", "content": original_image, "injected": "original", "iteration": 0}
                )
                restored_text = (
                    "Please answer the question based on the restored image.Here is the restored image:"
                )
                gen_context = self.inferencer.update_context_text(restored_text, gen_context)
                cfg_img_context = self.inferencer.update_context_text(restored_text, cfg_img_context)
                reasoning_steps.append({"type": "text", "content": restored_text, "iteration": 0})

            current_mode = "text"
            with tqdm(
                total=max_iterations,
                desc="Reasoning Steps",
                leave=False,
                disable=not verbose_iter,
            ) as pbar:
                for iteration in range(max_iterations):
                    if current_mode == "text":
                        pbar.set_description(f"Step {iteration + 1}/{max_iterations}: Generating text")
                        generated_text, next_action = self.generate_text_with_next_token_check(
                            gen_context,
                            do_sample=inference_kwargs.get("do_sample", True),
                            temperature=inference_kwargs.get("text_temperature", 0.3),
                        )
                        gen_context = self.inferencer.update_context_text(generated_text, gen_context)
                        cfg_img_context = self.inferencer.update_context_text(
                            generated_text, cfg_img_context
                        )
                        reasoning_steps.append(
                            {"type": "text", "content": generated_text, "iteration": iteration + 1}
                        )
                        if next_action == "end":
                            pbar.update(1)
                            break
                        current_mode = "image" if next_action == "image" else "text"

                    elif current_mode == "image":
                        pbar.set_description(f"Step {iteration + 1}/{max_iterations}: Generating image")
                        try:
                            gen_image_kwargs = {
                                k: v
                                for k, v in inference_kwargs.items()
                                if k.startswith("cfg_")
                                or k in ["timestep_shift", "num_timesteps", "enable_taylorseer"]
                            }
                            raw = self.inferencer.gen_image(
                                image_shapes,
                                gen_context,
                                cfg_text_precontext=cfg_text_context,
                                cfg_img_precontext=cfg_img_context,
                                **gen_image_kwargs,
                            )
                            generated_image = self._unwrap_gen_image(raw)
                            reasoning_steps.append(
                                {"type": "image", "content": generated_image, "iteration": iteration + 1}
                            )
                            processed_image = self.inferencer.vae_transform.resize_transform(
                                pil_img2rgb(generated_image)
                            )
                            gen_context = self.inferencer.update_context_image(
                                processed_image, gen_context, vae=True, vit=True
                            )
                            cfg_text_context = self.inferencer.update_context_image(
                                processed_image, cfg_text_context, vae=True, vit=True
                            )
                            current_mode = "text"
                        except Exception as e:
                            traceback.print_exc()
                            print(f"图像生成失败: {e}", flush=True)
                            current_mode = "text"

                    pbar.update(1)

        return reasoning_steps


def save_reasoning_results(
    reasoning_steps: List[Dict[str, Any]],
    output_dir: Path,
) -> tuple[Optional[Image.Image], str]:
    """Write step JSON and images under output_dir; return (last image, concatenated text)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = []
    text_parts: List[str] = []
    last_image: Optional[Image.Image] = None
    img_idx = 0

    for step in reasoning_steps:
        entry = {k: v for k, v in step.items() if k != "content"}
        if step["type"] == "text":
            content = step["content"]
            entry["content"] = content
            text_parts.append(content)
            serializable.append(entry)
        else:
            img_idx += 1
            img_path = output_dir / f"step_{img_idx:02d}.png"
            img = step["content"]
            if isinstance(img, Image.Image):
                img.save(img_path)
                last_image = img
                entry["content_path"] = img_path.name
            serializable.append(entry)

    summary = "\n\n".join(text_parts)
    (output_dir / "reasoning_result.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if last_image is not None:
        last_image.save(output_dir / "result.png")
    return last_image, summary
