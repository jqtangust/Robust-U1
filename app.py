#!/usr/bin/env python3
"""
Gradio demo for local Robust-U1 image editing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gradio as gr
import numpy as np
from PIL import Image

from inference._paths import REPO_ROOT
from inference.model_setup import load_inferencer, set_seed

DEFAULT_PROMPT = (
    "Is the light green?\n"
    "Options:\n"
    "A. Yes\n"
    "B. No"
)

HIDDEN_PROMPT_SUFFIX = (
    "Please restore this corrupted image to its clean version.\n"
    "Based on what you observe in the restored image, please select the correct answer from the options above."
)

APP_CSS = """
:root {
    --page-max-width: 100vw;
    --ink-strong: #102a43;
    --ink-soft: #486581;
    --surface-primary: rgba(255, 255, 255, 0.92);
    --surface-secondary: rgba(244, 247, 251, 0.88);
    --surface-border: rgba(148, 163, 184, 0.24);
    --surface-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
    --accent-start: #0f766e;
    --accent-end: #0b5ed7;
}

body {
    background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 34%),
        radial-gradient(circle at top right, rgba(14, 116, 144, 0.14), transparent 30%),
        linear-gradient(180deg, #eef4f7 0%, #f8fbfd 52%, #eef3f8 100%);
}

body,
.gradio-container,
input,
textarea,
button {
    font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif !important;
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    padding: 24px 12px 40px !important;
}

.app-shell {
    gap: 20px;
}

.hero-card {
    position: relative;
    overflow: hidden;
    padding: 26px 34px 24px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(232, 243, 248, 0.96) 100%);
    border: 1px solid rgba(148, 163, 184, 0.2);
    box-shadow: var(--surface-shadow);
}

.hero-card::after {
    content: "";
    position: absolute;
    inset: auto -80px -110px auto;
    width: 260px;
    height: 260px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(11, 94, 215, 0.15), transparent 68%);
}

.hero-title {
    margin: 0;
    font-size: clamp(2.1rem, 4vw, 3.4rem);
    line-height: 1.05;
    color: var(--ink-strong);
}

.hero-subtitle {
    max-width: 760px;
    margin: 12px 0 0;
    font-size: 0.98rem;
    line-height: 1.6;
    color: #52606d;
}

.panel {
    border: 1px solid var(--surface-border);
    border-radius: 24px;
    background: var(--surface-primary);
    box-shadow: var(--surface-shadow);
    padding: 20px;
    backdrop-filter: blur(12px);
}

.panel-header {
    margin-bottom: 14px;
}

.eyebrow {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #486581;
}

.panel-title {
    margin-top: 4px;
    font-size: 1.22rem;
    font-weight: 600;
    color: var(--ink-strong);
}

.control-bar {
    margin-top: 16px;
}

.toolbar {
    align-items: center;
    gap: 12px;
}

.accordion-shell {
    border-radius: 24px !important;
    border: 1px solid var(--surface-border) !important;
    background: var(--surface-primary) !important;
    box-shadow: var(--surface-shadow);
}

.accordion-shell > .label-wrap {
    padding-top: 4px;
    padding-bottom: 4px;
}

#run-button {
    min-height: 52px;
    border: none !important;
    background: linear-gradient(135deg, var(--accent-start) 0%, var(--accent-end) 100%) !important;
    box-shadow: 0 18px 32px rgba(11, 94, 215, 0.22);
}

#run-button:hover {
    filter: brightness(1.03);
}

#secondary-button {
    min-height: 52px;
    border: 1px solid rgba(148, 163, 184, 0.24) !important;
    background: rgba(248, 250, 252, 0.9) !important;
    color: var(--ink-strong) !important;
}

.input-image,
.output-image {
    width: 100% !important;
}

.input-image img,
.output-image img {
    border-radius: 18px;
}

.reasoning-box textarea {
    line-height: 1.55;
}

.footer-note {
    margin-top: 6px;
    font-size: 0.95rem;
    line-height: 1.65;
    color: #52606d;
}

@media (max-width: 900px) {
    .hero-card {
        padding: 22px 22px 20px;
    }

    .panel {
        padding: 16px;
    }

    .toolbar {
        flex-direction: column;
        align-items: stretch;
    }
}
"""


def _build_header_html() -> str:
    return """
    <section class="hero-card">
      <h1 class="hero-title">Robust-U1</h1>
      <p class="hero-subtitle">
        Robust-U1: Can MLLMs Self-Recover Corrupted Visual Content for Robust Understanding?
      </p>
    </section>
    """


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust-U1 Gradio demo")
    parser.add_argument("--model-path", type=str, default=None, help="Path to a local model directory")
    args = parser.parse_args()
    model_path = args.model_path

    inferencer = None
    pil_img2rgb = None
    if model_path:
        inferencer, meta = load_inferencer(model_path)
        pil_img2rgb = meta["pil_img2rgb"]

    def edit_image(
        image,
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
    ):
        if image is None:
            yield None, "Upload an image to begin."
            return
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if inferencer is None or pil_img2rgb is None:
            yield image, "Preview mode is active. Launch with --model-path to enable inference."
            return

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
        result_text = ""
        last_image = None
        model_prompt = (
            f"{prompt.rstrip()}\n\n{HIDDEN_PROMPT_SUFFIX}" if prompt.strip() else HIDDEN_PROMPT_SUFFIX
        )
        for chunk in inferencer(
            image=image,
            text=model_prompt,
            think=show_thinking,
            understanding_output=False,
            **inference_hyper,
        ):
            if isinstance(chunk, str):
                result_text += chunk
            else:
                last_image = chunk
            yield last_image, result_text

    def update_edit_thinking_visibility(show):
        return gr.update(visible=show), gr.update(visible=show)

    default_in = REPO_ROOT / "assets" / "degraded.png"
    if not default_in.is_file():
        default_in = None
    default_pil = Image.open(default_in).convert("RGB") if default_in else None

    def reset_workspace():
        return (
            default_pil,
            DEFAULT_PROMPT,
            None,
            gr.update(value="", visible=True),
            True,
            4.0,
            2.0,
            0.0,
            3.0,
            50,
            1.0,
            "text_channel",
            1024,
            False,
            0.3,
            0,
            gr.update(visible=False),
        )

    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
    )

    with gr.Blocks(
        theme=theme,
        css=APP_CSS,
        title="Robust-U1",
        elem_classes="app-shell",
        fill_width=True,
    ) as demo:
        gr.HTML(_build_header_html())

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes="panel"):
                gr.HTML(
                    """
                    <div class="panel-header">
                      <div class="eyebrow">Input</div>
                      <div class="panel-title">Source Image and Edit Instruction</div>
                    </div>
                    """
                )
                edit_image_input = gr.Image(
                    label="Input image",
                    value=default_pil,
                    type="pil",
                    sources=["upload", "clipboard"],
                    elem_classes="input-image",
                )
                edit_prompt = gr.Textbox(
                    label="Prompt",
                    value=DEFAULT_PROMPT,
                    placeholder="Enter an edit instruction or question.",
                    lines=4,
                )
                with gr.Row(elem_classes="control-bar toolbar"):
                    edit_show_thinking = gr.Checkbox(label="Show reasoning trace", value=True)
                    edit_btn = gr.Button("Run Inference", variant="primary", elem_id="run-button")
                    reset_btn = gr.Button("Reset Workspace", elem_id="secondary-button")

            with gr.Column(scale=1, elem_classes="panel"):
                gr.HTML(
                    """
                    <div class="panel-header">
                      <div class="eyebrow">Output</div>
                      <div class="panel-title">Generated Result and Trace</div>
                    </div>
                    """
                )
                edit_image_output = gr.Image(
                    label="Edited output",
                    type="pil",
                    interactive=False,
                    show_download_button=True,
                    elem_classes="output-image",
                )
                edit_thinking_output = gr.Textbox(
                    label="Reasoning trace",
                    visible=True,
                    lines=8,
                    show_copy_button=True,
                    elem_classes="reasoning-box",
                )

        with gr.Accordion("Advanced Inference Controls", open=False, elem_classes="accordion-shell"):
            with gr.Row():
                edit_seed = gr.Slider(
                    minimum=0,
                    maximum=1_000_000,
                    value=0,
                    step=1,
                    label="Seed",
                    info="Use 0 for a non-deterministic run, or a positive integer for reproducibility.",
                )
                edit_cfg_text_scale = gr.Slider(
                    minimum=1.0,
                    maximum=8.0,
                    value=4.0,
                    step=0.1,
                    label="CFG text scale",
                )
            with gr.Row():
                edit_cfg_img_scale = gr.Slider(
                    minimum=1.0,
                    maximum=4.0,
                    value=2.0,
                    step=0.1,
                    label="CFG image scale",
                )
                edit_cfg_interval = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    label="CFG activation start",
                )
            with gr.Row():
                edit_cfg_renorm_type = gr.Dropdown(
                    choices=["global", "channel", "text_channel"],
                    value="text_channel",
                    label="CFG renormalization type",
                )
                edit_cfg_renorm_min = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.1,
                    label="CFG renormalization floor",
                )
            with gr.Row():
                edit_num_timesteps = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Diffusion timesteps",
                )
                edit_timestep_shift = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=3.0,
                    step=0.5,
                    label="Timestep shift",
                )
            edit_thinking_params = gr.Group(visible=True)
            with edit_thinking_params:
                with gr.Row():
                    edit_do_sample = gr.Checkbox(label="Sample reasoning tokens", value=False)
                    edit_max_think_token_n = gr.Slider(
                        minimum=64,
                        maximum=4006,
                        value=1024,
                        step=64,
                        label="Max reasoning tokens",
                    )
                    edit_text_temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="Reasoning temperature",
                    )

        edit_show_thinking.change(
            fn=update_edit_thinking_visibility,
            inputs=[edit_show_thinking],
            outputs=[edit_thinking_output, edit_thinking_params],
        )

        reset_btn.click(
            fn=reset_workspace,
            inputs=None,
            outputs=[
                edit_image_input,
                edit_prompt,
                edit_image_output,
                edit_thinking_output,
                edit_show_thinking,
                edit_cfg_text_scale,
                edit_cfg_img_scale,
                edit_cfg_interval,
                edit_timestep_shift,
                edit_num_timesteps,
                edit_cfg_renorm_min,
                edit_cfg_renorm_type,
                edit_max_think_token_n,
                edit_do_sample,
                edit_text_temperature,
                edit_seed,
                edit_thinking_params,
            ],
        )

        gr.on(
            triggers=[edit_btn.click, edit_prompt.submit],
            fn=edit_image,
            inputs=[
                edit_image_input,
                edit_prompt,
                edit_show_thinking,
                edit_cfg_text_scale,
                edit_cfg_img_scale,
                edit_cfg_interval,
                edit_timestep_shift,
                edit_num_timesteps,
                edit_cfg_renorm_min,
                edit_cfg_renorm_type,
                edit_max_think_token_n,
                edit_do_sample,
                edit_text_temperature,
                edit_seed,
            ],
            outputs=[edit_image_output, edit_thinking_output],
        )

        gr.HTML(
            """
            <div class="footer-note">
              Inference is powered by the local <code>modeling</code> package used in this repository.
              Launch with <code>--model-path</code> to load weights and enable inference.
            </div>
            """
        )

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))


if __name__ == "__main__":
    main()
