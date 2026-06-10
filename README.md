<div align="center">

# [ICML 2026] *Robust-U1*: *Can MLLMs Self-Recover Corrupted Visual Content for Robust Understanding?*
This is the official repository for *Robust-U1*.

[Jiaqi Tang^](https://jqt.me/), 
[Jianmin Chen^](https://github.com/Ch921-cell), 
[Youyang Zhai^](), 
\
[Wei Wei**](https://scholar.google.com/citations?hl=zh-CN&user=v8KMYlwAAAAJ), 
[Runtao Liu](), 
[Mengjie Zhao](), 
[Xiangyu Wu](), 
[Qingfa Xiao](), and 
\
[Qifeng Chen*](https://cqf.io)

^: Equal contribution. *: Corresponding author. **: Co-corresponding author.

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://openreview.net/forum?id=I6W6cxVVts)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffd21e)](https://huggingface.co/Jiaqi-hkust/Robust-U1)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

## 📰 **News**
- **[2026-06-07]** 🔥 *We release the [code](https://github.com/jqtangust/Robust-U1) and [models](https://huggingface.co/Jiaqi-hkust/Robust-U1) on Hugging Face.*
- **[2026-05-07]** 🚀 Our paper is accepted by **ICML 2026**.

---

## 🔭 **Motivation**

- 🚩 **Black-Box Alignment**: Existing feature-alignment methods lack interpretability and fail to explicitly model the corruption process.
- 🚩 **Text-Only Compensation**: Text-based reasoning cannot recover lost pixel-level visual details for faithful visual understanding.

This motivates a key question: Can MLLMs recover corrupted visual content by themselves?

<div align="center">
  <img src="assets/Motivation.png" width="90%" alt="Motivation Overview">
  <br>
</div>

---

## 🛠️ **Installation**

- **Clone the repository:**
   ```bash
   git clone https://github.com/jqtangust/Robust-U1.git
   cd Robust-U1
   ```

- **Create environment:**
   ```bash
   conda create -n Robust-U1 python=3.10
   conda activate Robust-U1
   pip install -r requirements.txt
   pip install -e .
   ```
   
---

### 🏰 **Pretrained checkpoints (reference)**

| Checkpoint | Link | Note |
|:----------:|:----:|:----:|
| BAGEL-7B-MoT | [ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) | Used as initial weights for training. |
| **Robust-U1** | [Jiaqi-hkust/Robust-U1](https://huggingface.co/Jiaqi-hkust/Robust-U1) | Final model for visual self-recovery and multimodal reasoning. |
| **Robust-U1-RL** | [Jiaqi-hkust/Robust-U1-RL](https://huggingface.co/Jiaqi-hkust/Robust-U1-RL) | Fine-tuned with reinforcement learning. |
| **Robust-U1-SFT** | [Jiaqi-hkust/Robust-U1-SFT](https://huggingface.co/Jiaqi-hkust/Robust-U1-SFT) | Fine-tuned with supervised learning. |

---

## ⏳ **Demo**

### 🖥️ CLI

Run the command-line demo with a local model path and an output directory for recovered images:

```bash
export MODEL_PATH="/path/to/Robust-U1"
export OUTPUT_DIR="./outputs"

python demo.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR"
```

### 🌐 GUI

Set the model path and start the local Gradio demo:

```bash
export MODEL_PATH="/path/to/Robust-U1"
python app.py --model-path "$MODEL_PATH"
```

The demo is available at `http://localhost:7860` by default.

GUI online demo: [Hugging Face Space](https://huggingface.co/spaces/Jiaqi-hkust/Robust-U1).

<div align="center">
  <img src="assets/demo.png" alt="Robust-U1 Demo">
</div>

---


## 🧠 **Training**

Robust-U1 is trained with a three-stage pipeline: visual self-recovery, reinforcement learning for visual quality alignment, and multimodal reasoning for robust visual understanding.

### 🎓 Stage I & III:

We use [MathCanvas](https://github.com/shiwk24/MathCanvas/) for both supervised fine-tuning and multimodal reasoning training. Stage I adapts the base unified MLLM to recover clean images from corrupted inputs, while Stage III trains the model to reason over both corrupted and recovered images.

1. Prepare the MathCanvas training framework:

   ```bash
   git clone https://github.com/shiwk24/MathCanvas.git
   cd MathCanvas/BAGEL-Canvas
   ```

2. Download the base model [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT).

3. Prepare the training data:

   * For Stage I, prepare paired corrupted-clean image data for visual self-recovery.
   * For Stage III, prepare reasoning data with corrupted images, recovered images, questions, and reasoning-chain annotations.

4. Modify the dataset paths in `data/dataset_info.py` and configure the corresponding training scripts with your local paths.

5. Run Stage-I supervised fine-tuning to obtain the SFT checkpoint:

   ```bash
   bash scripts/train/stage1.sh
   ```

6. After Stage-II reinforcement learning, run Stage-III multimodal reasoning training:

   ```bash
   bash scripts/train/stage2.sh
   ```

### 🎓 Stage II:

We use [Flow-GRPO](https://github.com/yifan123/flow_grpo) to further align the recovery model with pixel-level structural fidelity and semantic consistency. The Robust-U1 rewards are packaged in [`rewards/`](./rewards) and can be registered directly in Flow-GRPO.

1. Prepare Flow-GRPO and expose Robust-U1 rewards:

   ```bash
   git clone https://github.com/yifan123/flow_grpo.git
   cd flow_grpo
   ```

2. Register the Robust-U1 reward adapter in `flow_grpo/rewards.py`:

   ```python
   from rewards import FLOW_GRPO_REFERENCE_REWARD_NAMES, register_flow_grpo_rewards

   # after Flow-GRPO builds score_functions
   register_flow_grpo_rewards(score_functions)

   # reference-based rewards use clean target images
   elif score_name in FLOW_GRPO_REFERENCE_REWARD_NAMES:
       scores, rewards = score_fns[score_name](images, ref_images)
   ```

3. Prepare restoration data with corrupted images and clean references. Each JSONL record should contain:

   ```json
   {"prompt": "Please restore this corrupted image to its clean version.", "image": "corrupted/000001.png", "target_image": "clean/000001.png"}
   ```

4. Configure `config/grpo.py`:

   ```python
   config.dataset = "/path/to/dataset/restoration"
   config.pretrained.model = "/path/to/Robust-U1-SFT"
   config.reward_fn = {
       "restoration": 1.0,
       "tinyclip": 0.2,
   }
   ```

5. Run reinforcement learning:

   ```bash
   bash scripts/multi_node/bagel/main.sh 0
   ```

   The launcher should point to the restoration config, for example:

   ```bash
   accelerate launch --config_file scripts/accelerate_configs/fsdp.yaml \
     --num_processes 8 \
     scripts/train_bagel.py \
     --config config/grpo.py:restoration_bagel
   ```


---

## 📊 **Evaluation**

We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for anti-degradation evaluation.

1. Clone the VLMEvalKit repository and install dependencies:

   ```bash
   git clone https://github.com/open-compass/VLMEvalKit.git
   cd VLMEvalKit
   pip install -e .
   ```

2. Prepare the evaluation datasets according to VLMEvalKit requirements.


3. **Image Degradation Pipeline**: Generate corrupted images for robustness evaluation.

   We provide an image degradation pipeline for generating corrupted images to evaluate model robustness.

   Navigate to the degradation pipeline directory and process images:

   ```bash
   cd add_degradation
   python generate_pipeline_open_source.py --input_dir <input_dir> --output_base_dir <output_base_dir> --dataset_name <dataset_name> --verbose
   ```

   The script will generate three output directories with different degradation intensities for each image.

4. Configure the model path and evaluation settings in the VLMEvalKit configuration file.

5. Run the evaluation command:

   ```bash
   python run.py --model <your_model_name_or_path> --data <dataset_name>
   ```

### 🔬 R-Bench Evaluation

For R-Bench evaluation, we use [R-Bench](https://github.com/Q-Future/R-Bench) to assess model performance under real-world corruptions.

1. Clone the R-Bench repository:

   ```bash
   git clone https://github.com/Q-Future/R-Bench.git
   ```

2. Evaluate using VLMEvalKit with R-Bench dataset:

   ```bash
   cd VLMEvalKit
   python run.py --data R-Bench-Dis --model <your_model_name_or_path> --verbose
   ```

3. For full dataset evaluation, follow the R-Bench pipeline as described in the [R-Bench repository](https://github.com/Q-Future/R-Bench).

---

## ⭐️ **Citation**

If you find this repository useful, please cite our paper:

```bibtex
@misc{tang2026robustu1mllmsselfrecovercorrupted,
      title={Robust-U1: Can MLLMs Self-Recover Corrupted Visual Content for Robust Understanding?}, 
      author={Jiaqi Tang and Jianmin Chen and Youyang Zhai and Wei Wei and Runtao Liu and Mengjie Zhao and Xiangyu Wu and Qingfa Xiao and Qifeng Chen},
      year={2026},
      eprint={2606.08063},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.08063}, 
}
```

---

## 🤝 **Acknowledgements**

We thank the authors of [BAGEL](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT), [MathCanvas](https://github.com/shiwk24/MathCanvas/) and [Flow-GRPO](https://github.com/yifan123/flow_grpo) for their contributions.