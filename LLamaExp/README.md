# LLamaExp: LLaMA-3.2 Vision for Neutrino Event Classification

This directory provides the implementation of Meta's **LLaMA-3.2-11B-Vision-Instruct** model fine-tuned on simulated Liquid Argon Time Projection Chamber (LArTPC) neutrino interactions.

---

## Overview

- **Input Projections**: Dual 2D energy deposition readouts ($zx$ beam view and $zy$ drift view) formatted as 16-bit TIFF files.
- **Physics Target Classes**:
  1. $\nu_e$ CC (Electron Neutrino Charged Current) — diffuse electromagnetic showers originating near vertex.
  2. $\nu_\mu$ CC (Muon Neutrino Charged Current) — continuous, MIP-like muon tracks.
  3. Neutral Current (NC) — hadronic and photon activity with absence of primary leptons.
- **Human-in-the-Loop Interpretability**:
  - Topological and physical reasoning generated prior to event classification (`<explanation>...</explanation>\n\n<label>...</label>`).
  - Cross-attention saliency extraction to align natural language descriptions with energy deposition regions.
  - Hard sample mining to identify ambiguous interactions and evaluate explanation-guided recovery.

---

## Directory Contents

| File | Description |
| :--- | :--- |
| `dataset.py` | PyTorch `Dataset` with TIFF log-scaling, label-leakage scrubbing, and LLaMA chat templating |
| `train_llama.py` | Unified QLoRA fine-tuning script with token-weighted loss, multi-task and stage-1 warm-start support |
| `evaluate_llama.py` | Autoregressive evaluation generating confusion matrices and classification reports |
| `generate_explanations.py` | Batch explanation generation with minimum token constraints for physics fidelity |
| `generate_attention_map.py` | Cross-attention saliency extraction and visualization on $zx$ and $zy$ projections |
| `run_inference.py` | Interactive inference demo supporting standard and structured CoT physics prompts |
| `accelerate_config.yaml` | Multi-GPU distributed training configuration |
| `train.sh` | Launch script for distributed training via Accelerate |

---

## Usage

### 1. Fine-Tuning LLaMA-3.2 Vision

#### Single GPU Fine-Tuning:
```bash
python train_llama.py \
    --epochs 1 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4 \
    --lora_r 8 \
    --label_weight 10.0 \
    --desc_weight 5.0
```

#### Distributed Multi-GPU Training (Accelerate):
```bash
bash train.sh --epochs 1 --batch_size 4 --grad_accum 4
```

#### Warm-Starting from Stage 1 Classifier Adapter:
```bash
python train_llama.py \
    --from_stage1 \
    --stage1_adapter ./adapters/r-8-label-first-no-exp_final \
    --epochs 1
```

---

### 2. Autoregressive Evaluation

```bash
python evaluate_llama.py \
    --adapter_path ./adapters/multitask_solution2_r-8 \
    --output_dir ./predictions/llama_multitask
```

---

### 3. Cross-Attention Saliency Map Extraction

```bash
python generate_attention_map.py \
    --adapter_path ./adapters/multitask_solution2_r-8 \
    --output_dir ./attention_outputs \
    --num_events 5
```

---

### 4. Interactive Inference & Chain-of-Thought

```bash
# Standard Prompt
python run_inference.py --adapter_path ./adapters/multitask_solution2_r-8

# Structured Physics CoT (Vertex, Track, Shower, Topological Conclusion)
python run_inference.py --adapter_path ./adapters/multitask_solution2_r-8 --structured
```
