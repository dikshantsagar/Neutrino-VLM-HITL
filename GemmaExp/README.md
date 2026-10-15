# GemmaExp: Gemma Vision for Neutrino Event Classification

This directory provides scripts and tools for fine-tuning and evaluating Google's Gemma Vision models (Gemma-4-E4B, Gemma-4-12B, Gemma-4-27B) on simulated Liquid Argon Time Projection Chamber (LArTPC) neutrino interactions.

---

## Overview

- **Input Data**: Dual 2D pixel energy projections ($zx$ beam view and $zy$ drift view) formatted as 16-bit TIFF files.
- **Physics Target Classes**:
  1. $\nu_e$ CC (Electron Neutrino Charged Current) — characterized by fuzzy electromagnetic showers.
  2. $\nu_\mu$ CC (Muon Neutrino Charged Current) — characterized by straight, minimum-ionizing muon tracks.
  3. Neutral Current (NC) — hadronic and photon activity lacking primary charged lepton signatures.
- **Model Architecture**: Gemma Vision multi-modal models fine-tuned with 4-bit QLoRA and token-weighted next-token supervision.

---

## Directory Contents

| File | Purpose |
| :--- | :--- |
| `dataset.py` | PyTorch `Dataset` with TIFF log-scaling, label-leakage scrubbing, and Gemma chat templating |
| `train_gemma.py` | Unified QLoRA fine-tuning script across Gemma model sizes (E4B, 4b, 12b, 27b) |
| `evaluate_gemma.py` | Autoregressive evaluation generating confusion matrices and classification reports |
| `compute_metrics.py` | Metric calculator (Accuracy, Macro Precision, Recall, One-vs-Rest ROC-AUC) |
| `inference_gemma.py` | Interactive / single-event inference demo displaying ground truth vs model reasoning |

---

## Usage

### 1. Fine-Tuning Gemma Models

```bash
# Fine-tune Gemma-4-E4B with explanations
python train_gemma.py --model_size E4B --epochs 1 --lr 2e-4 --batch_size 4

# Fine-tune Gemma-4-12B
python train_gemma.py --model_size 12b --batch_size 2 --grad_accum 8

# Label-only baseline (without explanation generation)
python train_gemma.py --model_size E4B --without_explanations
```

### 2. Autoregressive Evaluation

```bash
# Evaluate fine-tuned model checkpoint
python evaluate_gemma.py \
    --model_size E4B \
    --adapter_path ./adapters/gemma_4_E4B_r-8_exp_first \
    --output_dir ./predictions/gemma_4_E4B
```

### 3. Compute Comprehensive Metrics

```bash
python compute_metrics.py --dir ./predictions
```

### 4. Interactive Single-Event Inference

```bash
python inference_gemma.py \
    --model_size E4B \
    --adapter_path ./adapters/gemma_4_E4B_r-8_exp_first
```
