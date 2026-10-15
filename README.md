# Neutrino-VLM-HITL: Explainable Vision-Language Models for Neutrino Event Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-QLoRA-green.svg)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**Neutrino-VLM-HITL** is a scientific framework applying multi-modal Vision-Language Models (VLMs) with **Human-in-the-Loop (HITL)** verification and topological reasoning to classify simulated Liquid Argon Time Projection Chamber (LArTPC) neutrino interaction events (e.g., for the Deep Underground Neutrino Experiment, DUNE).

Rather than treating deep learning models as uninterpretable black boxes, this repository integrates:
1. **Multi-Modal Event Classification**: Distinguishing $\nu_e$ Charged Current (CC), $\nu_\mu$ Charged Current (CC), and Neutral Current (NC) interactions from dual 2D detector energy deposition projections.
2. **Physics-Grounded Explanations**: Generating verifiable physical rationales (vertex activity, electromagnetic shower morphology, track length/straightness) *before* arriving at the event classification.
3. **Human-in-the-Loop Error Mining & Recovery**: Identifying failure modes of baseline classifiers and evaluating whether explanation-conditioned models resolve ambiguous and difficult topologies.
4. **Attention Saliency Alignment**: Extracting cross-attention maps connecting textual tokens with spatial detector energy depositions.

---

## Supported Architectures

The repository is modularly organized into model family experiments:

- **`LLamaExp/`**: Meta **LLaMA-3.2-11B-Vision-Instruct** with 4-bit QLoRA, token-weighted multi-task loss, cross-attention saliency extraction, and hard-sample mining.
- **`GemmaExp/`**: Google **Gemma Vision** models (Gemma-4-E4B, Gemma-4-12B, Gemma-4-27B) with unified QLoRA fine-tuning and evaluation pipelines.
- **`QwenExp/`**: Scaffolded for upcoming **Qwen-VL** (Qwen2-VL / Qwen2.5-VL) model integration.

---

## Repository Structure

```
Neutrino-VLM-HITL/
├── README.md                      # Primary repository documentation
├── requirements.txt               # Environment dependencies
├── .gitignore                     # Git ignore rules for checkpoints, data, logs
├── GemmaExp/                      # Google Gemma Vision Experiments
│   ├── README.md                  # Gemma experiment documentation
│   ├── dataset.py                 # LArTPC dataset loader & Gemma prompt formatting
│   ├── train_gemma.py             # Parameterized QLoRA fine-tuning (E4B, 4b, 12b, 27b)
│   ├── evaluate_gemma.py          # Autoregressive test evaluation & confusion matrix
│   ├── compute_metrics.py         # Accuracy, precision, recall, macro ROC-AUC calculator
│   └── inference_gemma.py         # Interactive single-event inference demo
├── LLamaExp/                      # Meta LLaMA-3.2 Vision Experiments
│   ├── README.md                  # LLaMA experiment documentation
│   ├── dataset.py                 # LArTPC dataset loader & LLaMA chat templating
│   ├── train_llama.py             # QLoRA fine-tuning with token-weighted multi-task loss
│   ├── evaluate_llama.py          # Autoregressive test evaluation & confusion matrix
│   ├── generate_explanations.py   # Deep reasoning generation with token floor constraints
│   ├── generate_attention_map.py  # Cross-attention saliency extraction on zx and zy views
│   ├── run_inference.py           # Interactive demo with structured chain-of-thought prompts
│   ├── accelerate_config.yaml     # Accelerate distributed training configuration
│   └── train.sh                   # Launch script for multi-GPU training
└── QwenExp/                       # Reserved for Qwen-VL integration
    └── .gitkeep
```

---

## Physics & Data Representation

Each neutrino interaction is recorded across two 2D projected pixel maps:
1. **$zx$-Plane View**: Projection along the beam axis ($z$).
2. **$zy$-Plane View**: Projection along the drift axis ($y$).

### Event Classification Targets

| Interaction Class | Characteristic Topological Signature |
| :--- | :--- |
| **$\nu_e$ CC** | Primary electron-initiated electromagnetic shower near the interaction vertex (diffuse, branching, spatial extension). |
| **$\nu_\mu$ CC** | Primary minimum-ionizing muon candidate with a continuous, relatively straight track extending through the detector. |
| **Neutral Current (NC)** | Absence of a primary electron shower or muon track; hadronic clusters, proton stubs, or detached photon showers. |

### 16-Bit TIFF Preprocessing Pipeline

LArTPC energy depositions possess high dynamic range spanning minimum-ionizing particles to dense hadronic Bragg peaks. The preprocessing pipeline applies:
1. **Noise Thresholding & Clamping**: Energy depositions $< 0.25\text{ MeV}$ are set to zero; depositions $> 30.0\text{ MeV}$ are clamped.
2. **Logarithmic Compression**: $\log(1 + x)$ compression scales high-energy deposits smoothly without discarding faint track tails.
3. **Min-Max Normalization**: Rescaling $\log(1 + x) \in [0, \log(1 + 30)]$ to $[0.0, 1.0]$.
4. **8-Bit Conversion**: Conversion to 3-channel RGB PIL Images compatible with pre-trained visual backbones.

### Preventing Teacher-Forcing Data Leakage

During fine-tuning with explanations, ground-truth descriptions might contain mentions of the interaction type. The `dataset.py` module applies single-pass regex scrubbing to replace explicit mentions with standardized terms before teacher forcing, preventing the model from trivial label memorization.

---

## Installation

```bash
git clone https://github.com/dikshantsagar/Neutrino-VLM-HITL.git
cd Neutrino-VLM-HITL

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quickstart

### 1. Fine-Tuning LLaMA-3.2 Vision (11B)

```bash
cd LLamaExp

# Single GPU training with token-weighted loss on labels and human descriptions
python train_llama.py \
    --epochs 1 \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4 \
    --lora_r 8 \
    --label_weight 10.0 \
    --desc_weight 5.0

# Or distributed multi-GPU training with Accelerate
bash train.sh --epochs 1 --batch_size 4 --grad_accum 4
```

### 2. Fine-Tuning Gemma Vision

```bash
cd GemmaExp

# Train Gemma-4-E4B
python train_gemma.py --model_size E4B --epochs 1 --lr 2e-4 --batch_size 4

# Train Gemma-4-12B
python train_gemma.py --model_size 12b --batch_size 2 --grad_accum 8
```

---

## Human-in-the-Loop & Interpretability Workflows

### 1. Cross-Attention Saliency Extraction

Visualize where the model's visual cross-attention focuses on both the $zx$ and $zy$ projections during generation:

```bash
cd LLamaExp

python generate_attention_map.py \
    --adapter_path ./adapters/multitask_solution2_r-8 \
    --output_dir ./attention_outputs \
    --num_events 4
```

### 2. Structured Chain-of-Thought Inference

Prompt the model with explicit physics decomposition criteria (vertex recoil, candidate muon track, shower extent, topological conclusion):

```bash
cd LLamaExp

python run_inference.py \
    --adapter_path ./adapters/multitask_solution2_r-8 \
    --structured
```

---

## Evaluation & Metrics

Evaluate model checkpoints across full test sets and produce confusion matrices and comprehensive performance metrics:

```bash
# Evaluate LLaMA-3.2
cd LLamaExp
python evaluate_llama.py --adapter_path ./adapters/multitask_solution2_r-8 --output_dir ./predictions

# Evaluate Gemma
cd GemmaExp
python evaluate_gemma.py --model_size E4B --adapter_path ./adapters/gemma_4_E4B_r-8_exp_first

# Compute Accuracy, Macro Precision, Recall, and Macro One-vs-Rest ROC-AUC
python compute_metrics.py --dir ./predictions
```

---

## Future Roadmap

- [ ] **QwenExp**: Fine-tuning and benchmark comparison using Qwen2-VL and Qwen2.5-VL.
- [ ] **Counterfactual Track Removal**: Ablation tests removing primary candidate tracks to verify causal explanation sensitivity.
- [ ] **Intermediate Concept Heads**: Explicit multi-task heads for vertex coordinate regression, prong counts, and shower opening angles.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
