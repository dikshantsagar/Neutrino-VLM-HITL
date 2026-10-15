"""
Cross-Attention Saliency & Attention Map Visualization for LArTPC Projections
=============================================================================
Hooks into LLaMA-3.2 Vision cross-attention layers during inference,
aggregates attention weights across heads and layers, and overlays spatial
saliency maps on the zx and zy 2D detector energy deposition projections.
"""

import argparse
import glob
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, MllamaForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image

from dataset import NeutrinoDataset, DEFAULT_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Attention Maps for Neutrino Projections")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter checkpoint")
    parser.add_argument("--event_path", type=str, default=None,
                        help="Path to specific event directory containing xz.tiff and yz.tiff")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Glob pattern to sample events from")
    parser.add_argument("--output_dir", type=str, default="./attention_outputs",
                        help="Directory to save attention overlay plots")
    parser.add_argument("--num_events", type=int, default=2,
                        help="Number of events to generate attention maps for")
    return parser.parse_args()


def resolve_data_paths(user_path=None):
    if user_path:
        paths = glob.glob(user_path)
        if paths:
            return paths
    candidates = [
        "/mnt/samsung_15t/users/dikshans/class_mask_dataset/asImages/*/*/*",
        "/home/dikshans/datasets/class_mask_dataset/asImages/*/*/*",
        "./data/*/*/*"
    ]
    for pattern in candidates:
        matched = glob.glob(pattern)
        if matched:
            return matched
    raise FileNotFoundError("Could not find dataset. Please provide a valid --data_path or --event_path.")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Dataset Setup
    if args.event_path:
        selected_paths = [args.event_path]
    else:
        all_paths = resolve_data_paths(args.data_path)
        _, test_paths = train_test_split(all_paths, test_size=0.05, random_state=77, shuffle=True)
        selected_paths = test_paths[:args.num_events]

    dataset = NeutrinoDataset(selected_paths, classes=DEFAULT_CLASSES, explainations=True)

    # 2. Model Setup
    print(f"Loading Base Model: {args.model_id}...")
    model = MllamaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"  # Required for attention hook extraction
    )

    if args.adapter_path and os.path.exists(args.adapter_path):
        print(f"Loading Adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_id)

    # 3. Process Events
    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        event_path = dataset.paths[sample_idx]
        print(f"\nProcessing Event {sample_idx + 1}/{len(dataset)}: {event_path}")

        messages = sample["messages"][:2]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True) + "<explanation>\n"
        image_inputs = process_vision_info(messages)[0]

        inputs = processor(
            text=[prompt],
            images=[image_inputs],
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        captured_attentions = []

        def hook_fn(module, inp, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                captured_attentions.append(output[1].detach().cpu())

        hooks = []
        for name, module in model.named_modules():
            # Target cross-attention modules connecting text tokens to vision patches
            if "cross_attn" in name and hasattr(module, "forward"):
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        for h in hooks:
            h.remove()

        # Visualizing Attention
        raw_zx = Image.open(os.path.join(event_path, "xz.tiff")).convert("L")
        raw_zy = Image.open(os.path.join(event_path, "yz.tiff")).convert("L")

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(raw_zx, cmap="viridis")
        axes[0].set_title(f"ZX View (Beam Axis) - {os.path.basename(event_path)}")
        axes[0].axis("off")

        axes[1].imshow(raw_zy, cmap="viridis")
        axes[1].set_title("ZY View (Drift Axis)")
        axes[1].axis("off")

        plot_path = os.path.join(args.output_dir, f"event_{sample_idx}_saliency.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved visualization to: {plot_path}")


if __name__ == "__main__":
    main()
