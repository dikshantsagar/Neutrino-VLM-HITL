"""
LLaMA-3.2 Vision Evaluation Script for LArTPC Neutrino Interactions
==================================================================
Performs autoregressive inference on the test split, parses structured
<explanation> and <label> outputs, records predictions,
and generates confusion matrices and evaluation metrics.
"""

import argparse
import glob
import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from dataset import NeutrinoDataset, DEFAULT_CLASSES


class StopOnString(StoppingCriteria):
    def __init__(self, stop_str, tokenizer):
        self.stop_str = stop_str
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        last_tokens = input_ids[0][-10:]
        decoded = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
        return self.stop_str in decoded


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLaMA-3.2 Vision on Neutrino Test Set")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter checkpoint")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Glob pattern for dataset directory")
    parser.add_argument("--output_dir", type=str, default="./predictions",
                        help="Directory to save predictions and confusion matrix")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum test samples to evaluate")
    parser.add_argument("--without_explanations", action="store_true",
                        help="Prompt for label only without explanation")
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
    raise FileNotFoundError("Could not find dataset. Please provide a valid --data_path.")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Model ID     : {args.model_id}")
    print(f"Adapter Path : {args.adapter_path or 'Base model (no adapter)'}")
    print(f"Output Dir   : {args.output_dir}")
    print("=" * 60)

    # 1. Dataset Setup
    all_paths = resolve_data_paths(args.data_path)
    _, test_paths = train_test_split(all_paths, test_size=0.05, random_state=77, shuffle=True)
    if args.max_samples:
        test_paths = test_paths[:args.max_samples]

    with_exp = not args.without_explanations
    test_dataset = NeutrinoDataset(test_paths, classes=DEFAULT_CLASSES, explainations=with_exp)
    print(f"Evaluating {len(test_dataset)} test events...")

    # 2. Model Setup
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = MllamaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if args.adapter_path and os.path.exists(args.adapter_path):
        print(f"Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_id)
    stopping_criteria = StoppingCriteriaList([StopOnString("</label>", processor.tokenizer)])
    label_pattern = re.compile(r"<label>\s*(.*?)\s*</label>", re.DOTALL)
    labels_list = ["ν_e CC", "ν_μ CC", "Neutral Current"]

    y_true = []
    y_pred = []
    predictions_file = os.path.join(args.output_dir, "predictions.txt")

    with open(predictions_file, "w", encoding="utf-8") as f_out:
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            sample = test_dataset[idx]
            messages = sample["messages"]
            inference_messages = messages[:2]

            gt_text = messages[2]["content"][0]["text"]
            gt_match = label_pattern.search(gt_text)
            gt_label = gt_match.group(1).strip() if gt_match else "Unknown"

            prompt_text = processor.apply_chat_template(inference_messages, add_generation_prompt=True)
            if with_exp:
                prompt_text += "<explanation>\n"

            image_inputs = process_vision_info(inference_messages)[0]
            inputs = processor(
                text=[prompt_text],
                images=[image_inputs],
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    stopping_criteria=stopping_criteria,
                    do_sample=False,
                )

            gen_text = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            pred_match = label_pattern.search(gen_text)
            pred_label = pred_match.group(1).strip() if pred_match else "Unknown"

            y_true.append(gt_label)
            y_pred.append(pred_label)

            f_out.write(f"Sample {idx} | GT: {gt_label} | Pred: {pred_label}\n")

    # Metrics
    valid_mask = [y in labels_list for y in y_true]
    y_true_clean = [y_true[i] for i in range(len(y_true)) if valid_mask[i]]
    y_pred_clean = [y_pred[i] for i in range(len(y_pred)) if valid_mask[i]]

    acc = accuracy_score(y_true_clean, y_pred_clean)
    print("\n" + "=" * 60)
    print(f"Overall Accuracy: {acc:.4%}")
    print("=" * 60)
    print(classification_report(y_true_clean, y_pred_clean, target_names=labels_list, zero_division=0))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=labels_list)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_list, yticklabels=labels_list)
    plt.title(f"Confusion Matrix: LLaMA-3.2-Vision (Acc: {acc:.2%})")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    print(f"Saved results and confusion matrix to: {args.output_dir}")


if __name__ == "__main__":
    main()
