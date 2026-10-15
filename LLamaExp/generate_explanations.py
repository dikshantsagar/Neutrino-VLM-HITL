"""
Generate Physics Explanations for Neutrino Interactions
========================================================
Runs autoregressive explanation generation on benchmark events,
enforcing minimum token limits to ensure deep physical reasoning
(vertex characteristics, track topologies, electromagnetic showers).
"""

import argparse
import os
import random
import re
import torch
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
from sklearn.model_selection import train_test_split

from dataset import NeutrinoDataset, DEFAULT_CLASSES


class StopOnString(StoppingCriteria):
    def __init__(self, stop_str, tokenizer, min_new_tokens=0):
        self.stop_str = stop_str
        self.tokenizer = tokenizer
        self.min_new_tokens = min_new_tokens
        self.start_len = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.start_len is None:
            self.start_len = input_ids.shape[1]
        if (input_ids.shape[1] - self.start_len) < self.min_new_tokens:
            return False
        last_tokens = input_ids[0][-15:]
        decoded = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
        return self.stop_str in decoded


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Neutrino Event Explanations")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to explanation adapter checkpoint")
    parser.add_argument("--paths_file", type=str, default=None,
                        help="Optional file listing specific event paths to evaluate")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Glob pattern for dataset directory")
    parser.add_argument("--output_file", type=str, default="./generated_explanations.md",
                        help="Output markdown file for generated explanations")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to generate explanations for")
    parser.add_argument("--min_new_tokens", type=int, default=60,
                        help="Minimum explanation length before allowing closing tag")
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
    import glob
    for pattern in candidates:
        matched = glob.glob(pattern)
        if matched:
            return matched
    raise FileNotFoundError("Could not find dataset. Please provide a valid --data_path or --paths_file.")


def main():
    args = parse_args()

    if args.paths_file and os.path.exists(args.paths_file):
        with open(args.paths_file, "r", encoding="utf-8") as f:
            test_paths = [x.strip() for x in f if x.strip() and os.path.exists(x.strip())]
    else:
        all_paths = resolve_data_paths(args.data_path)
        _, test_paths = train_test_split(all_paths, test_size=0.05, random_state=77, shuffle=True)

    random.seed(42)
    selected_paths = test_paths[:args.num_samples]
    dataset = NeutrinoDataset(selected_paths, classes=DEFAULT_CLASSES, explainations=True)

    print(f"Generating explanations for {len(dataset)} events using {args.adapter_path}...")

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
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_id)

    stop_explanation = StoppingCriteriaList([
        StopOnString("</explanation>", processor.tokenizer, min_new_tokens=args.min_new_tokens)
    ])
    stop_label = StoppingCriteriaList([
        StopOnString("</label>", processor.tokenizer, min_new_tokens=0)
    ])

    exp_re = re.compile(r"<explanation>\s*(.*?)(?:</explanation>|$)", re.DOTALL)
    lbl_re = re.compile(r"<label>\s*(.*?)(?:</label>|$)", re.DOTALL)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as md:
        md.write("# Generated Neutrino Event Explanations\n\n")
        md.write(f"**Model**: `{args.model_id}` | **Adapter**: `{args.adapter_path}`\n\n---\n\n")

        for idx in tqdm(range(len(dataset)), desc="Generating Explanations"):
            sample = dataset[idx]
            messages = sample["messages"]
            path = dataset.paths[idx]

            # 1. Generate Explanation
            inf_msgs = messages[:2]
            prompt = processor.apply_chat_template(inf_msgs, add_generation_prompt=True) + "<explanation>\n"
            img_info = process_vision_info(inf_msgs)[0]
            inputs = processor(text=[prompt], images=[img_info], return_tensors="pt").to(model.device)

            with torch.no_grad():
                out_exp = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    stopping_criteria=stop_explanation,
                )
            gen_exp_raw = processor.decode(out_exp[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            match_exp = exp_re.search(gen_exp_raw)
            gen_explanation = match_exp.group(1).strip() if match_exp else gen_exp_raw.strip()

            # 2. Complete Generation for Label
            prompt_full = prompt + gen_explanation + "\n</explanation>\n\n<label>\n"
            inputs_full = processor(text=[prompt_full], images=[img_info], return_tensors="pt").to(model.device)

            with torch.no_grad():
                out_lbl = model.generate(
                    **inputs_full,
                    max_new_tokens=20,
                    do_sample=False,
                    stopping_criteria=stop_label,
                )
            gen_lbl_raw = processor.decode(out_lbl[0][inputs_full.input_ids.shape[1]:], skip_special_tokens=True)
            match_lbl = lbl_re.search(gen_lbl_raw)
            gen_label = match_lbl.group(1).strip() if match_lbl else gen_lbl_raw.strip()

            # Ground Truth
            gt_text = messages[2]["content"][0]["text"]
            gt_match = lbl_re.search(gt_text)
            gt_label = gt_match.group(1).strip() if gt_match else "Unknown"

            md.write(f"### Event {idx + 1} | Path: `{os.path.basename(path)}`\n\n")
            md.write(f"- **Ground Truth**: `{gt_label}`\n")
            md.write(f"- **Predicted Label**: `{gen_label}` {'✅' if gen_label == gt_label else '❌'}\n\n")
            md.write(f"**Explanation:**\n> {gen_explanation}\n\n---\n\n")
            md.flush()

    print(f"Saved generated explanations report to: {args.output_file}")


if __name__ == "__main__":
    main()
