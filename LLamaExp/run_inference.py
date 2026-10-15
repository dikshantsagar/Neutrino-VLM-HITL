"""
LLaMA-3.2 Vision Inference Demo for LArTPC Neutrino Interactions
================================================================
Runs single-event or random-sample inference, supporting both standard
and structured chain-of-thought physics prompts (vertex, track, shower analysis).
"""

import argparse
import glob
import os
import random
import torch
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from sklearn.model_selection import train_test_split

from dataset import NeutrinoDataset, DEFAULT_CLASSES

STRUCTURED_USER_PROMPT = """Classify the neutrino event as ν_e CC, ν_μ CC, or Neutral Current and provide a detailed physical explanation of your reasoning.
In your explanation, thoroughly analyze both the zx and zy projections:
1. Vertex activity: location of the interaction vertex and any nuclear recoil or short stubs.
2. Track analysis: presence, length, straightness, and continuity of candidate muon or hadronic tracks.
3. Shower analysis: presence, spatial extent, fuzziness, or absence of electromagnetic showers and conversion gaps.
4. Topological conclusion: how the combined visual evidence identifies the event class.

Give your answer in the following format:
<explanation>
{{REASONING}}
</explanation>

<label>
{{LABEL}}
</label>
"""


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
    parser = argparse.ArgumentParser(description="Run Inference with LLaMA-3.2 Vision")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter checkpoint")
    parser.add_argument("--event_path", type=str, default=None,
                        help="Path to a specific event directory containing xz.tiff and yz.tiff")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Glob pattern for dataset to sample random event from")
    parser.add_argument("--structured", action="store_true",
                        help="Use structured chain-of-thought prompt (vertex, track, shower)")
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

    # 1. Select event
    if args.event_path:
        event_paths = [args.event_path]
    else:
        all_paths = resolve_data_paths(args.data_path)
        _, test_paths = train_test_split(all_paths, test_size=0.05, random_state=77, shuffle=True)
        event_paths = [random.choice(test_paths)]

    dataset = NeutrinoDataset(event_paths, classes=DEFAULT_CLASSES, explainations=True)
    sample = dataset[0]
    messages = sample["messages"]

    print("=" * 60)
    print("EVENT SELECTED:", event_paths[0])
    print("=" * 60)
    print("GROUND TRUTH / REFERENCE:")
    print(messages[2]["content"][0]["text"])
    print("=" * 60)

    # 2. Model Loading
    print(f"Loading Base Model: {args.model_id}...")
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

    if args.adapter_path and os.path.exists(args.adapter_path):
        print(f"Applying Adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_id)

    # 3. Prompting & Generation
    if args.structured:
        user_content = [
            messages[1]["content"][0],
            messages[1]["content"][1],
            {"type": "text", "text": STRUCTURED_USER_PROMPT}
        ]
        inference_messages = [messages[0], {"role": "user", "content": user_content}]
    else:
        inference_messages = messages[:2]

    prompt_text = processor.apply_chat_template(inference_messages, add_generation_prompt=True)
    prompt_text += "<explanation>\n"

    image_inputs = process_vision_info(inference_messages)[0]
    inputs = processor(
        text=[prompt_text],
        images=[image_inputs],
        return_tensors="pt"
    ).to(model.device)

    stopping_criteria = StoppingCriteriaList([StopOnString("</label>", processor.tokenizer, min_new_tokens=40)])

    print("\nGenerating model explanation & classification...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            stopping_criteria=stopping_criteria
        )

    response = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\n" + "=" * 60)
    print("MODEL OUTPUT:")
    print("=" * 60)
    print("<explanation>\n" + response.strip())


if __name__ == "__main__":
    main()
