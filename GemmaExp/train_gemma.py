"""
Gemma Vision LoRA Fine-Tuning for LArTPC Neutrino Event Classification
======================================================================
Unified script for parameter-efficient fine-tuning (QLoRA) across Gemma Vision
model sizes (e.g., E4B, 4b, 12b, 27b) with weighted loss and explanation supervision.
"""

import argparse
import glob
import os
import re
import gc
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import wandb
from qwen_vl_utils import process_vision_info

from dataset import NeutrinoDataset, DEFAULT_CLASSES


class WeightedSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer supporting sample- and token-level loss weighting
    to prioritize critical classification labels and verified human explanations.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_weights = inputs.pop("loss_weights", None)
        outputs = model(**inputs)

        logits = outputs.get("logits")
        labels = inputs.get("labels")

        if logits is not None and labels is not None and loss_weights is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weights[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, logits.size(-1))
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)

            loss = loss_fct(shift_logits, shift_labels)
            weighted_loss = loss * shift_weights

            valid_mask = shift_labels != -100
            num_valid = valid_mask.sum()
            if num_valid > 0:
                loss = weighted_loss.sum() / num_valid
            else:
                loss = weighted_loss.sum() * 0.0
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs.pop("loss_weights", None)
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        if labels is None and "labels" in inputs:
            labels = inputs["labels"]
        return loss, logits, labels


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma Vision model on Neutrino Events")
    parser.add_argument("--model_size", type=str, default="E4B", choices=["E4B", "4b", "12b", "27b"],
                        help="Size of Gemma model (default: E4B)")
    parser.add_argument("--model_id", type=str, default=None,
                        help="HuggingFace model ID (defaults to google/gemma-4-{model_size}-it)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Glob pattern for dataset directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save LoRA adapters")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Per-device eval batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--without_explanations", action="store_true",
                        help="Train purely on labels without explanations")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--wandb_project", type=str, default="NeutrinoVLM", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Custom wandb run name")
    return parser.parse_args()


def resolve_data_paths(user_path=None):
    if user_path:
        paths = glob.glob(user_path)
        if paths:
            return paths
    candidate_paths = [
        "/mnt/samsung_15t/users/dikshans/class_mask_dataset/asImages/*/*/*",
        "/home/dikshans/datasets/class_mask_dataset/asImages/*/*/*",
        "./data/*/*/*"
    ]
    for pattern in candidate_paths:
        matched = glob.glob(pattern)
        if matched:
            print(f"Found {len(matched)} events using data path: {pattern}")
            return matched
    raise FileNotFoundError("Could not find dataset. Please provide a valid --data_path.")


def main():
    args = parse_args()

    model_id = args.model_id or f"google/gemma-4-{args.model_size}-it"
    exp_suffix = "no_exp" if args.without_explanations else "exp_first"
    default_name = f"gemma_4_{args.model_size}_r-{args.lora_r}_{exp_suffix}"
    run_name = args.wandb_name or default_name
    output_dir = args.output_dir or f"./adapters/{run_name}"

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_NAME"] = run_name
    os.environ["WANDB_WATCH"] = "false"

    print("=" * 60)
    print(f"Model ID    : {model_id}")
    print(f"Run Name    : {run_name}")
    print(f"Output Dir  : {output_dir}")
    print(f"Explanations: {not args.without_explanations}")
    print("=" * 60)

    # 1. Dataset loading and split
    all_paths = resolve_data_paths(args.data_path)
    train_paths, test_paths = train_test_split(all_paths, test_size=0.05, random_state=77, shuffle=True)

    with_exp = not args.without_explanations
    train_dataset = NeutrinoDataset(train_paths, classes=DEFAULT_CLASSES, explainations=with_exp)
    eval_dataset = NeutrinoDataset(test_paths[:100], classes=DEFAULT_CLASSES, explainations=with_exp)
    print(f"Train samples: {len(train_dataset)} | Fast eval samples: {len(eval_dataset)}")

    # 2. BitsAndBytes 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    local_rank = os.environ.get("LOCAL_RANK")
    device_map = {"": int(local_rank)} if local_rank is not None else "auto"

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        r=args.lora_r,
        bias="none",
        target_modules=[
            "q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear",
            "gate_proj.linear", "up_proj.linear", "down_proj.linear"
        ],
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=1000,
        eval_on_start=False,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        learning_rate=args.lr,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        report_to="wandb",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        label_names=["labels"],
    )

    def collate_fn(examples):
        texts = [processor.apply_chat_template(ex["messages"], tokenize=False) for ex in examples]
        image_inputs = [process_vision_info(ex["messages"])[0] for ex in examples]

        batch = processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            return_offsets_mapping=True,
        )

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)] if hasattr(processor, "image_token") else []
        for img_tok in image_tokens:
            labels[labels == img_tok] = -100

        offset_mapping = batch.pop("offset_mapping")
        assistant_header_pattern = re.compile(r"<start_of_turn>model\s*")

        for i in range(len(examples)):
            match = assistant_header_pattern.search(texts[i])
            if match:
                start_char = match.end()
                offsets = offset_mapping[i]
                mask_indices = offsets[:, 0] < start_char
                labels[i, mask_indices] = -100

        loss_weights = torch.ones_like(labels, dtype=torch.float32)
        batch["labels"] = labels
        batch["loss_weights"] = loss_weights
        return batch

    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
