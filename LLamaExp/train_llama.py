"""
LLaMA-3.2 Vision LoRA Fine-Tuning for LArTPC Neutrino Interactions
==================================================================
Unified fine-tuning script supporting multi-task explanation and classification,
label-first / explanation-first supervision, warm-starting from Stage 1 adapters,
and token-weighted loss optimization for LLaMA-3.2-11B-Vision-Instruct.
"""

import argparse
import glob
import os
import re
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
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
        if ignore_keys is None:
            ignore_keys = []
        if "past_key_values" not in ignore_keys:
            ignore_keys = list(ignore_keys) + ["past_key_values"]

        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )

        if isinstance(logits, tuple):
            logits = logits[0]
        if labels is None and "labels" in inputs:
            labels = inputs["labels"]
        return loss, logits, labels


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA-3.2 Vision on Neutrino Events")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Glob pattern for dataset directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save LoRA adapters")
    parser.add_argument("--from_stage1", action="store_true",
                        help="Warm-start training from a Stage 1 classification adapter")
    parser.add_argument("--stage1_adapter", type=str, default=None,
                        help="Path to Stage 1 adapter checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Per-device eval batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank r")
    parser.add_argument("--oversample_desc", type=int, default=1,
                        help="Oversampling multiplier for events with real human descriptions")
    parser.add_argument("--label_weight", type=float, default=10.0,
                        help="Loss weight multiplier for classification <label> tokens")
    parser.add_argument("--desc_weight", type=float, default=5.0,
                        help="Loss weight multiplier for real <explanation> tokens")
    parser.add_argument("--without_explanations", action="store_true",
                        help="Train purely on labels without explanations")
    parser.add_argument("--wandb_project", type=str, default="NeutrinoVLM", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Custom wandb run name")
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
            print(f"Found {len(matched)} events using data path: {pattern}")
            return matched
    raise FileNotFoundError("Could not find dataset. Please provide a valid --data_path.")


def main():
    args = parse_args()

    mode_name = "no_exp" if args.without_explanations else "exp_first"
    run_name = args.wandb_name or f"llama3.2v11b_r-{args.lora_r}_{mode_name}"
    output_dir = args.output_dir or f"./adapters/{run_name}"

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_NAME"] = run_name
    os.environ["WANDB_WATCH"] = "false"

    print("=" * 60)
    print(f"Model ID       : {args.model_id}")
    print(f"Run Name       : {run_name}")
    print(f"Output Dir     : {output_dir}")
    print(f"Explanations   : {not args.without_explanations}")
    print(f"Stage 1 Warmup : {args.from_stage1}")
    print("=" * 60)

    # 1. Dataset loading and split
    all_paths = resolve_data_paths(args.data_path)
    train_paths, test_paths = train_test_split(all_paths, test_size=0.05, random_state=77, shuffle=True)

    with_exp = not args.without_explanations
    train_dataset = NeutrinoDataset(
        train_paths,
        classes=DEFAULT_CLASSES,
        explainations=with_exp,
        require_desc=False,
        oversample_desc=args.oversample_desc
    )
    test_dataset = NeutrinoDataset(
        test_paths[:200],
        classes=DEFAULT_CLASSES,
        explainations=with_exp,
        require_desc=False
    )
    print(f"Train dataset size: {len(train_dataset)} | Fast Eval dataset size: {len(test_dataset)}")

    # 2. BitsAndBytes 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    local_rank = os.environ.get("LOCAL_RANK")
    device_map = {"": int(local_rank)} if local_rank is not None else "auto"

    model = MllamaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=bnb_config
    )
    model.config.keys_to_ignore_at_inference = ["past_key_values"]
    processor = AutoProcessor.from_pretrained(args.model_id)

    # 3. LoRA or Stage 1 warm-start
    if args.from_stage1 and args.stage1_adapter and os.path.exists(args.stage1_adapter):
        print(f"Loading Stage 1 classification adapter from: {args.stage1_adapter}")
        model = PeftModel.from_pretrained(model, args.stage1_adapter, is_trainable=True)
        peft_config_to_pass = None
    else:
        print("Initializing fresh LoRA config on base model...")
        r = args.lora_r
        peft_config = LoraConfig(
            lora_alpha=2 * r,
            lora_dropout=0.05,
            r=r,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        peft_config_to_pass = peft_config

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
        save_only_model=True,
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

        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        for img_tok in image_tokens:
            labels[labels == img_tok] = -100

        offset_mapping = batch.pop("offset_mapping")
        assistant_header_pattern = re.compile(r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*")
        label_pattern = re.compile(r"<label>.*?</label>", re.DOTALL)
        explanation_pattern = re.compile(r"<explanation>.*?</explanation>", re.DOTALL)

        # Mask prompt tokens prior to assistant turn
        for i in range(len(examples)):
            match = assistant_header_pattern.search(texts[i])
            if match:
                start_char = match.end()
                offsets = offset_mapping[i]
                labels[i, offsets[:, 0] < start_char] = -100

        loss_weights = torch.ones_like(labels, dtype=torch.float32)

        for i, example in enumerate(examples):
            text = texts[i]
            match_header = assistant_header_pattern.search(text)
            if not match_header:
                continue

            assistant_start_idx = match_header.end()
            assistant_text = text[assistant_start_idx:]

            # Upweight <label> section
            match_lbl = label_pattern.search(assistant_text)
            if match_lbl:
                start_char = assistant_start_idx + match_lbl.start()
                end_char = assistant_start_idx + match_lbl.end()
                offsets = offset_mapping[i]
                mask = (offsets[:, 0] >= start_char) & (offsets[:, 0] < end_char) & (offsets[:, 1] > 0)
                loss_weights[i, mask] = args.label_weight

            # Upweight real verified <explanation> section
            if example.get("has_real_desc", False):
                match_exp = explanation_pattern.search(assistant_text)
                if match_exp:
                    start_char = assistant_start_idx + match_exp.start()
                    end_char = assistant_start_idx + match_exp.end()
                    offsets = offset_mapping[i]
                    mask = (offsets[:, 0] >= start_char) & (offsets[:, 0] < end_char) & (offsets[:, 1] > 0)
                    loss_weights[i, mask] = args.desc_weight

        batch["labels"] = labels
        batch["loss_weights"] = loss_weights
        return batch

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        shifted_preds = np.zeros_like(preds)
        shifted_preds[:, 1:] = preds[:, :-1]

        preds = np.where(labels != -100, shifted_preds, processor.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

        decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

        correct, total = 0, 0
        lbl_re = re.compile(r"<label>\s*(.*?)\s*</label>", re.DOTALL)
        for p, l in zip(decoded_preds, decoded_labels):
            p_match, l_match = lbl_re.search(p), lbl_re.search(l)
            if l_match:
                total += 1
                if p_match and p_match.group(1).strip() == l_match.group(1).strip():
                    correct += 1
        return {"accuracy": correct / total if total > 0 else 0.0}

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        peft_config=peft_config_to_pass,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
