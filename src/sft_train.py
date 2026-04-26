import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_DATA_DIR = Path("data/sft_dataset")
OUTPUT_DIR   = Path("models/sft_qwen")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH   = 256
BATCH_SIZE   = 4
EPOCHS       = 3
LR           = 2e-4


@dataclass
class SFTConfig:
    model_name: str  = MODEL_NAME
    max_length: int  = MAX_LENGTH
    batch_size: int  = BATCH_SIZE
    epochs: int      = EPOCHS
    lr: float        = LR


def load_sft_data(split: str) -> Dataset:
    path = SFT_DATA_DIR / f"sft_{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def format_prompt(instruction: str, output: str = "") -> str:
    prompt = (
        f"<|im_start|>system\n"
        f"You are a travel and dining expert specializing in Asian restaurant atmospheres.<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}"
    )
    return prompt


def tokenize_fn(examples: dict, tokenizer) -> dict:
    prompts = [
        format_prompt(inst, out)
        for inst, out in zip(examples["instruction"], examples["output"])
    ]
    tokenized = tokenizer(
        prompts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def train():
    cfg = SFTConfig()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training device: {device}")

    print(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = load_sft_data("train")
    val_ds   = load_sft_data("val")

    train_ds = train_ds.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("Starting SFT training...")
    train_result = trainer.train()

    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    metrics = train_result.metrics
    with open(OUTPUT_DIR / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ SFT training complete → {OUTPUT_DIR}/final")
    print(f"Metrics: {metrics}")


def generate_description(image_vibe_label: str, model_path: str = None) -> str:
    from peft import PeftModel

    path = model_path or str(OUTPUT_DIR / "final")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, path)
    model.eval()

    instruction = "Describe the scenery and atmosphere visible in this restaurant or cafe photo in one vivid sentence."
    prompt = format_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "<|im_start|>assistant" in decoded:
        response = decoded.split("<|im_start|>assistant")[-1].strip()
    else:
        response = decoded.split(instruction)[-1].strip()
    return response


if __name__ == "__main__":
    train()
