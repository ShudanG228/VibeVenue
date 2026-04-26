import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from clip_encoder import encode_image, encode_text, cosine_similarity, get_device

SFT_MODEL_DIR = Path("models/sft_qwen/final")
RLHF_OUTPUT   = Path("models/rlhf_qwen")
SFT_DATA_PATH = Path("data/sft_dataset/sft_train.json")
RLHF_OUTPUT.mkdir(parents=True, exist_ok=True)
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = get_device()


def clip_reward(generated_text: str, image_path: str) -> float:
    try:
        text_emb  = encode_text(generated_text)
        image_emb = encode_image(image_path)
        reward = cosine_similarity(text_emb, image_emb)
        return float((reward + 1.0) / 2.0)
    except Exception:
        return 0.0


def format_query(instruction: str) -> str:
    return (
        f"<|im_start|>system\nYou are a travel expert specializing in Asian restaurants.<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def train_rlhf(num_steps: int = 100):
    print(f"[RLHF] Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(str(SFT_MODEL_DIR), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    with open(SFT_DATA_PATH) as f:
        raw_data = json.load(f)
    raw_data = [d for d in raw_data if Path(d["image_path"]).exists()][:500]
    print(f"[RLHF] {len(raw_data)} samples loaded")

    print("[RLHF] Loading model...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        str(SFT_MODEL_DIR),
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    ppo_config = PPOConfig(
        model_name=str(SFT_MODEL_DIR),
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
        gradient_accumulation_steps=2,
        ppo_epochs=1,
        seed=42,
        log_with=None,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    gen_kwargs = {
        "max_new_tokens": 60,
        "do_sample": True,
        "temperature": 0.5,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
    }

    reward_log = []
    instruction = "Describe the scenery and atmosphere of this Asian restaurant in one vivid sentence."

    print("[RLHF] Starting PPO training...")
    for step in tqdm(range(num_steps)):

        indices = np.random.choice(len(raw_data), size=2, replace=False)
        batch_data = [raw_data[i] for i in indices]
        image_paths = [d["image_path"] for d in batch_data]

        queries = [format_query(instruction)] * 2
        query_tensors = [
            tokenizer(q, return_tensors="pt", truncation=True, max_length=200)["input_ids"].squeeze(0)
            for q in queries
        ]

        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **gen_kwargs)

        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        rewards = [
            torch.tensor(clip_reward(resp, img), dtype=torch.float32)
            for resp, img in zip(responses, image_paths)
        ]

        mean_reward = float(np.mean([r.item() for r in rewards]))
        reward_log.append({"step": step, "mean_reward": mean_reward})

        if step % 10 == 0:
            print(f"  Step {step:3d} | reward: {mean_reward:.4f} | {responses[0][:80]}...")

        ppo_trainer.step(query_tensors, response_tensors, rewards)

    ppo_trainer.save_pretrained(str(RLHF_OUTPUT / "final"))
    tokenizer.save_pretrained(str(RLHF_OUTPUT / "final"))

    with open(RLHF_OUTPUT / "reward_log.json", "w") as f:
        json.dump(reward_log, f, indent=2)

    print(f"\n✅ RLHF complete → {RLHF_OUTPUT}/final")
    if reward_log:
        print(f"   Initial reward: {reward_log[0]['mean_reward']:.4f}")
        print(f"   Final reward:   {reward_log[-1]['mean_reward']:.4f}")


if __name__ == "__main__":
    train_rlhf(num_steps=300)
