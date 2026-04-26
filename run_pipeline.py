import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def step_collect():
    print("\n" + "="*50)
    print("STEP 1: Collecting restaurant data from Google Places")
    print("="*50)
    from data_collection import collect_all_cities
    records = collect_all_cities(max_per_city=20)
    print(f"✅ Collected {len(records)} restaurants")


def step_index():
    print("\n" + "="*50)
    print("STEP 2: Building FAISS index")
    print("="*50)
    from build_index import build_index
    build_index()


def step_sft_data():
    print("\n" + "="*50)
    print("STEP 3: Generating SFT training data")
    print("="*50)
    from generate_sft_data import generate_sft_dataset
    data = generate_sft_dataset()
    print(f"✅ Generated {len(data)} SFT samples")


def step_sft_train():
    print("\n" + "="*50)
    print("STEP 4: SFT fine-tuning Qwen2.5-0.5B")
    print("="*50)
    from sft_train import train
    train()


def step_rlhf():
    print("\n" + "="*50)
    print("STEP 5: RLHF training with CLIP reward")
    print("="*50)
    from rlhf_train import train_rlhf
    train_rlhf(num_steps=100)


def step_eval():
    print("\n" + "="*50)
    print("STEP 6: Evaluation")
    print("="*50)
    from evaluation import run_full_evaluation
    summary = run_full_evaluation()
    print("\nEvaluation Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


STEPS = {
    "collect":   step_collect,
    "index":     step_index,
    "sft_data":  step_sft_data,
    "sft_train": step_sft_train,
    "rlhf":      step_rlhf,
    "eval":      step_eval,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VibeVenue Pipeline Runner")
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()) + ["all"],
        default="all",
        help="Which pipeline step to run"
    )
    args = parser.parse_args()

    if args.step == "all":
        for name, fn in STEPS.items():
            fn()
    else:
        STEPS[args.step]()
