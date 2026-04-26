import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from clip_encoder import encode_image, cosine_similarity
from inference import search

RESULTS_DIR = Path("evaluation_results")
RESULTS_DIR.mkdir(exist_ok=True)


def clip_similarity_score(query_path: str, result_photo_path: str) -> float:
    q_emb = encode_image(query_path)
    r_emb = encode_image(result_photo_path)
    return cosine_similarity(q_emb, r_emb)


def recall_at_k(results, expected_country: str, k: int = 5) -> float:
    """
    Checks if the expected country appears in top-K results.
    Used as a proxy for relevance since we don't have explicit ground truth.
    """
    top_k = results[:k]
    return float(any(r.country == expected_country for r in top_k))


def mean_reciprocal_rank(results_list: list, expected_countries: list[str]) -> float:
    rrs = []
    for results, country in zip(results_list, expected_countries):
        for rank, r in enumerate(results, 1):
            if r.country == country:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    return float(np.mean(rrs))


def benchmark_inference(photo_paths: list[str], n_runs: int = 10) -> dict:
    times = []
    for path in photo_paths[:n_runs]:
        t0 = time.time()
        search(path, top_k=10, use_llm=False)
        times.append((time.time() - t0) * 1000)

    return {
        "mean_ms":   float(np.mean(times)),
        "std_ms":    float(np.std(times)),
        "min_ms":    float(np.min(times)),
        "max_ms":    float(np.max(times)),
        "throughput_qps": float(1000 / np.mean(times)),
    }


def analyze_failures(photo_paths: list[str], expected_countries: list[str], top_k: int = 5) -> dict:
    """
    Run retrieval on a set of query photos and identify failure cases.
    A failure = expected country not in top-K results.
    """
    failures = []
    successes = []

    for path, country in tqdm(zip(photo_paths, expected_countries), total=len(photo_paths)):
        results, desc, ms = search(path, top_k=top_k, use_llm=False)
        hit = recall_at_k(results, country, k=top_k)

        record = {
            "query_path":       path,
            "expected_country": country,
            "vibe_desc":        desc,
            "hit":              bool(hit),
            "top_results":      [(r.name, r.country, r.image_score) for r in results[:3]],
        }
        (successes if hit else failures).append(record)

    return {
        "n_total":    len(photo_paths),
        "n_success":  len(successes),
        "n_failure":  len(failures),
        "recall_at_k": len(successes) / len(photo_paths) if photo_paths else 0,
        "failures":   failures[:10],  # keep top-10 for analysis
    }



def plot_clip_score_distribution(scores: list[float], label: str = "CLIP Similarity"):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(scores, bins=30, kde=True, ax=ax, color="#2E86AB")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {label} Scores")
    ax.axvline(np.mean(scores), color="red", linestyle="--", label=f"Mean: {np.mean(scores):.3f}")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "clip_score_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅ Saved → {out}")


def plot_reward_curve(reward_log_path: str):
    """Plot RLHF reward over training steps."""
    with open(reward_log_path) as f:
        log = json.load(f)
    steps   = [entry["step"]        for entry in log]
    rewards = [entry["mean_reward"] for entry in log]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, rewards, color="#E84855", linewidth=2)
    ax.set_xlabel("PPO Step")
    ax.set_ylabel("Mean CLIP Reward")
    ax.set_title("RLHF Training: CLIP Reward over PPO Steps")

    if len(rewards) >= 10:
        window = max(1, len(rewards) // 10)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax.plot(steps[:len(smoothed)], smoothed, color="#2E86AB",
                linewidth=2, linestyle="--", label="Smoothed")
        ax.legend()

    plt.tight_layout()
    out = RESULTS_DIR / "rlhf_reward_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅ Saved → {out}")


def visualize_failure_cases(failures: list[dict], n: int = 4):
    """Show query images and their (wrong) top retrieved results."""
    if not failures:
        print("No failure cases to visualize.")
        return

    fig, axes = plt.subplots(min(n, len(failures)), 4, figsize=(16, 4 * min(n, len(failures))))
    if min(n, len(failures)) == 1:
        axes = [axes]

    for i, failure in enumerate(failures[:n]):
        try:
            query_img = Image.open(failure["query_path"]).resize((200, 200))
            axes[i][0].imshow(query_img)
            axes[i][0].set_title(f"Query\n(expect: {failure['expected_country']})", fontsize=8)
            axes[i][0].axis("off")
        except Exception:
            axes[i][0].axis("off")

        for j, (name, country, score) in enumerate(failure["top_results"][:3]):
            axes[i][j+1].set_title(f"{name[:20]}\n{country}\nsim={score:.3f}", fontsize=7)
            axes[i][j+1].axis("off")

    plt.suptitle("Failure Cases: Expected Country Not in Top Results", fontsize=12)
    plt.tight_layout()
    out = RESULTS_DIR / "failure_cases.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅ Saved → {out}")


def run_full_evaluation():
    from data_collection import _city_to_country, ASIAN_CITIES
    import pickle

    with open("data/processed/restaurant_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    eval_photos   = [m["photo_path"] for m in meta[:50] if Path(m["photo_path"]).exists()]
    eval_countries= [m["country"]    for m in meta[:50] if Path(m["photo_path"]).exists()]

    print(f"Evaluation set: {len(eval_photos)} photos")

    print("\n[1/4] Benchmarking inference time...")
    timing = benchmark_inference(eval_photos, n_runs=10)
    print(f"  Mean: {timing['mean_ms']:.1f}ms | Throughput: {timing['throughput_qps']:.2f} QPS")

    print("\n[2/4] Computing CLIP similarity scores...")
    clip_scores = []
    for path in tqdm(eval_photos[:30]):
        results, _, _ = search(path, top_k=1, use_llm=False)
        if results and Path(results[0].photo_path).exists():
            score = clip_similarity_score(path, results[0].photo_path)
            clip_scores.append(score)

    if clip_scores:
        plot_clip_score_distribution(clip_scores)
        print(f"  Mean CLIP similarity: {np.mean(clip_scores):.4f}")

    print("\n[3/4] Failure case analysis...")
    analysis = analyze_failures(eval_photos[:30], eval_countries[:30], top_k=5)
    print(f"  Recall@5: {analysis['recall_at_k']:.4f}")
    print(f"  Failures: {analysis['n_failure']} / {analysis['n_total']}")
    visualize_failure_cases(analysis["failures"])

    reward_log = "models/rlhf_qwen/reward_log.json"
    if Path(reward_log).exists():
        print("\n[4/4] Plotting RLHF reward curve...")
        plot_reward_curve(reward_log)

    summary = {
        "inference_timing": timing,
        "mean_clip_similarity": float(np.mean(clip_scores)) if clip_scores else 0,
        "recall_at_5": analysis["recall_at_k"],
        "n_eval_photos": len(eval_photos),
    }
    with open(RESULTS_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Evaluation complete → {RESULTS_DIR}/evaluation_summary.json")
    return summary


if __name__ == "__main__":
    run_full_evaluation()
