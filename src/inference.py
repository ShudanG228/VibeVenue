import pickle
import time
import numpy as np
import faiss
import torch
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from clip_encoder import encode_image, encode_text, get_device

PROCESSED_DIR = Path("data/processed")
INDEX_PATH    = PROCESSED_DIR / "restaurant.faiss"
META_PATH     = PROCESSED_DIR / "restaurant_meta.pkl"

RLHF_MODEL_DIR = Path("models/rlhf_qwen/final")
SFT_MODEL_DIR  = Path("models/sft_qwen/final")
BASE_MODEL     = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = get_device()

_faiss_index = None
_meta        = None
_llm_model   = None
_llm_tokenizer = None


@dataclass
class RestaurantResult:
    name:        str
    city:        str
    country:     str
    address:     str
    rating:      float
    num_ratings: int
    lat:         float
    lng:         float
    photo_path:  str
    image_score: float
    text_score:  float
    ensemble_score: float
    vibe_description: str = ""


def _load_index():
    global _faiss_index, _meta
    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            _meta = pickle.load(f)
        print(f"[Index] Loaded {_faiss_index.ntotal} vectors, {len(_meta)} metadata rows")


def _load_llm():
    global _llm_model, _llm_tokenizer
    if _llm_model is None:
    
        model_path = RLHF_MODEL_DIR if RLHF_MODEL_DIR.exists() else SFT_MODEL_DIR
        print(f"[LLM] Loading from {model_path}...")
        _llm_tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        _llm_tokenizer.pad_token = _llm_tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, trust_remote_code=True
        )

        _llm_model = PeftModel.from_pretrained(base, str(model_path))
        _llm_model.eval()
        print("[LLM] Ready.")


def generate_vibe_description(image_emb: np.ndarray) -> str:
    _load_llm()
    instruction = (
        "Describe the scenery and atmosphere of an Asian restaurant or cafe "
        "in one vivid, evocative sentence suitable for travel discovery."
    )
    prompt = (
        f"<|im_start|>system\n"
        f"You are a travel and dining expert specializing in Asian restaurant atmospheres.<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = _llm_tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = _llm_model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=_llm_tokenizer.eos_token_id,
        )
    decoded = _llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in decoded:
        return decoded.split("<|im_start|>assistant")[-1].strip()
    return decoded.split(instruction)[-1].strip()


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def search(
    query_image: Image.Image | str | Path,
    top_k: int = 10,
    use_llm: bool = True,
) -> tuple[list[RestaurantResult], str, float]:
    
    _load_index()
    t0 = time.time()

    img_emb = encode_image(query_image).astype(np.float32).reshape(1, -1)

    img_scores, img_indices = _faiss_index.search(img_emb, top_k * 3)
    img_ranking = img_indices[0].tolist()
    img_score_map = {idx: float(score) for idx, score in zip(img_indices[0], img_scores[0])}

    vibe_desc = ""
    text_ranking = []
    text_score_map = {}

    if use_llm:
        try:
            vibe_desc = generate_vibe_description(img_emb)
            text_emb = encode_text(vibe_desc).astype(np.float32).reshape(1, -1)
            txt_scores, txt_indices = _faiss_index.search(text_emb, top_k * 3)
            text_ranking = txt_indices[0].tolist()
            text_score_map = {idx: float(score) for idx, score in zip(txt_indices[0], txt_scores[0])}
        except Exception as e:
            print(f"[WARN] LLM generation failed: {e}, using image-only.")
            text_ranking = img_ranking

    if text_ranking:
        rrf_scores = reciprocal_rank_fusion([img_ranking, text_ranking])
    else:
        rrf_scores = {idx: 1.0 / (r + 1) for r, idx in enumerate(img_ranking)}

    seen_places: dict[str, RestaurantResult] = {}
    for idx, ensemble_score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
        if idx >= len(_meta):
            continue
        row = _meta[idx]
        place_id = row["place_id"]
        if place_id in seen_places:
            continue

        result = RestaurantResult(
            name=row["name"],
            city=row["city"],
            country=row["country"],
            address=row["address"],
            rating=row["rating"],
            num_ratings=row["num_ratings"],
            lat=row["lat"],
            lng=row["lng"],
            photo_path=row["photo_path"],
            image_score=img_score_map.get(idx, 0.0),
            text_score=text_score_map.get(idx, 0.0),
            ensemble_score=ensemble_score,
            vibe_description=vibe_desc,
        )
        seen_places[place_id] = result

        if len(seen_places) >= top_k:
            break

    results = list(seen_places.values())
    inference_ms = (time.time() - t0) * 1000

    return results, vibe_desc, inference_ms


if __name__ == "__main__":

    from pathlib import Path
    photos = list(Path("data/raw/photos").glob("*.jpg"))
    if photos:
        results, desc, ms = search(photos[0], top_k=5)
        print(f"\nVibe: {desc}")
        print(f"Inference: {ms:.1f}ms")
        for i, r in enumerate(results):
            print(f"{i+1}. {r.name} ({r.city}) | img={r.image_score:.3f} | ens={r.ensemble_score:.4f}")
