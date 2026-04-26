import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from clip_encoder import encode_image, encode_texts_batch, cosine_similarity

RAW_DIR       = Path("data/raw")
SFT_DATA_PATH = Path("data/sft_dataset/sft_train.json")
SFT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

VIBE_CANDIDATES = [
    ("bamboo forest",         "A serene bamboo forest cafe with dappled morning light filtering through tall green stalks, creating a tranquil Zen atmosphere."),
    ("rooftop city view",     "A stylish rooftop cafe overlooking a glittering Asian cityscape at dusk, with warm string lights and a cool evening breeze."),
    ("traditional courtyard", "A charming traditional courtyard restaurant with ancient stone walls, hanging red lanterns, and the scent of jasmine tea in the air."),
    ("night market street",   "A vibrant night market setting with colorful neon signs, bustling crowds, and the irresistible aroma of street food on a warm tropical night."),
    ("minimalist modern",     "A sleek minimalist cafe with floor-to-ceiling windows, clean white surfaces, and a calm Nordic-Asian aesthetic perfect for focused work."),
    ("tropical garden",       "A lush tropical garden restaurant surrounded by exotic plants and flowers, with open-air seating and birdsong accompanying every meal."),
    ("mountain valley",       "A cozy mountain retreat cafe perched above misty valleys, with rustic wooden decor and panoramic views of terraced rice fields."),
    ("seaside waterfront",    "A breezy waterfront cafe with unobstructed ocean views, salty sea air, and the gentle sound of waves lapping against wooden piers."),
    ("cherry blossom park",   "A delightful outdoor cafe nestled beneath blooming cherry blossom trees, with pink petals drifting into matcha lattes in spring."),
    ("industrial chic",       "An industrial-chic cafe in a converted warehouse with exposed brick, raw steel beams, and an edgy urban energy that fuels creativity."),
    ("colonial heritage",     "A historic colonial-era building cafe with high ceilings, mosaic tile floors, rattan furniture, and an atmosphere steeped in Southeast Asian history."),
    ("zen rock garden",       "A peaceful Japanese zen garden restaurant where raked gravel patterns and perfectly placed stones invite quiet contemplation between courses."),
    ("riverside floating",    "A unique floating restaurant on a tranquil river, where diners enjoy local cuisine while watching traditional wooden boats drift by."),
    ("lotus pond",            "A romantic cafe beside a still lotus pond, with the gentle fragrance of white flowers and dragonflies hovering over the mirrored water."),
    ("foggy highland",        "A highland tea plantation cafe shrouded in cool morning fog, surrounded by endless rows of emerald tea bushes and cool mountain air."),
    ("lantern alley",         "A magical narrow alleyway cafe lit by hundreds of hanging paper lanterns, casting a warm amber glow over stone cobblestones at night."),
    ("glass greenhouse",      "A stunning glass greenhouse restaurant filled with tropical plants and natural light, where diners feel immersed in a botanical garden."),
    ("beachfront shack",      "A casual beachfront shack with wooden deck chairs, colorful hammocks, and the sound of waves just steps from your table."),
    ("temple courtyard",      "A serene restaurant set within ancient temple grounds, with stone carvings, incense, and the distant sound of monks chanting."),
    ("sky bridge view",       "A dramatic sky-high restaurant suspended above the clouds with panoramic glass walls and a breathtaking bird's eye view of the city."),
]

VIBE_LABELS       = [v[0] for v in VIBE_CANDIDATES]
VIBE_DESCRIPTIONS = [v[1] for v in VIBE_CANDIDATES]

_vibe_text_embs = None

def _get_vibe_embs() -> np.ndarray:
    global _vibe_text_embs
    if _vibe_text_embs is None:
        prompts = [f"a photo of a restaurant with {label} scenery" for label in VIBE_LABELS]
        _vibe_text_embs = encode_texts_batch(prompts)
    return _vibe_text_embs


def classify_vibe_topk(image_path: str, top_k: int = 3) -> list[tuple[str, str, float]]:
    img_emb   = encode_image(image_path)
    vibe_embs = _get_vibe_embs()
    scores    = vibe_embs @ img_emb
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        (VIBE_LABELS[i], VIBE_DESCRIPTIONS[i], float(scores[i]))
        for i in top_indices
    ]


def generate_sft_dataset(top_k: int = 3, min_confidence: float = 0.10) -> list[dict]:
    
    photos_dir  = RAW_DIR / "photos"
    photo_paths = list(photos_dir.glob("*.jpg"))
    print(f"Found {len(photo_paths)} photos, generating up to {len(photo_paths) * top_k} samples")

    dataset = []
    for path in tqdm(photo_paths, desc="Generating SFT data"):
        try:
            vibes = classify_vibe_topk(str(path), top_k=top_k)
            for label, description, score in vibes:
                if score >= min_confidence:
                    dataset.append({
                        "image_path":  str(path),
                        "vibe_label":  label,
                        "description": description,
                        "confidence":  round(score, 4),
                        "instruction": "Describe the scenery and atmosphere visible in this restaurant or cafe photo in one vivid sentence.",
                        "output":      description,
                    })
        except Exception as e:
            print(f"[WARN] {path.name}: {e}")


    import random
    random.seed(42)
    random.shuffle(dataset)

    n = len(dataset)
    splits = {
        "train": dataset[:int(0.8*n)],
        "val":   dataset[int(0.8*n):int(0.9*n)],
        "test":  dataset[int(0.9*n):],
    }

    for split_name, split_data in splits.items():
        out = SFT_DATA_PATH.parent / f"sft_{split_name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"✅ {split_name}: {len(split_data)} samples → {out}")

    return dataset


if __name__ == "__main__":
    data = generate_sft_dataset(top_k=3)
    print(f"\nTotal SFT samples: {len(data)}")
    if data:
        print("\nExample:")
        print(json.dumps(data[0], indent=2))
