import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

from clip_encoder import encode_images_batch

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH    = PROCESSED_DIR / "restaurant.faiss"
META_PATH     = PROCESSED_DIR / "restaurant_meta.pkl"
EMBED_PATH    = PROCESSED_DIR / "restaurant_embeddings.npy"


def build_index() -> None:
    meta_file = RAW_DIR / "restaurants.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        restaurants = json.load(f)

    rows: list[dict] = []
    photo_paths: list[str] = []

    for r in restaurants:
        for photo_path in r["photo_paths"]:
            if Path(photo_path).exists():
                rows.append({
                    "place_id":    r["place_id"],
                    "name":        r["name"],
                    "city":        r["city"],
                    "country":     r["country"],
                    "address":     r["address"],
                    "rating":      r["rating"],
                    "num_ratings": r["num_ratings"],
                    "lat":         r["lat"],
                    "lng":         r["lng"],
                    "photo_path":  photo_path,
                })
                photo_paths.append(photo_path)

    print(f"Total photos to encode: {len(photo_paths)}")

    embeddings = encode_images_batch(photo_paths, batch_size=32)
    print(f"Embeddings shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"FAISS index: {index.ntotal} vectors")

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(rows, f)
    np.save(str(EMBED_PATH), embeddings)

    print(f"✅ Saved index  → {INDEX_PATH}")
    print(f"✅ Saved meta   → {META_PATH}")
    print(f"✅ Saved embeds → {EMBED_PATH}")


if __name__ == "__main__":
    build_index()
