import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
MODEL_NAME = "openai/clip-vit-base-patch32"

_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is None:
        print(f"[CLIP] Loading {MODEL_NAME} on {DEVICE}...")
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        _model.eval()
        print(f"[CLIP] Model ready.")


def encode_image(image) -> np.ndarray:
    _load_model()
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")
    inputs = _processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = _model(**inputs, input_ids=torch.zeros(1,1,dtype=torch.long).to(DEVICE), attention_mask=torch.ones(1,1,dtype=torch.long).to(DEVICE))
        emb = outputs.image_embeds
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze().cpu().float().numpy()


def encode_images_batch(paths, batch_size: int = 32) -> np.ndarray:
    _load_model()
    all_embs = []
    dummy_input = _processor(text=["a"], return_tensors="pt", padding=True)
    dummy_ids = dummy_input["input_ids"].to(DEVICE)
    dummy_mask = dummy_input["attention_mask"].to(DEVICE)
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i: i + batch_size]
        images = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"[WARN] Could not load {p}: {e}")
        if not images:
            continue
        inputs = _processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
        batch_ids = dummy_ids.expand(len(images), -1)
        batch_mask = dummy_mask.expand(len(images), -1)
        with torch.no_grad():
            outputs = _model(**inputs, input_ids=batch_ids, attention_mask=batch_mask)
            emb = outputs.image_embeds
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs) if all_embs else np.zeros((0, 512), dtype=np.float32)


def encode_text(text: str) -> np.ndarray:
    _load_model()
    inputs = _processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    dummy_img = _processor(images=Image.new("RGB",(224,224)), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = _model(**dummy_img, **inputs)
        emb = outputs.text_embeds
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze().cpu().float().numpy()


def encode_texts_batch(texts: list) -> np.ndarray:
    _load_model()
    inputs = _processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    dummy_img = _processor(images=Image.new("RGB",(224,224)), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = _model(**dummy_img, **inputs)
        emb = outputs.text_embeds
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().float().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    text_emb = encode_text("cozy bamboo forest cafe with morning mist")
    print(f"Text: {text_emb.shape}")
    from pathlib import Path
    photos = list(Path("data/raw/photos").glob("*.jpg"))
    if photos:
        img_emb = encode_image(photos[0])
        print(f"Image: {img_emb.shape}")
        print(f"Same dim: {img_emb.shape == text_emb.shape}")