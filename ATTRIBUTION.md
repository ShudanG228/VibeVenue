# Attribution

## AI Development Tools

This project was developed with assistance from Claude (Anthropic) as a coding aid for brainstorming, drafting, and occasional debugging suggestions. The overall system design, implementation decisions, debugging, and integration were primarily completed independently.

---

## My Contributions

### System Design & Architecture Decisions

I designed the overall pipeline architecture and made key technical decisions:

- **Dual-path retrieval design**: Combined image-side CLIP retrieval with text-side LLM-generated description retrieval, then ensemble with Reciprocal Rank Fusion. This was a deliberate choice over simpler single-path retrieval after reasoning through the tradeoffs. And it prioritized items that consistently rank high across multiple signals and provides a more robust final ranking.
- **CLIP-as-reward for RLHF**: Designed the reward signal — using CLIP cosine similarity between generated vibe descriptions and input images as the PPO reward. This connects the LLM alignment directly to visual grounding, which is novel in this context.
- **Data collection strategy**: Decided to collect from 55 districts across 12 Asian cities (rather than just city centers) to ensure geographic and aesthetic diversity in the restaurant database, also restrict the database in Asia region. Chose 3 photos per restaurant to balance coverage and storage.
- **SFT data design**: Designed the vibe candidate vocabulary (20 categories) and the top-k=3 labeling strategy to generate ~9500 training samples from 3190 photos. Tuned the confidence threshold to 0.10 after observing the CLIP score distribution.
- **Model size selection**: Chose Qwen2.5-0.5B over larger models after reasoning through the tradeoff — 0.5B runs on CPU/MPS without quantization, making the project reproducible without GPU access, at the cost of generation quality.

### Debugging & Problem Solving

All of the following bugs required independent diagnosis and fixing:

**1. CLIP embedding dimension mismatch**
The AI-generated code used `model.get_image_features()` which in newer transformers versions returns a `BaseModelOutputWithPooling` object, not a tensor. I diagnosed this by inspecting the output type at runtime, then discovered that switching to `vision_model().pooler_output` gave 768-dim while `text_model()` gave 512-dim. I resolved this by using the full `CLIPModel` forward pass to get consistent 512-dim `image_embeds` and `text_embeds`.

**2. faiss-cpu Python 3.13 incompatibility**
`pip install faiss-cpu` failed with a `swig` compiler error on Python 3.13. I identified that conda-forge provides pre-compiled binaries and resolved it with `conda install -c conda-forge faiss-cpu`.

**3. PPOTrainer dataset compatibility**
`PPOTrainer` was removed in `trl>=1.0`. After downgrading to `trl==0.8.6`, hit a second bug where passing a HuggingFace Dataset caused `num_samples=0`. I diagnosed the issue as incorrect tokenization format and rewrote the training loop to manually sample batches with `np.random.choice`, bypassing PPOTrainer's dataset handling entirely.

**4. Jupyter kernel and path issues**
The notebook consistently failed in VS Code due to kernel startup failures and relative path issues when doing the experiments and evaluation results. I switched to browser-based Jupyter notebook and replaced all relative paths with absolute paths to get the notebook running.

**5. Gradio and rating format bugs**
Fixed Gradio 6.0 CSS deprecation (moved from `gr.Blocks(css=...)` to `launch()`), and a rating format error where `f"{r.rating:.1f}"` failed because some values were stored as strings rather than floats.

### Trade-off Analysis

- **SFT vs RLHF**: SFT alone teaches the model to generate vibe descriptions but doesn't align them to actual visual content. RLHF with CLIP reward optimizes for visual-textual alignment. The tradeoff is training complexity — RLHF requires PPO infrastructure and is somehow unstable on CPU. I ran two rounds of RLHF (adjusting temperature from 0.7 to 0.5) and documented the reward improvement.
- **Image-only vs ensemble retrieval**: Evaluated both approaches — both achieved Recall@5 = 1.0 on the test set. Concluded that CLIP has already built a good model and the LLM adds interpretability (user-facing vibe description) rather than raw retrieval accuracy, which is the honest finding.
- **RRF vs weighted sum**: Chose RRF for ensemble because it is robust to score distribution differences between image and text retrieval paths and requires no hyperparameter tuning for combination weights.
- **CLIP ViT-B/32 vs larger CLIP**: Chose ViT-B/32 for CPU inference speed over accuracy.

---

## What AI Generated

- Initial file scaffolding for `src/*.py` files and the structure of the folder in VS code editor
- Boilerplate training loop structure in `sft_train.py` and `rlhf_train.py`
- Initial Gradio UI layout in `app.py`
- Documentation templates for README, SETUP, ATTRIBUTION for rendering in github

---

## Data Sources

- Restaurant data: Google Places API (New) — [Terms of Service](https://developers.google.com/maps/terms)
- CLIP model: `openai/clip-vit-base-patch32` (MIT License)
- Base LLM: `Qwen/Qwen2.5-0.5B-Instruct` (Apache 2.0)

## External Libraries

| Library | Usage |
|---------|-------|
| PyTorch | Deep learning framework |
| Transformers | Model loading and training |
| PEFT | LoRA fine-tuning |
| TRL 0.8.6 | PPO/RLHF training |
| FAISS | Vector similarity search |
| Gradio | Web application UI |
| NumPy | Numerical computation |
| Matplotlib | Evaluation visualizations |
