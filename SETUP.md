# Setup Instructions

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) or Linux
- Python 3.10+
- Conda (recommended)
- 8GB RAM minimum, 16GB recommended
- Google Places API key (only needed if re-collecting data)

## Step-by-Step Installation

### 1. Clone the repository

```bash
git clone https://github.com/ShudanG228/VibeVenue.git
cd VibeVenue
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
conda install -c conda-forge faiss-cpu  # required on Apple Silicon
```

### 3. Configure API key (only needed for data collection)

```bash
cp .env.example .env
```

Open `.env` and replace `your_key_here` with your Google Places API key.

---

## Quick Testing (For graders)

The pre-built FAISS index and trained models are included in the repository. You can launch the web app directly without running the full pipeline:

```bash
python app.py
# Then open website http://localhost:7860
# Upload any scenery photo to get matching restaurant recommendations
```

No API key needed for this option.

---

## Full Pipeline

Only needed if you want to re-collect data and retrain models from scratch. Requires a Google Places API key.

```bash
python run_pipeline.py --step collect    # fetch restaurant data
python run_pipeline.py --step index      # build FAISS index
python run_pipeline.py --step sft_data   # generate SFT training data
python run_pipeline.py --step sft_train  # fine-tune Qwen2.5-0.5B
python run_pipeline.py --step rlhf       # RLHF with CLIP reward
python run_pipeline.py --step eval       # run evaluation
python app.py                            # launch web app
```

**Getting a Google Places API key:**
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project and enable "Places API (New)"
3. Go to Credentials → Create Credentials → API Key
4. Billing must be enabled but the $200/month free tier covers our usage

---

## Troubleshooting

**`faiss-cpu` installation fails:**
```bash
conda install -c conda-forge faiss-cpu
```

**MPS errors on Apple Silicon:**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python run_pipeline.py --step sft_train
```

**CUDA/CPU memory issue(Out of memory during training):**
Open `src/sft_train.py` and change `BATCH_SIZE = 4` to `BATCH_SIZE = 2`.

**Module not found errors:**
Make sure you run all commands from the `VibeVenue/` root directory.