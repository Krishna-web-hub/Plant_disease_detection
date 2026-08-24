# Plant Disease Detection

Two-stage image classifier (crop → disease) for tomato, potato, and pepper
leaves, served behind a FastAPI backend with a RAG fallback for low-confidence
predictions, plus web and mobile clients. Photos of plants outside the
trained set are diagnosed by a vision LLM instead, grounded with the same
literature retrieval — so the product isn't hard-capped at three crops, and
unlike a plain "ask an AI" tool, every answer (trained-model or LLM-fallback)
is checked against real sources rather than taken on faith.

## Positioning vs. commercial plant-ID apps (e.g. MyPlantIn)

Those apps show a confident-sounding diagnosis with a vague "expert tips"
blurb and no way to check it, behind a subscription paywall. This project
makes different tradeoffs on purpose:

| | Typical plant-ID app | This project |
|---|---|---|
| Confidence | Not shown | Shown on every prediction, including an honest "uncertain" response when it's genuinely unsure |
| Treatment advice | Generic text, no citations | Grounded in PubMed abstracts + land-grant extension fact sheets, with clickable source links |
| Coverage limits | Rarely stated | Explicit: trained crops (Tomato/Potato/Pepper) vs. AI-vision fallback for everything else, and the UI tells you which one you're looking at |
| Cost | Subscription | None — self-hosted, only external cost is optional LLM API usage |

Not a target: 24,000-species plant *identification*, care reminders, or
human botanist chat — this stays scoped to disease diagnosis, where the
grounding/transparency edge above actually matters.

## Pipeline

```
data/raw/PlantVillage  →  training/prepare_data.py  →  data/processed/
                                                          (manifest.csv, class weights)
                                                                 │
                                                                 ▼
                                        training/train_crop_classifier.py
                                        training/train_disease_classifier.py --crop <Crop>
                                                                 │
                                                                 ▼
                                        models/crop_classifier/best.pt
                                        models/disease_classifier/<Crop>/best.pt
                                                                 │
                                                                 ▼
                                        deployment/api  (FastAPI, confidence routing)
                                                │                       │
                                                ▼                       ▼
                                      deployment/rag (fallback)   deployment/web, deployment/mobile
```

Crop classification runs first.
- **Confident + trained crop** → the matching disease classifier runs.
  High confidence returns the treatment directly; mid confidence triggers a
  RAG lookup (vector store + PubMed + LLM) for a second opinion; low
  confidence returns an "uncertain" response.
- **Not confident, or a crop we haven't trained a disease classifier for**
  → falls back to a vision-capable LLM (via OpenRouter) that identifies the
  plant/disease from the photo directly, then the *same* RAG grounding
  (vector store + PubMed) is run against its answer before a final reasoning
  pass — so an unfamiliar plant still gets a literature-checked response, not
  a raw LLM guess. This only fires when `OPENROUTER_API_KEY` is set; without
  it, unrecognized crops fall back to the plain "uncertain" response instead.

## Status

- **Data**: prepared — 20,639 images across 15 categories (Pepper, Potato,
  Tomato), stratified 70/15/15 train/val/test split, duplicates and one
  corrupt file excluded. See `reports/data_quality_report.json` and
  `reports/eda_summary.json`.
- **Models**: not yet trained — `models/` is empty. Training must run on a
  GPU machine (see `config/training_config.yaml`).
- **API / RAG / web app / mobile app**: code complete, untested end-to-end
  (needs trained weights and an `OPENROUTER_API_KEY` to fully exercise). The
  web app's build/typecheck (`npm run build`) passes.
- **Tests**: none yet (`tests/` is empty).

## Before you train

Everything downstream of training is already built and wired up. Checklist
for what's ready vs. what still needs you:

- [x] Data prepared (`data/processed/manifest.csv` + class weights)
- [x] Training scripts (`training/train_crop_classifier.py`, `train_disease_classifier.py`)
- [x] API with confidence routing + graceful degradation (`deployment/api/`)
- [x] RAG grounding for both known crops and the AI vision fallback (`deployment/rag/`)
- [x] Web app (`deployment/web/`) — builds clean (`npm run build`)
- [x] Mobile app (`deployment/mobile/`)
- [ ] **Run training** (see step 2 below) — needs a GPU machine; this repo
      has no `torch` installed and no GPU detected in this environment
- [ ] Set `OPENROUTER_API_KEY` (copy `.env.example` → `.env`, or `export` it)
      so RAG and the AI fallback work — optional for direct model predictions
- [ ] Point `API_BASE_URL` (`deployment/web/src/config.ts`,
      `deployment/mobile/src/config.ts`) at wherever you end up running the API
- [ ] Write tests (`tests/` is currently empty)

Once `models/crop_classifier/best.pt` and `models/disease_classifier/<Crop>/best.pt`
exist, `deployment/api/model_loader.py` picks them up automatically on the
next API startup — no code changes needed.

## Setup

```bash
pip install -r requirements.txt
```

Training and the API need `torch`/`timm` (GPU strongly recommended for
training — the `vit_base_patch16_224` crop classifier is slow on CPU). RAG
needs `chromadb` + `sentence-transformers` plus a live `OPENROUTER_API_KEY`.

## 1. Prepare data

Only needed once, or after adding new images to `data/raw/PlantVillage`:

```bash
python training/prepare_data.py
```

Writes `data/processed/manifest.csv`, `class_weights_crop.json`, and
per-crop weights under `class_weights_disease/`.

## 2. Train

```bash
python training/train_crop_classifier.py
python training/train_disease_classifier.py --crop Tomato
python training/train_disease_classifier.py --crop Potato
python training/train_disease_classifier.py --crop Pepper
```

Config lives in `config/training_config.yaml` (image size, batch size, LR,
epochs, early-stopping patience). Each run writes `best.pt` +
`class_to_idx.json` to its `output_dir`; early stopping keeps only the
best-val-accuracy checkpoint.

## 3. Run the API

```bash
uvicorn deployment.api.app:app --reload
```

- `GET /health` — model/RAG load status.
- `POST /predict` — multipart image upload, returns `DIRECT`, `RAG_ENHANCED`,
  `RAG_PENDING`, `AI_FALLBACK`, or `UNCERTAIN` depending on confidence and
  crop coverage (`config/inference_config.yaml` sets the thresholds). Missing
  models or an unavailable RAG pipeline degrade gracefully (503 / RAG skipped)
  rather than crashing the service.

RAG pulls from the internal treatment DB (`deployment/api/treatment_db.json`),
cached extension fact sheets (`deployment/rag/build_extension_cache.py`),
and a live PubMed lookup, then asks an LLM (via OpenRouter) to reason over
the retrieved context. Requires `OPENROUTER_API_KEY` in the environment.

For plants outside Tomato/Potato/Pepper, `PlantDiseaseRAG.diagnose_unknown_plant`
(`deployment/rag/rag_pipeline.py`) sends the photo to a vision-capable model
(`config/inference_config.yaml → rag.llm.model`, default
`anthropic/claude-3.5-sonnet` via OpenRouter — supports images) to identify
the plant + disease as JSON, then runs the same vector-store/PubMed retrieval
keyed on that identification before a second LLM call produces the final,
source-cited answer. `AI_FALLBACK` responses carry a `vision_confidence`
(the model's self-reported estimate, not a calibrated number) and should be
presented to users as a second opinion, not a verified diagnosis.

## Response types (`/predict`)

Every response has a `type` field the clients switch on. Five shapes.
Wherever sources appear (`rag_sources`, `sources`), each entry is
`{"source": "...", "url": "..." | null}` — `url` is set for anything
individually citable (PubMed abstracts, extension fact sheets) so the web
and mobile clients render them as clickable links, not just a name to trust.
`treatment.severity` (`none`/`moderate`/`moderate_to_high`/`high`) is
rendered as a badge on every response carrying a `treatment` object.

**`DIRECT`** — trained model is confident (`confidence >= confidence_threshold`).
```json
{
  "type": "DIRECT",
  "crop": "Tomato",
  "category": "Tomato_Early_blight",
  "disease": "Early Blight",
  "confidence": 0.93,
  "treatment": { "crop": "Tomato", "disease": "Early Blight", "...": "..." },
  "rag_used": false
}
```

**`RAG_ENHANCED`** — disease confidence is mid-range (between
`rag_trigger_threshold` and `confidence_threshold`) and RAG is available;
treatment is grounded with retrieved sources plus an LLM's second opinion.
```json
{
  "type": "RAG_ENHANCED",
  "crop": "Tomato",
  "category": "Tomato_Early_blight",
  "model_confidence": 0.71,
  "treatment": { "...": "..." },
  "rag_reasoning": "The model's detection is plausible given...",
  "rag_sources": [
    { "source": "PlantVillage", "url": null },
    { "source": "PubMed", "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/" }
  ],
  "rag_latency_seconds": 2.3,
  "rag_used": true
}
```

**`RAG_PENDING`** — confidence was in the RAG range but the RAG pipeline
isn't available (no `OPENROUTER_API_KEY`, or `chromadb`/`sentence-transformers`
not installed); falls back to the raw model prediction.
```json
{
  "type": "RAG_PENDING",
  "crop": "Tomato",
  "category": "Tomato_Early_blight",
  "model_confidence": 0.71,
  "treatment": { "...": "..." },
  "message": "Confidence is in the RAG range, but the RAG pipeline is unavailable...",
  "rag_used": false
}
```

**`AI_FALLBACK`** — the plant is outside the trained crop set (low crop
confidence, or a crop with no trained disease classifier) and RAG is
available; a vision LLM identifies the plant/disease and it's grounded the
same way as `RAG_ENHANCED`.
```json
{
  "type": "AI_FALLBACK",
  "plant": "Rose",
  "disease": "Black spot",
  "is_healthy": false,
  "vision_confidence": 0.8,
  "reasoning": "Based on the retrieved information and the visible symptoms...",
  "sources": [{ "source": "PubMed", "url": "https://pubmed.ncbi.nlm.nih.gov/23456789/" }],
  "rag_latency_seconds": 3.1,
  "message": "Outside our trained crops (Tomato/Potato/Pepper) — diagnosed by an AI vision model..."
}
```

**`UNCERTAIN`** — low confidence and no RAG/AI fallback available. Two
`stage`s: `crop_classification` (crop itself unclear) or
`disease_classification` (crop known, disease confidence too low).
```json
{
  "type": "UNCERTAIN",
  "stage": "crop_classification",
  "crop": "Tomato",
  "confidence": 0.42,
  "message": "Could not confidently identify the crop."
}
```

`GET /health` reports `{"status", "models_loaded", "disease_crops_available"}`.
`POST /predict` returns `503` if models aren't trained yet, or (only when RAG
is also unavailable) if the crop has no trained disease classifier.

## 4. Web app

```bash
cd deployment/web
npm install
npm run dev
```

Upload-and-diagnose single page: pick a photo, see crop/disease/confidence
and treatment (or the RAG-enhanced explanation) rendered from the same
`/predict` response the mobile app consumes. `API_BASE_URL` in `src/config.ts`
defaults to `http://localhost:8000`. The API has CORS enabled (`allow_origins:
["*"]`) so the browser can call it directly during development — tighten
that before deploying publicly.

## 5. Mobile app

```bash
cd deployment/mobile
npm install
npm start
```

Set `API_BASE_URL` in `src/config.ts` to your machine's LAN IP (not
`localhost` — it won't reach a phone or the Android emulator) before running
on a device.

## Layout

| Path | Purpose |
|---|---|
| `data/` | Raw images, processed manifest, RAG source registry |
| `config/` | Training, inference, and RAG configuration |
| `training/` | Data prep, dataset/engine/utils, per-stage train scripts |
| `models/` | Trained checkpoints (gitignored — populated by training) |
| `deployment/api/` | FastAPI inference service + treatment lookup |
| `deployment/rag/` | Vector store, retrievers (PubMed, extension docs), LLM reasoning |
| `deployment/web/` | Vite/React web client (upload-and-diagnose) |
| `deployment/mobile/` | Expo/React Native client |
| `notebooks/` | Data quality + EDA scripts |
| `reports/` | Data quality and EDA output |
| `tests/` | (empty — no tests yet) |
