"""FastAPI inference service: crop + disease classification with confidence-based RAG fallback.

Run from the project root: uvicorn deployment.api.app:app --reload
"""
import io
import logging

import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from deployment.api.model_loader import ModelBundle
from deployment.api.treatment import get_treatment

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Plant Disease Detection API")

# Dev-friendly: allow the web app (Vite on a different port) and the mobile
# app to call the API from the browser. Tighten to specific origins in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_inference_cfg = yaml.safe_load(open("config/inference_config.yaml"))
CONFIDENCE_THRESHOLD = _inference_cfg["rag"]["confidence_threshold"]
RAG_TRIGGER_THRESHOLD = _inference_cfg["rag"]["rag_trigger_threshold"]

_models = None
_rag = None


@app.on_event("startup")
def load_models():
    global _models, _rag
    try:
        _models = ModelBundle()
        logger.info("Models loaded: crop classifier + %d disease classifiers", len(_models.disease_models))
    except FileNotFoundError as e:
        logger.warning("Models not found (%s) — /predict will return 503 until training completes.", e)

    try:
        from deployment.rag.rag_pipeline import PlantDiseaseRAG
        _rag = PlantDiseaseRAG()
        logger.info("RAG pipeline ready.")
    except Exception as e:
        logger.warning("RAG pipeline unavailable (%s) — low-confidence predictions will skip RAG.", e)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": _models is not None,
        "disease_crops_available": sorted(_models.disease_models.keys()) if _models else [],
    }


def _ai_fallback_response(image):
    """Diagnose a plant the trained classifiers don't cover, via vision-LLM + RAG grounding.

    Returns None (rather than raising) on any failure so callers can fall
    back to the plain UNCERTAIN/503 response instead of erroring the request.
    """
    if _rag is None:
        return None
    try:
        result = _rag.diagnose_unknown_plant(image)
    except Exception as e:
        logger.warning("AI fallback diagnosis failed (%s)", e)
        return None

    return {
        "type": "AI_FALLBACK",
        "plant": result["plant"],
        "disease": result["disease"],
        "is_healthy": result["is_healthy"],
        "vision_confidence": result["vision_confidence"],
        "reasoning": result["reasoning"],
        "sources": result["sources"],
        "rag_latency_seconds": result["latency_seconds"],
        "message": "Outside our trained crops (Tomato/Potato/Pepper) — diagnosed by an AI vision "
                   "model, grounded with retrieved literature where available. Treat as a second "
                   "opinion, not a lab-verified result.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), use_rag: bool = Query(default=True)):
    if _models is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first (see training/).")

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    crop, crop_conf = _models.predict_crop(image)
    if crop_conf < CONFIDENCE_THRESHOLD:
        if use_rag:
            fallback = _ai_fallback_response(image)
            if fallback:
                return fallback
        return {
            "type": "UNCERTAIN",
            "stage": "crop_classification",
            "crop": crop,
            "confidence": crop_conf,
            "message": "Could not confidently identify the crop.",
        }

    if crop not in _models.disease_models:
        if use_rag:
            fallback = _ai_fallback_response(image)
            if fallback:
                return fallback
        raise HTTPException(status_code=503, detail=f"No trained disease classifier for crop '{crop}' yet.")

    category, disease_conf = _models.predict_disease(crop, image)
    treatment = get_treatment(category)

    if disease_conf >= CONFIDENCE_THRESHOLD:
        return {
            "type": "DIRECT",
            "crop": crop,
            "category": category,
            "disease": treatment["disease"],
            "confidence": disease_conf,
            "treatment": treatment,
            "rag_used": False,
        }

    if use_rag and disease_conf >= RAG_TRIGGER_THRESHOLD:
        if _rag is None:
            return {
                "type": "RAG_PENDING",
                "crop": crop,
                "category": category,
                "model_confidence": disease_conf,
                "treatment": treatment,
                "message": "Confidence is in the RAG range, but the RAG pipeline is unavailable "
                           "(check OPENROUTER_API_KEY and that chromadb/sentence-transformers are installed).",
                "rag_used": False,
            }

        rag_result = _rag.query(crop, treatment["disease"], disease_conf)
        return {
            "type": "RAG_ENHANCED",
            "crop": crop,
            "category": category,
            "model_confidence": disease_conf,
            "treatment": treatment,
            "rag_reasoning": rag_result["reasoning"],
            "rag_sources": rag_result["sources"],
            "rag_latency_seconds": rag_result["latency_seconds"],
            "rag_used": True,
        }

    return {
        "type": "UNCERTAIN",
        "stage": "disease_classification",
        "crop": crop,
        "category": category,
        "confidence": disease_conf,
        "message": "Low confidence detection. Consider consulting an expert.",
        "rag_used": False,
    }
