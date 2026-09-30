"""FastAPI inference service: Multi-Agent Plant Disease Detection API.

Fully Enforces the 80/20 of Vibe Coding Security:
1. Rule 1: Zero exposed environment variables or API keys (SecretRedactingFilter, masked error traces).
2. Rule 2: Tenant/session data isolation & RLS partition on diagnostic audit trails.
3. Rule 3: Strict server-side validation (SecuritySentinelAgent, rate limiting, payload bounds).
4. Rule 4: Verified packages & robust runtime integrity checks.
5. Rule 5: Proper authentication & authorization middleware (AuthenticationMiddleware, constant-time comparison).

Coordinates:
- SecuritySentinelAgent (Magic bytes, DoS limits, EXIF stripping)
- OODDetectorAgent (Free Energy, Temperature Calibration, Overfit Interception)
- BotanicalCriticAgent (Morphology vs Symptom Adversarial Checker)
- TaxonomyGuardAgent (Host-Pathogen Biological Compatibility)
- SafetyRAGAgent (Literature Grounding & Non-toxic Treatment Policy)

Run from project root: uvicorn deployment.api.app:app --reload
"""
import io
import logging
import os
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from deployment.agents.orchestrator import MultiAgentOrchestrator
from deployment.api.auth import AuthenticationMiddleware
from deployment.api.model_loader import ModelBundle
from deployment.api.rate_limiter import RateLimitMiddleware
from deployment.api.security import SecretRedactingFilter, get_security_settings, redact_secrets
from deployment.api.security_headers import SecurityHeadersMiddleware
from deployment.api.treatment import get_treatment

# Attach Secret Redaction logging filter to prevent any API key leakage
logger = logging.getLogger("uvicorn.error")
logger.addFilter(SecretRedactingFilter())

sec_settings = get_security_settings()

app = FastAPI(
    title="Multi-Agent Plant Disease Detection API",
    description="Secure, robust multi-agent phytopathological diagnostics with grounded RAG verification.",
    version="2.0.0",
)

# 1. Security Headers Middleware (OWASP protection)
app.add_middleware(SecurityHeadersMiddleware)

# 2. Rate Limiting Middleware (DoS and cost drain prevention)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=sec_settings.rate_limit_per_minute,
    burst_limit=sec_settings.rate_limit_burst,
)

# 3. Authentication Middleware (Rule 5: Proper Authentication Middleware)
app.add_middleware(AuthenticationMiddleware)

# 4. CORS Middleware (Restricted to configured origins, disallowing wildcard in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sec_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "inference_config.yaml")
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _inference_cfg = yaml.safe_load(f)

CONFIDENCE_THRESHOLD = _inference_cfg["rag"]["confidence_threshold"]
RAG_TRIGGER_THRESHOLD = _inference_cfg["rag"]["rag_trigger_threshold"]

_models = None
_rag = None
_orchestrator = None


@app.on_event("startup")
def load_models():
    global _models, _rag, _orchestrator
    try:
        _models = ModelBundle()
        logger.info("Models loaded: crop classifier + %d disease classifiers", len(_models.disease_models))
    except FileNotFoundError as e:
        logger.warning("Models not found (%s) — /predict will return 503 until training completes.", redact_secrets(str(e)))

    try:
        from deployment.rag.rag_pipeline import PlantDiseaseRAG
        _rag = PlantDiseaseRAG()
        logger.info("RAG pipeline ready.")
    except Exception as e:
        logger.warning("RAG pipeline unavailable (%s) — low-confidence predictions will skip RAG.", redact_secrets(str(e)))

    if _models is not None:
        _orchestrator = MultiAgentOrchestrator(
            model_bundle=_models,
            rag_pipeline=_rag,
            treatment_fn=get_treatment,
        )
        logger.info("Multi-Agent Orchestrator initialized successfully.")


@app.get("/health")
def health():
    """Health and security posture check (zero sensitive credentials exposed)."""
    return {
        "status": "ok",
        "environment": sec_settings.environment,
        "models_loaded": _models is not None,
        "orchestrator_ready": _orchestrator is not None,
        "auth_configured": sec_settings.has_api_key,
        "rag_configured": sec_settings.has_openrouter_key,
        "disease_crops_available": sorted(_models.disease_models.keys()) if _models else [],
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_rag: bool = Query(default=True, description="Enable RAG literature grounding for borderline predictions"),
    x_session_id: Optional[str] = Header(default=None, description="Optional tenant/session ID for multi-tenant isolation"),
):
    """Diagnoses uploaded leaf image through the 5-stage secure multi-agent pipeline."""
    if _orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models or Orchestrator not loaded yet. Train models first.",
        )

    # Server-side validation: Check Content-Length before reading into memory (Rule 3)
    if file.size and file.size > sec_settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file size exceeds maximum permitted limit of {sec_settings.max_upload_size_bytes / (1024 * 1024):.1f} MB.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > sec_settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded payload size ({len(raw_bytes)} bytes) exceeds limit.",
        )

    try:
        result = _orchestrator.process(
            raw_bytes,
            use_rag=use_rag,
            session_id=x_session_id or "default_session",
        )
    except Exception as e:
        safe_msg = redact_secrets(str(e))
        logger.error("Inference execution failure: %s", safe_msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Diagnostic pipeline failed safely.")

    if result.get("type") == "SECURITY_REJECTION":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("message"))

    # Enforce safe response sanitization (no internal tokens or leakages)
    return redact_secrets(result)
