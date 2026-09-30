"""Authentication & Authorization Middleware: Enforces Rule 5 (Proper Authentication Middleware).

Prevents:
1. Unauthenticated external access to proprietary classification models & LLM pipelines.
2. Timing attack vulnerabilities through constant-time HMAC key comparison.
3. Unrestricted API exploitation while allowing safe, clearly logged local development workflows.
"""
import hmac
import logging
from typing import Optional
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from deployment.api.security import get_security_settings

logger = logging.getLogger("plant_auth")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def safe_compare(val1: Optional[str], val2: Optional[str]) -> bool:
    """Constant-time string comparison to prevent timing side-channel attacks."""
    if not val1 or not val2:
        return False
    return hmac.compare_digest(val1.encode("utf-8"), val2.encode("utf-8"))


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Global authentication middleware protecting sensitive API endpoints."""

    def __init__(self, app):
        super().__init__(app)
        self.settings = get_security_settings()

    async def dispatch(self, request: Request, call_next):
        # Public endpoints that bypass authentication
        public_paths = {"/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}
        if request.url.path in public_paths:
            return await call_next(request)

        # Extract API key from header: 'X-API-Key' or 'Authorization: Bearer <key>'
        api_key = request.headers.get("x-api-key")
        auth_header = request.headers.get("authorization")
        if not api_key and auth_header and auth_header.lower().startswith("bearer "):
            api_key = auth_header[7:].strip()

        # Development Mode Handling:
        # If no API key is configured in settings and environment is development,
        # allow requests but inject security warning headers so developers know it's unauthenticated.
        if not self.settings.has_api_key:
            if self.settings.is_production:
                logger.error("SECURITY CRITICAL: Production server running without configured API_KEY!")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"type": "CONFIG_ERROR", "message": "Server authentication is misconfigured."},
                )
            # Dev mode fallback
            response = await call_next(request)
            response.headers["X-Security-Warning"] = "Insecure development mode: No PLANT_API_KEY configured."
            return response

        # Production / Secured Key Validation
        is_valid = safe_compare(api_key, self.settings.api_key) or safe_compare(
            api_key, self.settings.admin_api_key
        )

        if not is_valid:
            logger.warning(
                "Unauthorized access attempt rejected on %s from %s",
                request.url.path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "type": "UNAUTHORIZED",
                    "message": "Missing or invalid authentication credentials. Provide a valid 'X-API-Key' header.",
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        response = await call_next(request)
        return response
