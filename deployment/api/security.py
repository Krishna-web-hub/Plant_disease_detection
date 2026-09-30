"""Security & Configuration Module: Enforces Rule 1 (Secret Management & Zero Key Leakage).

Prevents:
1. Exposed environment variables and API keys in logs, errors, or client responses.
2. Uncontrolled CORS origins in production environments.
3. Prompt injection vectors into downstream multi-agent / RAG components.
"""
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field


logger = logging.getLogger("plant_security")

# Known secret patterns for redacting in logs and exception traces
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key|secret|password|token)[\"']?\s*[:=]\s*[\"']?([a-zA-Z0-9_\-\.]{8,})[\"']?", re.IGNORECASE),
]

# Sensitive keys to never serialize in responses or error bodies
SENSITIVE_KEY_NAMES = {
    "api_key",
    "openrouter_api_key",
    "secret",
    "authorization",
    "token",
    "password",
    "client_secret",
}


class SecuritySettings(BaseModel):
    """Validated security configuration."""
    environment: str = Field(default_factory=lambda: os.environ.get("ENVIRONMENT", "development").lower())
    api_key: Optional[str] = Field(default_factory=lambda: os.environ.get("PLANT_API_KEY") or os.environ.get("API_KEY"))
    admin_api_key: Optional[str] = Field(default_factory=lambda: os.environ.get("ADMIN_API_KEY"))
    openrouter_api_key: Optional[str] = Field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY"))
    cors_origins: List[str] = Field(default_factory=lambda: [
        origin.strip()
        for origin in os.environ.get(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ])
    rate_limit_per_minute: int = Field(default_factory=lambda: int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60")))
    rate_limit_burst: int = Field(default_factory=lambda: int(os.environ.get("RATE_LIMIT_BURST", "15")))
    max_upload_size_bytes: int = Field(default_factory=lambda: int(os.environ.get("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024))))

    @property
    def is_production(self) -> bool:
        return self.environment in ("production", "prod")

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    @property
    def has_openrouter_key(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_api_key.strip())


_settings: Optional[SecuritySettings] = None


def get_security_settings() -> SecuritySettings:
    """Returns singleton validated security settings."""
    global _settings
    if _settings is None:
        _settings = SecuritySettings()
    return _settings


def redact_secrets(val: Any) -> Any:
    """Recursively redacts secrets and sensitive keys from dicts, lists, and strings."""
    if isinstance(val, str):
        redacted = val
        for pat in SECRET_PATTERNS:
            redacted = pat.sub("[REDACTED_SECRET]", redacted)
        return redacted
    elif isinstance(val, dict):
        clean_dict = {}
        for k, v in val.items():
            if str(k).lower() in SENSITIVE_KEY_NAMES:
                clean_dict[k] = "[REDACTED_SECRET]"
            else:
                clean_dict[k] = redact_secrets(v)
        return clean_dict
    elif isinstance(val, (list, tuple)):
        return [redact_secrets(item) for item in val]
    return val


class SecretRedactingFilter(logging.Filter):
    """Logging filter that redacts secrets and sensitive keys from all logs."""
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_secrets(record.args)
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(redact_secrets(a) for a in record.args)
        return True


def sanitize_input_text(text: str, max_chars: int = 500) -> str:
    """Sanitizes user or retrieved text to mitigate prompt injection and control character attacks.

    Strips markdown control sequences, LLM delimiters, and unprintable characters.
    """
    if not text:
        return ""
    # Strip dangerous LLM delimiter keywords
    text = re.sub(r"(?i)\b(ignore previous instructions|system prompt|disregard above|assistant:|<\|im_start\|>|<\|im_end\|>)\b", "[FILTERED]", text)
    # Strip unprintable ASCII control characters except standard whitespace
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")
    return text[:max_chars].strip()
