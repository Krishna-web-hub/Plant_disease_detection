"""Sliding Window Rate Limiter: Enforces Rule 3 (Server-Side Protection against DoS & Cost Drain).

Prevents:
1. Malicious clients or automated bots flooding the GPU with image inference tasks.
2. Exhaustion of OpenRouter API credits through repeated high-frequency requests.
"""
import time
from collections import defaultdict
from typing import Dict, List, Tuple
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from deployment.api.security import get_security_settings


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding window rate limiter."""

    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 15):
        self.rpm = requests_per_minute
        self.burst = burst_limit
        self.window_seconds = 60.0
        self.history: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        """Checks if a request is within limits.

        Returns:
            (is_allowed, retry_after_seconds)
        """
        now = time.time()
        timestamps = self.history[client_id]

        # Prune timestamps older than window
        cutoff = now - self.window_seconds
        valid_timestamps = [t for t in timestamps if t > cutoff]
        self.history[client_id] = valid_timestamps

        # Check total within 60s window
        if len(valid_timestamps) >= self.rpm:
            oldest = valid_timestamps[0]
            retry_after = max(1, int(self.window_seconds - (now - oldest)))
            return False, retry_after

        # Check burst within last 5 seconds
        burst_cutoff = now - 5.0
        burst_count = sum(1 for t in valid_timestamps if t > burst_cutoff)
        if burst_count >= self.burst:
            return False, 5

        valid_timestamps.append(now)
        return True, 0

    def cleanup_idle(self, idle_seconds: float = 300.0):
        """Cleans up stale client tracking entries to prevent memory leaks."""
        now = time.time()
        dead_keys = [k for k, v in self.history.items() if not v or (now - v[-1]) > idle_seconds]
        for k in dead_keys:
            del self.history[k]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware applying sliding-window rate limiting per client IP / API key."""

    def __init__(self, app, requests_per_minute: int = 60, burst_limit: int = 15):
        super().__init__(app)
        self.limiter = SlidingWindowRateLimiter(requests_per_minute, burst_limit)

    async def dispatch(self, request: Request, call_next):
        # Allow health checks without rate limiting
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # Identify client by API Key header or Client IP
        client_key = (
            request.headers.get("x-api-key")
            or request.headers.get("authorization")
            or request.client.host
            if request.client
            else "unknown_client"
        )

        allowed, retry_after = self.limiter.is_allowed(client_key)
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "type": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Maximum {self.limiter.rpm} requests per minute allowed.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        response = await call_next(request)
        return response
