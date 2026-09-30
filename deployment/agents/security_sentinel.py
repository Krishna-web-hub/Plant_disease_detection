"""Agent 1: Security & Input Sentinel.

Enforces Principle 3 of Vibe Coding Security (Strict Server-Side Validation):
- Validates file magic bytes (real JPEG, PNG, WEBP signatures)
- Rejects decompression bombs (PIL MAX_IMAGE_PIXELS guard)
- Enforces strict file size limits (max 10MB)
- Sanitizes image by stripping EXIF metadata to prevent privacy leaks
- Validates visual integrity (detects corrupt or blank frames)
"""
import io
import logging
from typing import Tuple
from PIL import Image

logger = logging.getLogger(__name__)

# Max upload size: 10 MB
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Decompression bomb limit: 10 megapixels
MAX_IMAGE_PIXELS = 10_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Known magic byte signatures
MAGIC_SIGNATURES = {
    "jpeg": b"\xff\xd8\xff",
    "png": b"\x89PNG\r\n\x1a\n",
    "webp_riff": b"RIFF",
    "webp_tag": b"WEBP",
}


class SecuritySentinelAgent:
    """Gatekeeper agent validating raw byte streams before machine learning ingestion."""

    def __init__(self, max_size_bytes: int = MAX_FILE_SIZE_BYTES):
        self.max_size = max_size_bytes

    def inspect(self, raw_bytes: bytes) -> Tuple[bool, str, Image.Image | None]:
        """Performs multi-layer security inspection on uploaded payload.

        Returns:
            (is_valid, reason, sanitized_pil_image)
        """
        # 1. Size Validation
        byte_len = len(raw_bytes)
        if byte_len == 0:
            return False, "Security Error: Empty payload received.", None
        if byte_len > self.max_size:
            return (
                False,
                f"Security Error: Payload size ({byte_len / 1024 / 1024:.2f}MB) exceeds 10MB limit.",
                None,
            )

        # 2. Magic Bytes Inspection (Server-side validation)
        is_jpeg = raw_bytes.startswith(MAGIC_SIGNATURES["jpeg"])
        is_png = raw_bytes.startswith(MAGIC_SIGNATURES["png"])
        is_webp = raw_bytes.startswith(MAGIC_SIGNATURES["webp_riff"]) and (
            len(raw_bytes) > 12 and raw_bytes[8:12] == MAGIC_SIGNATURES["webp_tag"]
        )

        if not (is_jpeg or is_png or is_webp):
            return (
                False,
                "Security Error: Invalid file format signature. Only genuine JPEG, PNG, or WebP images are permitted.",
                None,
            )

        # 3. Decompression & PIL Parse
        try:
            raw_img = Image.open(io.BytesIO(raw_bytes))
            raw_img.verify()  # Verifies file integrity without decoding whole image
        except Image.DecompressionBombError as e:
            return False, f"Security Error: Decompression bomb detected ({e}).", None
        except Exception as e:
            return False, f"Security Error: Corrupt image payload detected ({e}).", None

        # Reopen for decoding (verify() invalidates image buffer)
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            w, h = img.size
            if w * h > MAX_IMAGE_PIXELS:
                return (
                    False,
                    f"Security Error: Decompression bomb detected ({w}x{h} = {w * h} pixels exceeds {MAX_IMAGE_PIXELS}).",
                    None,
                )
            if w < 32 or h < 32:
                return False, f"Security Error: Image dimensions ({w}x{h}) are too small for diagnostic analysis.", None

            # 4. EXIF Stripping & Sanitization
            # Rebuilding a clean RGB image drops all metadata (GPS, camera serials, scripts)
            sanitized = Image.new("RGB", img.size)
            sanitized.paste(img.convert("RGB"))

            # 5. Visual Integrity Check (Reject completely flat/blank images)
            extrema = sanitized.convert("L").getextrema()
            if extrema[0] == extrema[1]:
                return False, "Input Error: Uploaded image is entirely uniform/blank.", None

            return True, "Security validation passed.", sanitized

        except Exception as e:
            return False, f"Security Error: Failed to safely decode image ({e}).", None
