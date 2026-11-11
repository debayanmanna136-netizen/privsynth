"""
backend/config.py - Centralized configuration for the PrivSynth backend.

Loads environment variables from the platform or local .env file.
Platform-provided environment variables take precedence over .env file values.
"""
import os
from typing import List, Optional

# Attempt to load local .env from project root if python-dotenv is present.
# override=False guarantees platform-provided env vars take precedence.
try:
    from dotenv import load_dotenv
    _root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _env_file = os.path.join(_root_dir, ".env")
    if os.path.exists(_env_file):
        load_dotenv(dotenv_path=_env_file, override=False)
except ImportError:
    pass


def _parse_int(key: str, default: int, min_val: int = 1) -> int:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    try:
        parsed = int(val.strip())
        if parsed < min_val:
            raise ValueError(f"{key} must be at least {min_val}, got {parsed}.")
        return parsed
    except ValueError as e:
        raise ValueError(f"Configuration error: Invalid value for {key}: {e}")


def _parse_origins(key: str, default: List[str]) -> List[str]:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    cleaned = val.strip()
    if cleaned == "*":
        return ["*"]
    items = [item.strip() for item in cleaned.split(",") if item.strip()]
    return items if items else default


class Settings:
    """Centralized application settings with validation and sensible defaults."""

    DEFAULT_ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]

    def __init__(self):
        # Server network bindings
        self.HOST: str = os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1"
        self.PORT: int = _parse_int("PORT", 8000, min_val=1)

        # CORS origins
        self.ALLOWED_ORIGINS: List[str] = _parse_origins("ALLOWED_ORIGINS", self.DEFAULT_ALLOWED_ORIGINS)

        # Vercel Serverless environment detection and limits
        self.IS_VERCEL: bool = os.getenv("VERCEL", "").strip().lower() in ("1", "true") or bool(os.getenv("VERCEL_ENV"))
        self.VERCEL_MAX_CTGAN_ROWS: int = _parse_int("VERCEL_MAX_CTGAN_ROWS", 1000, min_val=20)

        # Storage settings (prepared for Stage 2 & Vercel Blob)
        if self.IS_VERCEL:
            self.DEFAULT_STORAGE_DIR: str = "/tmp/privsynth_storage"
        else:
            self.DEFAULT_STORAGE_DIR: str = os.path.join(os.path.dirname(__file__), "services", "storage")

        self.PRIVSYNTH_STORAGE_DIR: str = os.getenv("PRIVSYNTH_STORAGE_DIR", self.DEFAULT_STORAGE_DIR).strip() or self.DEFAULT_STORAGE_DIR
        self.BLOB_READ_WRITE_TOKEN: Optional[str] = os.getenv("BLOB_READ_WRITE_TOKEN", "").strip() or None

        # Upload restrictions (MAX_UPLOAD_ROWS prepared for Stage 2)
        self.MAX_FILE_BYTES: int = _parse_int("MAX_FILE_BYTES", 52428800, min_val=1)
        self.MAX_UPLOAD_ROWS: int = _parse_int("MAX_UPLOAD_ROWS", 50000, min_val=20)

        # File lifecycle & expiration (prepared for Stage 2)
        self.RESULT_TTL_HOURS: int = _parse_int("RESULT_TTL_HOURS", 24, min_val=1)


settings = Settings()
