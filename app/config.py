"""Runtime settings, read from environment variables (and an optional .env file).

Every setting has a working default, so the application runs with no .env at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root if it exists. Real environment variables win.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    max_upload_mb: float
    max_megapixels: float
    data_dir: Path
    gemini_api_key: str | None
    gemini_model: str
    vision_timeout_s: float
    check_timeout_s: float
    detector_dir: Path

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def max_pixels(self) -> int:
        return int(self.max_megapixels * 1_000_000)

    @property
    def history_db_path(self) -> Path:
        return self.data_dir / "history.db"


def get_settings() -> Settings:
    """Read settings on each call so tests (and restarts) pick up changes."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key.lower() in {"", "your_key_here", "your_gemini_api_key_here"}:
        key = None
    return Settings(
        max_upload_mb=_float("MAX_UPLOAD_MB", 25),
        max_megapixels=_float("MAX_MEGAPIXELS", 80),
        data_dir=_path("DATA_DIR", PROJECT_ROOT / "data"),
        gemini_api_key=key,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash",
        vision_timeout_s=_float("VISION_TIMEOUT_SECONDS", 30),
        check_timeout_s=_float("CHECK_TIMEOUT_SECONDS", 30),
        detector_dir=_path("DETECTOR_DIR", PROJECT_ROOT / "models"),
    )
