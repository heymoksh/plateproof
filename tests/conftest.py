from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.checks import AnalysisContext
from app.config import get_settings
from app.models import ClaimContext
from app.validation import validate_and_decode


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets its own data directory and no API keys or models."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DETECTOR_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    yield


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def make_ctx(data: bytes, filename: str = "photo.jpg", content_type: str | None = None, **claim) -> AnalysisContext:
    settings = get_settings()
    decoded = validate_and_decode(data, filename, content_type, settings.max_upload_bytes, settings.max_pixels)
    return AnalysisContext(decoded=decoded, claim=ClaimContext(**claim), settings=settings)


def first_photo(body: dict) -> dict:
    """Per-photo report from a claim response (single-photo claims)."""
    assert "photos" in body, body
    return body["photos"][0]
