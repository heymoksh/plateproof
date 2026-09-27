"""HTTP API and web UI for PlateProof."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import APP_NAME, __version__
from .config import PROJECT_ROOT, get_settings
from .models import ClaimContext, ClaimReport, ComplaintType, ImageSource
from .pipeline import analyze_claim
from .validation import HEIF_SUPPORTED, ImageValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("plateproof")

STATIC_DIR = PROJECT_ROOT / "static"
SAMPLES_DIR = PROJECT_ROOT / "samples"
MAX_PHOTOS = 6

app = FastAPI(
    title=APP_NAME,
    version=__version__,
    description="Screens food-delivery refund photos for image-level evidence that deserves human review.",
)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(ImageValidationError)
async def _validation_error(_: Request, exc: ImageValidationError) -> JSONResponse:
    return _error(exc.http_status, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def _request_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    fields = {str(err["loc"][-1]) for err in exc.errors() if err.get("loc")}
    if fields & {"image", "images"}:
        return _error(400, "missing_image", "No photo was uploaded. Attach at least one image.")
    return _error(422, "invalid_request", f"Invalid value for: {', '.join(sorted(fields)) or 'request'}.")


@app.exception_handler(Exception)
async def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error: %s", exc)
    return _error(500, "internal_error", "The analysis failed unexpectedly. Details are in the server log.")


@app.get("/api/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "heic_supported": HEIF_SUPPORTED,
        "vision_configured": settings.gemini_api_key is not None,
        "vision_model": settings.gemini_model,
        "max_upload_mb": settings.max_upload_mb,
        "max_photos": MAX_PHOTOS,
        "name": APP_NAME,
    }


def _parse_time(value: str | None, field: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise ImageValidationError("invalid_" + field, f"{field.replace('_', ' ').capitalize()} must look like "
                                   "2026-09-26T19:30 (ISO 8601).", 422) from None


def build_claim(
    order_id: str | None, customer_id: str | None, restaurant: str | None, ordered_items: str | None,
    order_placed_at: str | None, delivered_at: str | None, complaint_type: str | None,
    description: str | None, image_source: str | None,
) -> ClaimContext:
    clean = lambda v: (v or "").strip() or None  # noqa: E731
    try:
        source = ImageSource(image_source) if clean(image_source) else ImageSource.UNKNOWN
    except ValueError:
        allowed = ", ".join(s.value for s in ImageSource)
        raise ImageValidationError("invalid_image_source", f"Image source must be one of: {allowed}.", 422) from None
    try:
        complaint = ComplaintType(complaint_type) if clean(complaint_type) else ComplaintType.UNKNOWN
    except ValueError:
        allowed = ", ".join(c.value for c in ComplaintType)
        raise ImageValidationError("invalid_complaint_type", f"Complaint type must be one of: {allowed}.", 422) from None
    placed = _parse_time(clean(order_placed_at), "order_placed_at")
    delivered = _parse_time(clean(delivered_at), "delivered_at")
    try:
        return ClaimContext(
            order_id=clean(order_id), customer_id=clean(customer_id), restaurant=clean(restaurant),
            ordered_items=clean(ordered_items), order_placed_at=placed, delivered_at=delivered,
            complaint_type=complaint, description=clean(description), image_source=source,
        )
    except ValidationError as exc:
        if "delivered_at is earlier" in str(exc):
            raise ImageValidationError("invalid_order_times", "The delivery time is earlier than the order time.", 422) from None
        raise ImageValidationError("invalid_claim_context", "A claim field is too long.", 422) from None


@app.post("/api/analyze", response_model=ClaimReport)
async def analyze_claim_photos(
    images: list[UploadFile] | None = File(None, description=f"1 to {MAX_PHOTOS} photos for one refund claim."),
    image: UploadFile | None = File(None, description="Single photo (alternative to 'images')."),
    order_id: str | None = Form(None),
    customer_id: str | None = Form(None),
    restaurant: str | None = Form(None),
    ordered_items: str | None = Form(None),
    order_placed_at: str | None = Form(None, description="ISO 8601, e.g. 2026-09-26T19:30"),
    delivered_at: str | None = Form(None, description="ISO 8601, e.g. 2026-09-26T20:05"),
    complaint_type: str | None = Form(None),
    description: str | None = Form(None),
    image_source: str | None = Form(None),
    save_to_history: bool = Form(True),
    use_vision: bool = Form(True, description="Send the photos to Gemini if configured."),
) -> ClaimReport:
    settings = get_settings()
    uploads = [u for u in [*(images or []), image] if u is not None]
    if not uploads:
        raise ImageValidationError("missing_image", "No photo was uploaded. Attach at least one image.", 400)
    if len(uploads) > MAX_PHOTOS:
        raise ImageValidationError("too_many_photos", f"A claim can have at most {MAX_PHOTOS} photos.", 422)
    claim = build_claim(order_id, customer_id, restaurant, ordered_items, order_placed_at, delivered_at,
                        complaint_type, description, image_source)
    files = [(await u.read(settings.max_upload_bytes + 1), u.filename, u.content_type) for u in uploads]
    return await analyze_claim(files, claim, save_to_history, settings, use_vision)


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if SAMPLES_DIR.is_dir():  # lets the UI offer "try a sample"
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.get("/", include_in_schema=False)
async def index():
    page = STATIC_DIR / "index.html"
    if page.is_file():
        return FileResponse(page)
    return {"message": f"{APP_NAME} API. See /docs for the interactive API."}
