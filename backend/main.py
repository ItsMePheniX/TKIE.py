"""
backend/main.py
FastAPI server — REST API for the LLM-TKIE extraction pipeline.

Endpoints:
  POST /extract          — upload an image, get extracted JSON back
  GET  /health           — liveness check
  GET  /bad-samples      — list quarantined documents
"""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

# Load .env from repo root (one level above backend/)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from tkie.config import (
    ALLOWED_IMAGE_FORMATS,
    BAD_SAMPLE_DIR,
    MAX_UPLOAD_BYTES,
    MONGODB_DATABASE,
    MONGODB_TIMEOUT_MS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
)
from tkie.database import MongoStore
from tkie.exceptions import JSONFormattingError, LLMError, OCRError
from tkie.extractor import TKIEExtractor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM-TKIE API",
    description=(
        "Large Language Model Driven Transferable Key Information Extraction. "
        "Upload a document image and receive structured JSON output."
    ),
    version="0.1.0",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singleton extractor (initialised on startup)
# ---------------------------------------------------------------------------
_extractor: TKIEExtractor | None = None
_mongo_store: MongoStore | None = None
_mongodb_configured = False


def _read_and_verify_upload(file: UploadFile) -> tuple[bytes, str]:
    """Read a bounded image upload and return its bytes plus safe suffix."""
    declared_type = file.content_type or ""
    allowed_types = {"image/jpeg", "image/png", "image/tiff", "image/webp", "image/bmp"}
    if declared_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a JPEG, PNG, TIFF, WebP, or BMP image.",
        )

    chunks: list[bytes] = []
    size = 0
    while chunk := file.file.read(1024 * 1024):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
            )
        chunks.append(chunk)
    payload = b"".join(chunks)
    if not payload:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                image.verify()
                image_format = image.format
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
    ):
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid, safe image.")

    suffix = ALLOWED_IMAGE_FORMATS.get(image_format or "")
    if suffix is None:
        raise HTTPException(status_code=415, detail="The image format is not supported.")
    return payload, suffix


@app.on_event("startup")
async def startup() -> None:
    global _extractor, _mongo_store, _mongodb_configured
    model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    try:
        timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", str(OLLAMA_TIMEOUT_SECONDS)))
    except ValueError as exc:
        raise RuntimeError("OLLAMA_TIMEOUT_SECONDS must be a positive number.") from exc
    if timeout <= 0:
        raise RuntimeError("OLLAMA_TIMEOUT_SECONDS must be a positive number.")
    logger.info("Initialising TKIEExtractor (model=%s) …", model)
    _extractor = TKIEExtractor(
        ollama_model=model,
        ollama_base_url=base_url,
        ollama_timeout=timeout,
    )
    logger.info("TKIEExtractor ready.")

    mongo_uri = os.getenv("MONGODB_URI", "").strip()
    _mongodb_configured = bool(mongo_uri)
    if not mongo_uri:
        logger.info("MongoDB persistence disabled (MONGODB_URI is not configured).")
        return

    database = os.getenv("MONGODB_DATABASE", MONGODB_DATABASE)
    try:
        mongo_timeout = int(os.getenv("MONGODB_TIMEOUT_MS", str(MONGODB_TIMEOUT_MS)))
        if mongo_timeout <= 0:
            raise ValueError
        store = MongoStore(mongo_uri, database, mongo_timeout)
        store.connect()
        _mongo_store = store
        logger.info("MongoDB persistence ready (database=%s).", database)
    except (RuntimeError, ValueError):
        _mongo_store = None
        logger.exception("MongoDB persistence is unavailable; extraction will continue without storage.")


@app.on_event("shutdown")
async def shutdown() -> None:
    global _mongo_store
    if _mongo_store is not None:
        _mongo_store.close()
        _mongo_store = None


def _record_extraction(
    *,
    filename: str | None,
    status: str,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    """Persist an audit record without allowing persistence outages to break extraction."""
    if _mongo_store is None:
        return
    try:
        _mongo_store.save_extraction(
            filename=filename,
            status=status,
            model=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL),
            result=result,
            error=error,
        )
    except Exception:
        logger.exception("Could not save extraction audit record to MongoDB.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health() -> dict:
    """Liveness probe that does not depend on OCR or Ollama availability."""
    return {
        "status": "ok",
        "model": os.getenv("OLLAMA_MODEL", OLLAMA_MODEL),
        "mongo_connected": _mongo_store is not None,
    }


@app.get("/ready", tags=["System"])
def ready() -> dict:
    """Readiness probe; requires the local LLM service and model to be available."""
    if _extractor is None:
        raise HTTPException(status_code=503, detail="Extractor not initialised yet.")
    if _mongodb_configured and _mongo_store is None:
        raise HTTPException(status_code=503, detail="MongoDB persistence is not ready.")
    if not _extractor.is_ready():
        raise HTTPException(status_code=503, detail="Configured Ollama model is not ready.")
    return {"status": "ready", "model": os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)}


@app.post("/extract", tags=["Extraction"])
def extract(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a document image (JPEG, PNG, TIFF, WebP) and extract key fields.

    Returns the structured JSON or a 422 if the document is a bad sample,
    or 500 if JSON formatting ultimately fails.
    """
    if _extractor is None:
        raise HTTPException(status_code=503, detail="Extractor not initialised yet.")

    filename = file.filename
    payload, suffix = _read_and_verify_upload(file)

    # PaddleOCR needs a file path or ndarray. The suffix comes from verified
    # image content, never from the untrusted filename.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)

    try:
        result = _extractor.process(tmp_path)
    except JSONFormattingError as exc:
        logger.error("JSON formatting error: %s", exc)
        _record_extraction(filename=filename, status="failed", error="invalid_structured_output")
        raise HTTPException(
            status_code=502,
            detail="The extraction model returned invalid structured data.",
        )
    except (OCRError, LLMError) as exc:
        logger.exception("Extraction service failure")
        _record_extraction(filename=filename, status="failed", error="extraction_service_unavailable")
        raise HTTPException(
            status_code=502,
            detail="A local extraction service is unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected extraction failure")
        _record_extraction(filename=filename, status="failed", error="unexpected_error")
        raise HTTPException(
            status_code=500,
            detail="Extraction failed unexpectedly.",
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    if result is None:
        _record_extraction(filename=filename, status="quarantined")
        return JSONResponse(
            status_code=422,
            content={
                "detail": (
                    "Document failed all completeness checks and was quarantined. "
                    "Check the bad_samples/ directory."
                )
            },
        )

    _record_extraction(filename=filename, status="succeeded", result=result)
    return JSONResponse(content=result)


@app.get("/bad-samples", tags=["System"])
async def list_bad_samples() -> dict:
    """List all quarantined (bad sample) documents."""
    bad_dir = Path(BAD_SAMPLE_DIR)
    if not bad_dir.exists():
        return {"count": 0, "samples": []}

    images = [
        {"name": p.name, "sidecar": (bad_dir / (p.stem + ".txt")).exists()}
        for p in sorted(bad_dir.iterdir())
        if p.suffix.lower() not in {".txt"}
    ]
    return {"count": len(images), "samples": images}


@app.get("/extractions", tags=["Extraction"])
def list_extractions(limit: int = Query(default=50, ge=1, le=100)) -> dict:
    """List persisted extraction audit records, newest first."""
    if _mongo_store is None:
        raise HTTPException(status_code=503, detail="MongoDB persistence is not configured or unavailable.")
    try:
        records = _mongo_store.list_extractions(limit)
    except Exception as exc:
        logger.exception("Could not list extraction audit records from MongoDB.")
        raise HTTPException(status_code=503, detail="MongoDB persistence is unavailable.") from exc
    return {"count": len(records), "extractions": records}
