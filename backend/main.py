"""
backend/main.py
FastAPI server — REST API for the LLM-TKIE extraction pipeline.

Endpoints:
  POST /extract          — upload an image, get extracted JSON back
  GET  /health           — liveness check
  GET  /bad-samples      — list quarantined documents
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load .env from repo root (one level above backend/)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from tkie.config import BAD_SAMPLE_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL
from tkie.exceptions import JSONFormattingError
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singleton extractor (initialised on startup)
# ---------------------------------------------------------------------------
_extractor: TKIEExtractor | None = None


@app.on_event("startup")
async def startup() -> None:
    global _extractor
    model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    logger.info("Initialising TKIEExtractor (model=%s) …", model)
    _extractor = TKIEExtractor(ollama_model=model, ollama_base_url=base_url)
    logger.info("TKIEExtractor ready.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "model": os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)}


@app.post("/extract", tags=["Extraction"])
async def extract(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a document image (JPEG, PNG, TIFF, WebP) and extract key fields.

    Returns the structured JSON or a 422 if the document is a bad sample,
    or 500 if JSON formatting ultimately fails.
    """
    if _extractor is None:
        raise HTTPException(status_code=503, detail="Extractor not initialised yet.")

    # Accept common image types
    allowed = {"image/jpeg", "image/png", "image/tiff", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed}",
        )

    # Save upload to a temp file (PaddleOCR needs a file path or ndarray)
    import tempfile, shutil
    suffix = Path(file.filename or "upload").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = _extractor.process(tmp_path)
    except JSONFormattingError as exc:
        logger.error("JSON formatting error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    if result is None:
        return JSONResponse(
            status_code=422,
            content={
                "detail": (
                    "Document failed all completeness checks and was quarantined. "
                    "Check the bad_samples/ directory."
                )
            },
        )

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
