"""
tkie/config.py
Centralized configuration for the LLM-TKIE pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# OCR settings
# ---------------------------------------------------------------------------
OCR_LANGUAGE: str = "en"
# PaddleOCR does not expose a per-box detection confidence from its combined
# ``ocr()`` call. This is the DBNet box acceptance threshold passed directly
# to PaddleOCR; recognition confidence is filtered separately below.
DET_BOX_THRESHOLD: float = 0.6
REC_CONFIDENCE_THRESHOLD: float = 0.9

# Maximum pixel length of the longest image side before OCR (preserves aspect ratio)
IMAGE_MAX_SIDE: int = 1920

# ---------------------------------------------------------------------------
# Rotation strategy
# ---------------------------------------------------------------------------
ROTATIONS: list[int] = [0, 90, 180, 270]

# ---------------------------------------------------------------------------
# Ollama / LLM settings
# ---------------------------------------------------------------------------
OLLAMA_MODEL: str = "llama3:8b"          # swap to "qwen2:7b" as needed
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_TEMPERATURE: float = 0.0           # deterministic for structured extraction
OLLAMA_NUM_CTX: int = 8192
OLLAMA_TIMEOUT_SECONDS: float = 60.0

# ---------------------------------------------------------------------------
# MongoDB settings (configured through the environment at runtime)
# ---------------------------------------------------------------------------
MONGODB_DATABASE: str = "tkie"
MONGODB_TIMEOUT_MS: int = 5_000

# ---------------------------------------------------------------------------
# API upload settings
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
ALLOWED_IMAGE_FORMATS: dict[str, str] = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
    "BMP": ".bmp",
}

# ---------------------------------------------------------------------------
# Output / storage
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent   # backend/tkie → backend → repo root
BACKEND_DIR: Path = Path(__file__).parent.parent           # backend/tkie → backend

BAD_SAMPLE_DIR: Path = BACKEND_DIR / "bad_samples"
FEW_SHOT_PATH: Path = BACKEND_DIR / "examples" / "few_shot_examples.json"

# ---------------------------------------------------------------------------
# Target JSON schema template
# Fields here define the EXPECTED output keys the LLM must fill.
# Adjust to match your document domain.
# ---------------------------------------------------------------------------
JSON_TEMPLATE: dict = {
    "company_name": None,
    "document_date": None,
    "document_number": None,
    "vendor_name": None,
    "line_items": [],       # list of {"description": ..., "quantity": ..., "unit_price": ..., "amount": ...}
    "subtotal": None,
    "tax": None,
    "total": None,
    "currency": None,
    "payment_terms": None,
}

# JSON-schema-like contract enforced after the LLM response is parsed. Keep
# this in sync with JSON_TEMPLATE when adapting the extractor to a new domain.
JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "company_name": {"type": ["string", "null"]},
        "document_date": {"type": ["string", "null"]},
        "document_number": {"type": ["string", "null"]},
        "vendor_name": {"type": ["string", "null"]},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": ["string", "null"]},
                    "quantity": {"type": ["number", "null"]},
                    "unit_price": {"type": ["number", "null"]},
                    "amount": {"type": ["number", "null"]},
                },
                "required": ["description", "quantity", "unit_price", "amount"],
                "additionalProperties": False,
            },
        },
        "subtotal": {"type": ["number", "null"]},
        "tax": {"type": ["number", "null"]},
        "total": {"type": ["number", "null"]},
        "currency": {"type": ["string", "null"]},
        "payment_terms": {"type": ["string", "null"]},
    },
    "required": list(JSON_TEMPLATE),
    "additionalProperties": False,
}
