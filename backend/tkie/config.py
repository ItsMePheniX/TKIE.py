"""
tkie/config.py
Centralized configuration for the LLM-TKIE pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# OCR settings
# ---------------------------------------------------------------------------
OCR_LANGUAGE: str = "en"
DET_CONFIDENCE_THRESHOLD: float = 0.9
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
