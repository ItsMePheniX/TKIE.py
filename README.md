# LLM-TKIE

**Large Language Model Driven Transferable Key Information Extraction**

Extract structured JSON from non-standardised documents (invoices, receipts, purchase orders) using a decoupled OCR + LLM pipeline.

---

## Architecture

```
TKIE.py/
├── .env                          ← runtime config (model, URL, log level)
├── frontend/                     ← vanilla HTML/CSS/JS UI
│   ├── index.html
│   ├── style.css
│   └── app.js
└── backend/                      ← Python pipeline + FastAPI server
    ├── pyproject.toml            ← uv project manifest
    ├── .python-version           ← Python 3.11
    ├── main.py                   ← FastAPI entry point
    ├── examples/
    │   └── few_shot_examples.json
    └── tkie/
        ├── config.py             ← all tuneable constants
        ├── preprocessor.py       ← alpha strip, resize, rotate
        ├── ocr.py                ← DBNet++ + SVTR_LCNet, conf ≥ 0.9
        ├── llm.py                ← Prompts 1, 2, 3 (Ollama)
        ├── validator.py          ← regex JSON extractor + schema check
        ├── extractor.py          ← orchestrator (rotation loop + refinement)
        ├── bad_sample.py         ← quarantine failed documents
        └── exceptions.py        ← custom exception hierarchy
```

---

## Pipeline (Algorithm)

```
image
  │
  ▼  Preprocess (alpha strip → resize 1920px)
  │
  ▼  Rotation loop [0°, 90°, 180°, 270°]
  │    └─ OCR (DBNet++ detect, SVTR_LCNet recognise, conf ≥ 0.9)
  │    └─ Prompt 1: completeness check → True / False
  │         ✓ pass → proceed
  │         ✗ all fail → Bad Sample Repository
  │
  ▼  Prompt 2: few-shot extraction → raw JSON string
  │
  ▼  Regex validation
  │    ✓ valid → return dict
  │    ✗ invalid ──▶  Prompt 3: refinement → validate again
  │                       ✓ valid → return dict
  │                       ✗ invalid → JSONFormattingError
```

---

## Setup

### 1. Prerequisites

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Ollama
brew install ollama           # macOS
ollama pull llama3:8b         # or: ollama pull qwen2:7b
```

### 2. Backend

```bash
cd backend
uv sync                       # creates .venv and installs all deps
```

### 3. Configure

Edit `.env` in the repo root:

```env
OLLAMA_MODEL=llama3:8b
OLLAMA_BASE_URL=http://localhost:11434
LOG_LEVEL=INFO
```

### 4. Run the API server

```bash
# Make sure Ollama is running first:
ollama serve

# In backend/ directory:
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the frontend

Open `frontend/index.html` directly in your browser, or serve it:

```bash
cd frontend
python -m http.server 3000
# → http://localhost:3000
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/extract` | Upload image → extracted JSON |
| `GET`  | `/health`  | API liveness + active model |
| `GET`  | `/bad-samples` | List quarantined documents |

**Example:**
```bash
curl -X POST http://localhost:8000/extract \
     -F "file=@invoice.jpg"
```

---

## Customising the JSON Schema

Edit `JSON_TEMPLATE` in [`backend/tkie/config.py`](backend/tkie/config.py) and update the few-shot examples in [`backend/examples/few_shot_examples.json`](backend/examples/few_shot_examples.json) to match your document domain.

---

## Models

| Component | Model | Notes |
|-----------|-------|-------|
| Text detection | DBNet++ | Auto-downloaded by PaddleOCR |
| Text recognition | SVTR_LCNet | Auto-downloaded by PaddleOCR |
| LLM | LLaMA3-8B (Q4_0) | Via Ollama — swap to `qwen2:7b` in `.env` |
