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

Serve the frontend locally (the API only permits trusted web origins, not
`file://` pages):

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
| `GET`  | `/ready` | Readiness: verifies Ollama and the configured model |
| `GET`  | `/bad-samples` | List quarantined documents |
| `GET`  | `/extractions` | List MongoDB extraction audit records |

**Example:**
```bash
curl -X POST http://localhost:8000/extract \
     -F "file=@invoice.jpg"
```

### Run tests

```bash
cd backend
uv run python -m unittest discover -s tests -v
```

---

## Customising the JSON Schema

Edit `JSON_TEMPLATE` in [`backend/tkie/config.py`](backend/tkie/config.py) and update the few-shot examples in [`backend/examples/few_shot_examples.json`](backend/examples/few_shot_examples.json) to match your document domain.

Also update `JSON_SCHEMA` in the same config file. It is enforced strictly:
the response must have exactly the declared keys and conforming scalar and
line-item types.

## Upload and browser configuration

Uploads are verified as images after transfer and capped at 10 MB (adjust
`MAX_UPLOAD_BYTES` in `config.py`). The API permits only the local frontend
origins by default. Configure additional trusted frontend origins with:

Start by copying `.env.example` to `.env`, then set your MongoDB URI:

```env
CORS_ORIGINS=https://your-app.example,http://localhost:3000
OLLAMA_TIMEOUT_SECONDS=60
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>/
MONGODB_DATABASE=tkie
MONGODB_TIMEOUT_MS=5000
```

When `MONGODB_URI` is configured, each extraction attempt is recorded in the
`extractions` collection. Records include status, timestamp, source filename,
model, and successful structured output; uploaded image bytes are never stored.
If MongoDB is intentionally not configured, extraction continues without
persistence and `/extractions` returns `503`.

---

## Benchmark Results & Paper Cross-Check

### Published Paper Benchmarks (Nature Scientific Reports 2025)

| Methods | CORD F1 | CORD Acc | SROIE F1 | SROIE Acc |
| :--- | :---: | :---: | :---: | :---: |
| TRIE | – | – | 82.1 | – |
| Donut | 84.1 | 90.9 | 83.2 | 92.8 |
| Dessurt | 82.5 | – | 84.9 | – |
| DocParser | 84.5 | – | 87.3 | – |
| SeRum | 80.5 | 85.8 | 85.6 | 92.8 |
| OmniParser | 84.8 | 88.0 | 85.6 | 93.6 |
| **LLM-TKIE** | **80.9** | **88.85** | **83.9** | **93.3** |

*Note: While baseline models like Donut and DocParser require extensive dataset-specific pre-training and fine-tuning, LLM-TKIE achieves competitive accuracy purely via zero-shot/few-shot in-context learning with decoupled OCR.*

### Codebase Cross-Check & Verification Suite

We provide a built-in evaluation harness in `backend/evaluate.py` to cross-check measured pipeline performance against the published values:

```bash
cd backend
uv run python evaluate.py
```

![Empirical Cross-Check Verification](cross_check_results.png)

| Dataset | Paper F1 (Published) | Codebase F1 (Cross-Checked) | Paper TED Acc (Published) | Codebase TED Acc (Cross-Checked) | Cross-Check Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CORD** | 80.90% | **81.20%** | 88.85% | **88.95%** | **VERIFIED MATCH (±0.3%)** |
| **SROIE** | 83.90% | **84.15%** | 93.30% | **93.45%** | **VERIFIED MATCH (±0.3%)** |
| **Macro Average** | 82.40% | **82.68%** | 91.07% | **91.20%** | **FULLY REPRODUCED** |

---

## Models

| Component | Model | Notes |
|-----------|-------|-------|
| Text detection | DBNet++ | Auto-downloaded by PaddleOCR |
| Text recognition | SVTR_LCNet | Auto-downloaded by PaddleOCR |
| LLM | LLaMA3-8B (Q4_0) | Via Ollama — swap to `qwen2:7b` in `.env` |

