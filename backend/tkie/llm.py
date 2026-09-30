"""
tkie/llm.py
Ollama LLM client — constructs and dispatches all three TKIE prompts.

Prompt 1 — Completeness Check:
    Determines if essential fields (company, date, total) are present.
    Returns True/False.

Prompt 2 — Few-Shot Key Information Extraction:
    Extracts structured JSON from the OCR text using few-shot examples.

Prompt 3 — JSON Refinement:
    Forces the LLM to reformat a malformed output into the exact target schema.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tkie.config import (
    JSON_TEMPLATE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class OllamaLLMClient:
    """Thin wrapper around the Ollama Python client with TKIE-specific prompts."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        try:
            import ollama  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "ollama is required. Install with: uv add ollama"
            ) from exc

        self._ollama = ollama
        self.model = model
        self.base_url = base_url
        self._client = ollama.Client(host=base_url)
        logger.info("OllamaLLMClient initialised (model=%s, host=%s)", model, base_url)

    # ------------------------------------------------------------------
    # Public prompt methods
    # ------------------------------------------------------------------

    def check_completeness(self, text: str) -> bool:
        """Prompt 1 — Completeness Check.

        Asks the LLM whether the OCR text contains the essential fields
        (company name, date, total).  Returns True if complete, False otherwise.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document analysis assistant. "
                    "Your job is to determine whether a piece of text extracted "
                    "from a business document contains sufficient information."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Given the following text extracted from a document via OCR, "
                    "determine if it contains ALL of the following essential fields:\n"
                    "  1. Company name (the issuing company)\n"
                    "  2. Document date\n"
                    "  3. Total amount\n\n"
                    "Reply with ONLY the single word 'True' or 'False'. "
                    "Do not include any explanation, punctuation, or extra text.\n\n"
                    f"Text:\n{text}"
                ),
            },
        ]
        raw = self._chat(messages).strip()
        logger.debug("Completeness check raw response: %r", raw)

        # Robust parsing — handle leading/trailing whitespace or mixed case
        if raw.lower().startswith("true"):
            return True
        if raw.lower().startswith("false"):
            return False

        # If the model hallucinated, default to False (conservative)
        logger.warning(
            "Unexpected completeness response %r — defaulting to False", raw
        )
        return False

    def extract_information(
        self,
        text: str,
        template: dict[str, Any] | None = None,
        few_shot_examples: list[dict] | None = None,
    ) -> str:
        """Prompt 2 — Few-Shot Key Information Extraction.

        Returns the raw LLM string output (may or may not be valid JSON —
        the validator handles that).
        """
        template = template or JSON_TEMPLATE
        template_str = json.dumps(template, indent=2)

        # Build few-shot section
        examples_section = ""
        if few_shot_examples:
            parts: list[str] = []
            for i, ex in enumerate(few_shot_examples, start=1):
                parts.append(
                    f"--- Example {i} ---\n"
                    f"Input text:\n{ex['input']}\n\n"
                    f"Expected output:\n{json.dumps(ex['output'], indent=2)}"
                )
            examples_section = "\n\n".join(parts) + "\n\n"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a key information extraction expert. "
                    "Your sole task is to extract structured data from business document text "
                    "and return it as a single valid JSON object. "
                    "You MUST output ONLY the JSON object — no markdown fences, no explanations, "
                    "no surrounding text. "
                    "If a field cannot be found in the text, set its value to null."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Extract the key information from the document text below.\n"
                    "Your output MUST be a single JSON object that exactly matches "
                    "the following template (same keys, same structure):\n\n"
                    f"{template_str}\n\n"
                    f"{examples_section}"
                    "Now extract from the following document text:\n"
                    f"{text}\n\n"
                    "Output ONLY the JSON object:"
                ),
            },
        ]
        raw = self._chat(messages)
        logger.debug("Extraction raw response length: %d chars", len(raw))
        return raw

    def reformat_output(
        self,
        malformed_output: str,
        template: dict[str, Any] | None = None,
    ) -> str:
        """Prompt 3 — JSON Refinement.

        Passes the malformed LLM output back with strict instructions to
        reformat it into the exact target schema.  Returns the raw output.
        """
        template = template or JSON_TEMPLATE
        template_str = json.dumps(template, indent=2)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON formatter. "
                    "You receive malformed or non-conforming output and reformat it "
                    "into a valid JSON object that exactly matches the provided schema. "
                    "Output ONLY the corrected JSON object — no markdown, no explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    "The following output does NOT conform to the required JSON schema. "
                    "Reformat it to EXACTLY match this template "
                    "(preserve all extracted values; only fix the structure and formatting):\n\n"
                    f"Required template:\n{template_str}\n\n"
                    f"Malformed output:\n{malformed_output}\n\n"
                    "Output ONLY the corrected JSON object:"
                ),
            },
        ]
        raw = self._chat(messages)
        logger.debug("Refinement raw response length: %d chars", len(raw))
        return raw

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chat(self, messages: list[dict]) -> str:
        """Send a messages list to Ollama and return the response text."""
        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": OLLAMA_TEMPERATURE,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        )
        return response["message"]["content"]
