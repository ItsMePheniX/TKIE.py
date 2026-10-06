"""
tkie/extractor.py
Top-level orchestrator — implements the full LLM-TKIE Algorithm.

Algorithm:
  1. Preprocess image (alpha strip, resize)
  2. For each rotation in [0°, 90°, 180°, 270°]:
       a. OCR → unified text block
       b. Prompt 1 completeness check → True/False
       c. Break on first True
  3. If no rotation passed → quarantine to BadSampleRepository, return None
  4. Prompt 2 few-shot extraction → raw LLM output
  5. Regex validate output
     - If valid  → return parsed dict
     - If invalid → Prompt 3 refinement → validate again
       - If still invalid → raise JSONFormattingError
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tkie.bad_sample import BadSampleRepository
from tkie.config import FEW_SHOT_PATH, JSON_TEMPLATE, ROTATIONS
from tkie.exceptions import JSONFormattingError
from tkie.llm import OllamaLLMClient
from tkie.ocr import OCRPipeline
from tkie.preprocessor import ImagePreprocessor
from tkie.validator import JSONValidator

logger = logging.getLogger(__name__)


class TKIEExtractor:
    """Full LLM-TKIE pipeline orchestrator."""

    def __init__(
        self,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
        ollama_timeout: float | None = None,
        json_template: dict[str, Any] | None = None,
        few_shot_path: Path | str | None = None,
        bad_sample_dir: Path | str | None = None,
    ) -> None:
        self.template = json_template or JSON_TEMPLATE

        # Lazy OCR initialisation — PaddleOCR downloads models on first use
        self._preprocessor = ImagePreprocessor()
        self._ocr: OCRPipeline | None = None          # initialised on first process()
        self._llm = OllamaLLMClient(
            **{k: v for k, v in {
                "model": ollama_model,
                "base_url": ollama_base_url,
                "timeout": ollama_timeout,
            }.items() if v is not None}
        )
        self._validator = JSONValidator(self.template)
        self._bad_repo = BadSampleRepository(
            **({"output_dir": bad_sample_dir} if bad_sample_dir else {})
        )

        # Load few-shot examples
        self._few_shot_examples = self._load_few_shot(few_shot_path or FEW_SHOT_PATH)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image_path: str | Path) -> dict[str, Any] | None:
        """Run the full TKIE pipeline on a single image.

        Args:
            image_path: Path to the input document image.

        Returns:
            Extracted data as a dict, or None if the document was quarantined.

        Raises:
            JSONFormattingError: If the LLM fails to produce valid JSON
                                 after the refinement loop.
        """
        image_path = Path(image_path)
        logger.info("Processing: %s", image_path.name)

        # Lazy-initialise OCR (downloads PaddleOCR models on first call)
        if self._ocr is None:
            self._ocr = OCRPipeline()

        # ----------------------------------------------------------------
        # Step 1 — Preprocess
        # ----------------------------------------------------------------
        img_array = self._preprocessor.preprocess(image_path)

        # ----------------------------------------------------------------
        # Step 2 — Rotation loop + Completeness Check (Prompt 1)
        # ----------------------------------------------------------------
        best_text: str | None = None

        for degrees in ROTATIONS:
            rotated = self._preprocessor.rotate(img_array, degrees)
            text = self._ocr.extract_text(rotated)

            if not text.strip():
                logger.debug("Rotation %d°: OCR returned empty text, skipping.", degrees)
                continue

            logger.info("Rotation %d°: running completeness check …", degrees)
            is_complete = self._llm.check_completeness(text)

            if is_complete:
                logger.info("Rotation %d°: completeness check PASSED.", degrees)
                best_text = text
                break
            else:
                logger.info("Rotation %d°: completeness check failed.", degrees)

        # ----------------------------------------------------------------
        # Step 3 — Bad sample if all rotations failed
        # ----------------------------------------------------------------
        if best_text is None:
            reason = "completeness_check_failed_all_rotations"
            logger.warning("All rotations failed for %s → quarantining.", image_path.name)
            self._bad_repo.save(image_path, reason)
            return None

        # ----------------------------------------------------------------
        # Step 4 — Few-Shot Extraction (Prompt 2)
        # ----------------------------------------------------------------
        logger.info("Running few-shot extraction …")
        raw_output = self._llm.extract_information(
            text=best_text,
            template=self.template,
            few_shot_examples=self._few_shot_examples,
        )

        # ----------------------------------------------------------------
        # Step 5 — Validation (first pass)
        # ----------------------------------------------------------------
        valid, parsed = self._validator.validate(raw_output)

        if valid and parsed is not None:
            logger.info("Extraction validated on first pass.")
            return parsed

        # ----------------------------------------------------------------
        # Step 6 — Refinement Loop (Prompt 3)
        # ----------------------------------------------------------------
        logger.warning("First validation failed — invoking Prompt 3 refinement …")
        refined_output = self._llm.reformat_output(raw_output, self.template)

        valid2, parsed2 = self._validator.validate(refined_output)

        if valid2 and parsed2 is not None:
            logger.info("Extraction validated after refinement.")
            return parsed2

        # ----------------------------------------------------------------
        # Step 7 — Hard failure
        # ----------------------------------------------------------------
        raise JSONFormattingError(
            f"LLM failed to produce valid JSON for '{image_path.name}' "
            "after Prompt 2 extraction and Prompt 3 refinement.\n"
            f"Last raw output:\n{refined_output}"
        )

    def is_ready(self) -> bool:
        """Return whether the configured local LLM is reachable and loaded."""
        return self._llm.is_available()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_few_shot(path: Path | str) -> list[dict]:
        """Load few-shot examples from a JSON file.  Returns [] on failure."""
        p = Path(path)
        if not p.exists():
            logger.warning("Few-shot examples file not found: %s", p)
            return []
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning("Few-shot file must contain a JSON array.")
                return []
            logger.info("Loaded %d few-shot examples from %s", len(data), p.name)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load few-shot examples: %s", exc)
            return []
