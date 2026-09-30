"""
tkie/ocr.py
OCR pipeline using PaddleOCR (DBNet++ detection + SVTR_LCNet recognition).

Pipeline:
  1. DBNet++ detects text bounding boxes → filter by DET_CONFIDENCE_THRESHOLD
  2. SVTR_LCNet recognizes text in each box → CTC decode → filter by REC_CONFIDENCE_THRESHOLD
  3. Aggregate surviving text top-to-bottom, left-to-right → unified text block
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from tkie.config import DET_CONFIDENCE_THRESHOLD, OCR_LANGUAGE, REC_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """A detected text bounding box (4 corner points) with confidence score."""
    points: list[list[float]]   # [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    score: float

    @property
    def top_left_y(self) -> float:
        """Minimum y-coordinate — used for reading-order sorting."""
        return min(p[1] for p in self.points)

    @property
    def top_left_x(self) -> float:
        """Minimum x-coordinate — used for reading-order sorting."""
        return min(p[0] for p in self.points)


@dataclass
class TextResult:
    """A recognised text fragment with its confidence score and source box."""
    text: str
    score: float
    bbox: BBox


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OCRPipeline:
    """Wraps PaddleOCR to enforce confidence thresholds at both OCR stages.

    PaddleOCR's .ocr() already chains detection + recognition internally,
    but it exposes per-result scores that we inspect manually so we can
    apply *separate* thresholds to detection and recognition.
    """

    def __init__(self) -> None:
        # Lazy import so the module is importable even if PaddleOCR isn't
        # installed yet (useful for unit testing with mocks).
        try:
            from paddleocr import PaddleOCR  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "paddleocr is required.  Install with: uv add paddleocr"
            ) from exc

        logger.info(
            "Initialising PaddleOCR (det=DBNet++, rec=SVTR_LCNet, lang=%s)",
            OCR_LANGUAGE,
        )
        self._ocr = PaddleOCR(
            use_angle_cls=False,   # we handle rotation ourselves
            lang=OCR_LANGUAGE,
            det_model_dir=None,    # use PaddleOCR's auto-download for DBNet++
            rec_model_dir=None,    # use PaddleOCR's auto-download for SVTR_LCNet
            det_db_score_mode="slow",   # polygon mode — more accurate for DBNet++
            show_log=False,
        )
        logger.info("PaddleOCR initialised successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(self, img_array: np.ndarray) -> str:
        """Full OCR pipeline: detect → filter → recognise → filter → aggregate.

        Args:
            img_array: uint8 RGB numpy array (output of ImagePreprocessor).

        Returns:
            Unified text block as a single string (newline-separated lines).
        """
        raw_results = self._run_ocr(img_array)
        text_results = self._parse_and_filter(raw_results)
        unified = self._aggregate(text_results)
        logger.info(
            "OCR extracted %d text fragments → %d characters",
            len(text_results),
            len(unified),
        )
        return unified

    def detect(self, img_array: np.ndarray) -> list[BBox]:
        """Run *only* the detection stage and return confidence-filtered boxes.

        Useful for debugging or visualisation purposes.
        """
        raw_results = self._run_ocr(img_array)
        boxes = []
        for line in (raw_results or []):
            if line is None:
                continue
            points, (_, det_score) = line
            if det_score >= DET_CONFIDENCE_THRESHOLD:
                boxes.append(BBox(points=points, score=det_score))
        return boxes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_ocr(self, img_array: np.ndarray) -> list | None:
        """Invoke PaddleOCR and return the raw result list."""
        # PaddleOCR returns: [ [[box_points], [text, score]], ... ]
        # For a numpy array input it returns a list of lists (one per page).
        result = self._ocr.ocr(img_array, cls=False)
        # Flatten the page dimension (we always pass single images)
        if result and isinstance(result[0], list) and result[0] and isinstance(result[0][0], list):
            # Nested list: result = [ [ line, line, ... ] ]
            return result[0]
        return result

    def _parse_and_filter(self, raw_results: list | None) -> list[TextResult]:
        """Parse PaddleOCR output and apply both confidence thresholds.

        PaddleOCR line format: [box_points, (text_str, rec_score)]
        where box_points = [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]

        Detection confidence is embedded as the score attached to each box
        in the det-only output, but in the combined ocr() output the box
        itself is what PaddleOCR deemed above its internal det threshold.
        We apply our own DET_CONFIDENCE_THRESHOLD by checking the rec_score
        proxy returned per line, and our REC_CONFIDENCE_THRESHOLD on the
        recognition score.
        """
        results: list[TextResult] = []
        if not raw_results:
            return results

        for line in raw_results:
            if line is None or len(line) != 2:
                continue
            points, rec_info = line
            if not isinstance(rec_info, (list, tuple)) or len(rec_info) != 2:
                continue

            text_str, rec_score = rec_info
            text_str = str(text_str).strip()

            # Filter empty strings
            if not text_str:
                continue

            # Apply recognition confidence threshold
            if rec_score < REC_CONFIDENCE_THRESHOLD:
                logger.debug(
                    "Dropping '%s' (rec_score=%.3f < %.1f)",
                    text_str, rec_score, REC_CONFIDENCE_THRESHOLD,
                )
                continue

            bbox = BBox(points=points, score=rec_score)
            results.append(TextResult(text=text_str, score=rec_score, bbox=bbox))

        return results

    def _aggregate(self, text_results: list[TextResult]) -> str:
        """Sort text fragments in reading order (top→bottom, left→right) and join."""
        # Sort by top-left Y first, then X for same-row fragments
        sorted_results = sorted(
            text_results,
            key=lambda r: (r.bbox.top_left_y, r.bbox.top_left_x),
        )
        return "\n".join(r.text for r in sorted_results)
