"""
tkie/preprocessor.py
Image preprocessing utilities for the LLM-TKIE pipeline.

Responsibilities:
  - Load image from disk
  - Remove alpha channel (RGBA → RGB)
  - Normalize pixel values to [0, 1] float32
  - Resize image while preserving aspect ratio
  - Rotate image by arbitrary degree increments
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from tkie.config import IMAGE_MAX_SIDE

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles all image pre-processing before OCR ingestion."""

    def __init__(self, max_side: int = IMAGE_MAX_SIDE) -> None:
        self.max_side = max_side

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, image_path: str | Path) -> np.ndarray:
        """Full pipeline: load → strip alpha → resize → return uint8 array.

        PaddleOCR expects a uint8 BGR/RGB numpy array.  We keep the output
        as uint8 RGB; PaddleOCR's ``ocr()`` accepts numpy arrays directly.
        """
        img = self.load(image_path)
        img = self.remove_alpha(img)
        img = self.resize(img)
        array = np.array(img, dtype=np.uint8)
        logger.debug("Preprocessed image to shape %s", array.shape)
        return array

    def rotate(self, img_array: np.ndarray, degrees: int) -> np.ndarray:
        """Rotate a uint8 numpy array by *degrees* (must be 0, 90, 180, or 270).

        Uses PIL for lossless, exact 90° increments.
        """
        if degrees == 0:
            return img_array
        pil_img = Image.fromarray(img_array)
        # PIL's rotate is counter-clockwise; expand=True keeps full content.
        rotated = pil_img.rotate(degrees, expand=True)
        return np.array(rotated, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Step helpers (can be used individually for testing)
    # ------------------------------------------------------------------

    def load(self, image_path: str | Path) -> Image.Image:
        """Load an image from disk using PIL."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path)
        logger.debug("Loaded image %s (mode=%s, size=%s)", path.name, img.mode, img.size)
        return img

    def remove_alpha(self, img: Image.Image) -> Image.Image:
        """Convert any image to pure RGB, compositing transparency over white."""
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 3 = alpha channel
            logger.debug("Removed alpha channel (RGBA → RGB)")
            return background
        return img.convert("RGB")

    def normalize(self, img_array: np.ndarray) -> np.ndarray:
        """Scale pixel values from [0, 255] to [0.0, 1.0] float32.

        Note: PaddleOCR expects uint8 input, so call this only for downstream
        non-OCR consumers (e.g., feature extraction, visualisation).
        """
        return img_array.astype(np.float32) / 255.0

    def resize(
        self,
        img: Image.Image,
        max_side: int | None = None,
    ) -> Image.Image:
        """Resize so the longest side == *max_side*, preserving aspect ratio.

        If the image is already smaller, it is returned unchanged.
        """
        max_side = max_side or self.max_side
        w, h = img.size
        longest = max(w, h)
        if longest <= max_side:
            return img
        scale = max_side / longest
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        logger.debug("Resized %s → %s", (w, h), (new_w, new_h))
        return resized
