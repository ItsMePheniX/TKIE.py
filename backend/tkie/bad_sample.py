"""
tkie/bad_sample.py
Bad Sample Repository — quarantines documents that failed the pipeline.

When a document fails all rotation attempts during the completeness check,
it is copied into the bad_samples/ directory alongside a human-readable
sidecar file recording the failure reason and timestamp.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from tkie.config import BAD_SAMPLE_DIR

logger = logging.getLogger(__name__)


class BadSampleRepository:
    """Manages the bad sample quarantine directory."""

    def __init__(self, output_dir: Path | str = BAD_SAMPLE_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("BadSampleRepository initialised at %s", self.output_dir)

    def save(self, image_path: str | Path, reason: str) -> Path:
        """Copy *image_path* into the bad samples directory with a sidecar log.

        Args:
            image_path: Original image file path.
            reason:     Human-readable failure reason.

        Returns:
            Path to the copied image in the bad samples directory.
        """
        src = Path(image_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{timestamp}_{src.stem}"

        # Copy the image
        dest_image = self.output_dir / f"{stem}{src.suffix}"
        shutil.copy2(src, dest_image)

        # Write sidecar .txt
        sidecar = self.output_dir / f"{stem}.txt"
        sidecar.write_text(
            f"Original file : {src.resolve()}\n"
            f"Timestamp     : {datetime.now().isoformat()}\n"
            f"Failure reason: {reason}\n",
            encoding="utf-8",
        )

        logger.warning(
            "Bad sample saved: %s (reason: %s)", dest_image.name, reason
        )
        return dest_image
