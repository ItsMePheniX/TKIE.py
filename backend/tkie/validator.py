"""
tkie/validator.py
Regex-based JSON validator and schema conformance checker.

Algorithm (Algorithm 1 from the paper):
  1. Extract the first {...} JSON block from raw LLM output via regex
  2. Parse it with json.loads
  3. Verify all top-level keys from the template are present
  Returns (is_valid: bool, parsed_dict | None)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Matches the outermost {...} block, including nested braces, non-greedy.
# We use a manual balanced-brace scan for robustness instead of naive regex.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_outermost_json(text: str) -> str | None:
    """Extract the first well-balanced {...} JSON object from *text*.

    Handles:
      - Bare JSON (most common when system prompt is obeyed)
      - JSON wrapped in ``` or ```json fences (common LLM habit)
      - Leading/trailing prose around the JSON block
    """
    # 1. Try to strip markdown fences first
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate.startswith("{"):
            return candidate

    # 2. Find the first '{' and scan for a balanced closing '}'
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


class JSONValidator:
    """Validates and parses LLM output against the target JSON template."""

    def __init__(self, template: dict[str, Any]) -> None:
        self.template = template
        self._required_keys: set[str] = set(template.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, raw_text: str) -> tuple[bool, dict | None]:
        """Full validation pipeline.

        1. Extract a JSON block from the raw LLM output.
        2. Parse it.
        3. Verify all required top-level keys are present.

        Returns:
            (True, parsed_dict)  — if valid
            (False, None)        — if extraction, parsing, or key check fails
        """
        json_str = self.extract_json_block(raw_text)
        if json_str is None:
            logger.warning("No JSON block found in LLM output.")
            return False, None

        parsed = self._parse(json_str)
        if parsed is None:
            return False, None

        if not self.validate_against_template(parsed):
            return False, None

        logger.info("JSON validation passed.")
        return True, parsed

    def extract_json_block(self, text: str) -> str | None:
        """Extract the first balanced JSON object string from *text*."""
        block = _extract_outermost_json(text)
        if block is None:
            logger.debug("extract_json_block: no JSON block found.")
        return block

    def validate_against_template(self, parsed: dict) -> bool:
        """Check that all top-level template keys are present in *parsed*.

        Missing keys cause a validation failure; extra keys are tolerated.
        """
        missing = self._required_keys - set(parsed.keys())
        if missing:
            logger.warning("JSON missing required keys: %s", missing)
            return False
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse(self, json_str: str) -> dict | None:
        """Attempt to parse a JSON string; return None on failure."""
        try:
            result = json.loads(json_str)
            if not isinstance(result, dict):
                logger.warning("Parsed JSON is not a dict (got %s).", type(result).__name__)
                return None
            return result
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error: %s", exc)
            return None
