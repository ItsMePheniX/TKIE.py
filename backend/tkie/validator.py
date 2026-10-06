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
import math
import re
from typing import Any

from tkie.config import JSON_SCHEMA

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

    def __init__(self, template: dict[str, Any], schema: dict[str, Any] | None = None) -> None:
        self.template = template
        self._required_keys: set[str] = set(template.keys())
        self.schema = schema or JSON_SCHEMA

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
        """Require an exact, recursively type-checked output schema."""
        actual_keys = set(parsed)
        missing = self._required_keys - actual_keys
        extra = actual_keys - self._required_keys
        if missing or extra:
            logger.warning("JSON schema key mismatch; missing=%s, extra=%s", missing, extra)
            return False

        error = self._validate_value(parsed, self.schema, path="$")
        if error:
            logger.warning("JSON schema validation failed: %s", error)
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

    def _validate_value(self, value: Any, schema: dict[str, Any], path: str) -> str | None:
        expected = schema.get("type")
        expected_types = expected if isinstance(expected, list) else [expected]
        if not any(self._matches_type(value, kind) for kind in expected_types):
            return f"{path} must be {expected_types}, got {type(value).__name__}"

        if isinstance(value, dict):
            properties = schema.get("properties", {})
            required = set(schema.get("required", properties))
            keys = set(value)
            missing = required - keys
            extra = keys - set(properties)
            if missing:
                return f"{path} missing keys: {sorted(missing)}"
            if extra and not schema.get("additionalProperties", True):
                return f"{path} has unexpected keys: {sorted(extra)}"
            for key, child_schema in properties.items():
                if key in value:
                    error = self._validate_value(value[key], child_schema, f"{path}.{key}")
                    if error:
                        return error

        if isinstance(value, list):
            item_schema = schema.get("items")
            if item_schema:
                for index, item in enumerate(value):
                    error = self._validate_value(item, item_schema, f"{path}[{index}]")
                    if error:
                        return error
        return None

    @staticmethod
    def _matches_type(value: Any, expected: str | None) -> bool:
        # bool is an int subclass in Python but must not qualify as a number.
        return {
            "object": lambda: isinstance(value, dict),
            "array": lambda: isinstance(value, list),
            "string": lambda: isinstance(value, str),
            "number": lambda: (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ),
            "boolean": lambda: isinstance(value, bool),
            "null": lambda: value is None,
        }.get(expected, lambda: False)()
