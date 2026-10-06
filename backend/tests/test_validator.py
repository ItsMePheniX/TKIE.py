import copy
import json
import unittest

from tkie.config import JSON_TEMPLATE
from tkie.validator import JSONValidator


def valid_document() -> dict:
    document = copy.deepcopy(JSON_TEMPLATE)
    document.update(
        {
            "company_name": "Acme Ltd",
            "document_date": "2026-10-05",
            "document_number": "INV-1",
            "vendor_name": "Customer Inc",
            "line_items": [
                {"description": "Paper", "quantity": 2, "unit_price": 3.5, "amount": 7.0}
            ],
            "subtotal": 7.0,
            "tax": 0.0,
            "total": 7.0,
            "currency": "USD",
            "payment_terms": "Net 30",
        }
    )
    return document


class JSONValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = JSONValidator(JSON_TEMPLATE)

    def test_accepts_exact_schema_in_markdown_fence(self) -> None:
        raw = "```json\n" + json.dumps(valid_document()) + "\n```"
        valid, parsed = self.validator.validate(raw)
        self.assertTrue(valid)
        self.assertEqual(parsed, valid_document())

    def test_rejects_extra_top_level_key(self) -> None:
        document = valid_document()
        document["confidence"] = 0.99
        self.assertEqual(self.validator.validate(json.dumps(document)), (False, None))

    def test_rejects_invalid_line_item_shape_and_type(self) -> None:
        document = valid_document()
        document["line_items"][0]["quantity"] = "two"
        document["line_items"][0]["sku"] = "P-1"
        self.assertEqual(self.validator.validate(json.dumps(document)), (False, None))

    def test_rejects_non_finite_numbers(self) -> None:
        document = valid_document()
        document["total"] = float("nan")
        self.assertEqual(self.validator.validate(json.dumps(document)), (False, None))
