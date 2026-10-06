import unittest

from tkie.ocr import OCRPipeline


class OCRParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        # Avoid initialising PaddleOCR; the parser is deliberately unit-testable.
        self.pipeline = object.__new__(OCRPipeline)

    def test_filters_only_recognition_score_and_orders_reading_direction(self) -> None:
        raw = [
            [[[50, 10], [60, 10], [60, 20], [50, 20]], ("right", 0.95)],
            [[[10, 10], [20, 10], [20, 20], [10, 20]], ("left", 0.99)],
            [[[10, 30], [20, 30], [20, 40], [10, 40]], ("uncertain", 0.5)],
        ]
        results = self.pipeline._parse_and_filter(raw)
        self.assertEqual([result.text for result in results], ["right", "left"])
        self.assertEqual(self.pipeline._aggregate(results), "left\nright")

    def test_handles_empty_ocr_response(self) -> None:
        self.assertEqual(self.pipeline._parse_and_filter(None), [])
