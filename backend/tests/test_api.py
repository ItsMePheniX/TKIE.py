from io import BytesIO
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import main
from tkie.exceptions import LLMError


def png_upload() -> tuple[str, BytesIO, str]:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    buffer.seek(0)
    return "invoice.png", buffer, "image/png"


class FakeExtractor:
    def __init__(self, result: dict | None = None, error: Exception | None = None, ready: bool = True) -> None:
        self.result = result
        self.error = error
        self.ready = ready
        self.seen_path: Path | None = None
        self.path_existed_during_process = False

    def process(self, image_path: Path) -> dict | None:
        self.seen_path = image_path
        self.path_existed_during_process = image_path.exists()
        if self.error:
            raise self.error
        return self.result

    def is_ready(self) -> bool:
        return self.ready


class FakeMongoStore:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self.closed = False

    def save_extraction(self, **record) -> str:
        self.records.append(record)
        return "record-id"

    def list_extractions(self, limit: int) -> list[dict]:
        return self.records[:limit]

    def close(self) -> None:
        self.closed = True


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_extractor = main._extractor
        self.original_mongo_store = main._mongo_store
        self.original_mongodb_configured = main._mongodb_configured

    def tearDown(self) -> None:
        main._extractor = self.original_extractor
        main._mongo_store = self.original_mongo_store
        main._mongodb_configured = self.original_mongodb_configured

    def request(self, extractor: FakeExtractor, path: str = "/extract", mongo_store=None):
        with patch.dict(os.environ, {"MONGODB_URI": ""}):
            with patch("main.TKIEExtractor", return_value=extractor):
                with TestClient(main.app) as client:
                    if mongo_store is not None:
                        main._mongo_store = mongo_store
                    if path == "/extract":
                        return client.post(path, files={"file": png_upload()})
                    return client.get(path)

    def test_extract_returns_data_and_removes_temporary_upload(self) -> None:
        extractor = FakeExtractor(result={"company_name": "Acme"})
        response = self.request(extractor)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"company_name": "Acme"})
        self.assertTrue(extractor.path_existed_during_process)
        self.assertIsNotNone(extractor.seen_path)
        self.assertFalse(extractor.seen_path.exists())

    def test_extract_persists_successful_audit_record_when_mongo_is_connected(self) -> None:
        mongo_store = FakeMongoStore()
        response = self.request(
            FakeExtractor(result={"company_name": "Acme"}),
            mongo_store=mongo_store,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(mongo_store.records), 1)
        self.assertEqual(mongo_store.records[0]["status"], "succeeded")
        self.assertEqual(mongo_store.records[0]["result"], {"company_name": "Acme"})

    def test_extractions_endpoint_returns_persisted_records(self) -> None:
        mongo_store = FakeMongoStore()
        mongo_store.records.append({"status": "succeeded", "filename": "invoice.png"})
        response = self.request(FakeExtractor(), "/extractions", mongo_store)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 1)

    def test_extract_returns_422_for_quarantined_document(self) -> None:
        response = self.request(FakeExtractor(result=None))
        self.assertEqual(response.status_code, 422)
        self.assertIn("quarantined", response.json()["detail"].lower())

    def test_extract_returns_502_for_unavailable_llm(self) -> None:
        response = self.request(FakeExtractor(error=LLMError("connection refused")))
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "A local extraction service is unavailable.")

    def test_ready_distinguishes_available_and_unavailable_model(self) -> None:
        self.assertEqual(self.request(FakeExtractor(ready=True), "/ready").status_code, 200)
        self.assertEqual(self.request(FakeExtractor(ready=False), "/ready").status_code, 503)
