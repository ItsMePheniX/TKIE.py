import unittest

from tkie.database import MongoStore


class FakeInsertResult:
    inserted_id = "mongo-id"


class FakeCursor:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

    def sort(self, *_args):
        return self

    def limit(self, limit: int):
        return self.documents[:limit]


class FakeCollection:
    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.indexes: list[list[tuple[str, int]]] = []

    def create_index(self, index):
        self.indexes.append(index)

    def insert_one(self, document: dict) -> FakeInsertResult:
        stored = dict(document)
        stored["_id"] = "mongo-id"
        self.documents.append(stored)
        return FakeInsertResult()

    def find(self) -> FakeCursor:
        return FakeCursor(self.documents)


class FakeDatabase:
    def __init__(self) -> None:
        self.extractions = FakeCollection()


class MongoStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MongoStore("mongodb://unused", "tkie")
        self.store._database = FakeDatabase()

    def test_saves_audit_record_without_image_content(self) -> None:
        record_id = self.store.save_extraction(
            filename="invoice.png",
            status="succeeded",
            model="llama3:8b",
            result={"total": 12.5},
        )
        stored = self.store._database.extractions.documents[0]
        self.assertEqual(record_id, "mongo-id")
        self.assertEqual(stored["filename"], "invoice.png")
        self.assertEqual(stored["result"], {"total": 12.5})
        self.assertNotIn("image", stored)

    def test_lists_records_with_serializable_id(self) -> None:
        self.store.save_extraction(filename=None, status="quarantined", model="llama3:8b")
        records = self.store.list_extractions()
        self.assertEqual(records[0]["id"], "mongo-id")
        self.assertNotIn("_id", records[0])
