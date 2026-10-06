"""Optional MongoDB persistence for extraction audit records.

The module intentionally imports PyMongo only when a connection is requested,
so local development remains usable when ``MONGODB_URI`` is not configured.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from tkie.config import MONGODB_TIMEOUT_MS

logger = logging.getLogger(__name__)


class MongoStore:
    """Store extraction metadata and structured output in MongoDB."""

    def __init__(self, uri: str, database: str, timeout_ms: int = MONGODB_TIMEOUT_MS) -> None:
        self.uri = uri
        self.database_name = database
        self.timeout_ms = timeout_ms
        self._client: Any = None
        self._database: Any = None

    def connect(self) -> None:
        """Connect eagerly so a bad URI is reported at startup, not mid-request."""
        try:
            from pymongo import MongoClient
            from pymongo.errors import PyMongoError
        except ImportError as exc:
            raise RuntimeError("pymongo is required when MONGODB_URI is configured.") from exc

        try:
            self._client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=self.timeout_ms,
                connectTimeoutMS=self.timeout_ms,
            )
            self._client.admin.command("ping")
            self._database = self._client[self.database_name]
            self._database.extractions.create_index([("created_at", -1)])
            self._database.extractions.create_index([("status", 1), ("created_at", -1)])
        except PyMongoError as exc:
            self.close()
            raise RuntimeError("Could not connect to MongoDB.") from exc

    @property
    def connected(self) -> bool:
        return self._database is not None

    def save_extraction(
        self,
        *,
        filename: str | None,
        status: str,
        model: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> str:
        """Persist one extraction attempt; never stores the uploaded image itself."""
        if self._database is None:
            raise RuntimeError("MongoDB is not connected.")
        document: dict[str, Any] = {
            "created_at": datetime.now(UTC),
            "filename": filename,
            "status": status,
            "model": model,
        }
        if result is not None:
            document["result"] = result
        if error is not None:
            document["error"] = error
        return str(self._database.extractions.insert_one(document).inserted_id)

    def list_extractions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return newest audit records with Mongo's internal ID serialised safely."""
        if self._database is None:
            raise RuntimeError("MongoDB is not connected.")
        records: list[dict[str, Any]] = []
        for document in self._database.extractions.find().sort("created_at", -1).limit(limit):
            document["id"] = str(document.pop("_id"))
            records.append(document)
        return records

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._database = None
