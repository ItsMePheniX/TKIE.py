import unittest

from tkie.llm import OllamaLLMClient


class FakeClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def list(self):
        if self.error:
            raise self.error
        return self.response


class OllamaReadinessTests(unittest.TestCase):
    def test_accepts_configured_model_in_list_response(self) -> None:
        client = object.__new__(OllamaLLMClient)
        client.model = "llama3:8b"
        client._client = FakeClient({"models": [{"model": "llama3:8b"}]})
        self.assertTrue(client.is_available())

    def test_rejects_missing_model_or_failed_request(self) -> None:
        client = object.__new__(OllamaLLMClient)
        client.model = "llama3:8b"
        client._client = FakeClient({"models": [{"model": "qwen2:7b"}]})
        self.assertFalse(client.is_available())
        client._client = FakeClient(error=OSError("unreachable"))
        self.assertFalse(client.is_available())
