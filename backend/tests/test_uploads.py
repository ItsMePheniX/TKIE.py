from io import BytesIO
import unittest

from fastapi import HTTPException, UploadFile
from PIL import Image

from main import _read_and_verify_upload


def upload(payload: bytes, content_type: str) -> UploadFile:
    return UploadFile(filename="untrusted-name.exe", file=BytesIO(payload), headers={"content-type": content_type})


class UploadValidationTests(unittest.TestCase):
    def test_accepts_verified_png_and_uses_content_suffix(self) -> None:
        buffer = BytesIO()
        Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
        payload, suffix = _read_and_verify_upload(upload(buffer.getvalue(), "image/png"))
        self.assertTrue(payload.startswith(b"\x89PNG"))
        self.assertEqual(suffix, ".png")

    def test_rejects_content_type_that_is_not_an_image(self) -> None:
        with self.assertRaises(HTTPException) as context:
            _read_and_verify_upload(upload(b"not an image", "text/plain"))
        self.assertEqual(context.exception.status_code, 415)

    def test_rejects_malformed_image_bytes(self) -> None:
        with self.assertRaises(HTTPException) as context:
            _read_and_verify_upload(upload(b"not a png", "image/png"))
        self.assertEqual(context.exception.status_code, 422)
