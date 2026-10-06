"""
tkie/exceptions.py
Custom exceptions for the LLM-TKIE pipeline.
"""


class TKIEError(Exception):
    """Base exception for all TKIE errors."""


class JSONFormattingError(TKIEError):
    """Raised when the LLM fails to produce valid JSON after the refinement loop."""


class CompletenessCheckError(TKIEError):
    """Raised when a document fails all rotation attempts during completeness check."""


class OCRError(TKIEError):
    """Raised when the OCR pipeline encounters an unrecoverable error."""


class LLMError(TKIEError):
    """Raised when the local LLM service cannot complete a request."""
