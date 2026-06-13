"""
text_extraction.py — PDF → raw text via pypdf locally.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def extract_text_from_file(
    pdf_path: str,
    region: Optional[str] = None,
) -> str:
    """
    Extract all text from a local PDF file using pypdf locally.
    
    Args:
        pdf_path: Absolute or relative path to the PDF file.
        region:   Ignored (kept for compatibility).

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: If pdf_path doesn't exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Extracting text locally from %s using pypdf", path.name)
    
    # Check if file is pdf, otherwise try to decode as text
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            logger.info("Local pypdf extraction complete: %d characters extracted.", len(text))
            return text
        except ImportError:
            logger.error("pypdf is not installed. Please pip install pypdf")
            raise
    else:
        # Fallback for plain text files
        try:
            text = path.read_text(encoding="utf-8")
            logger.info("Local text read complete: %d characters.", len(text))
            return text
        except UnicodeDecodeError:
            raise ValueError(f"File {path.name} is not a PDF and cannot be read as UTF-8 text.")

def extract_text_from_s3(
    bucket: str,
    key: str,
    region: Optional[str] = None,
    poll_interval_seconds: int = 5,
    max_wait_seconds: int = 300,
) -> str:
    """Mock implementation for extracting from S3."""
    logger.warning("extract_text_from_s3 called but AWS is disabled. Returning simulated text.")
    return f"Simulated text from s3://{bucket}/{key}"
