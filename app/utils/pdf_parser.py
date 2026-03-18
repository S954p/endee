"""
Document processing module.
Handles text extraction from PDF/TXT files and text chunking.
"""

import logging
import io
from PyPDF2 import PdfReader
from app.config import settings

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text content from a PDF file.

    Args:
        file_content: Raw PDF file bytes

    Returns:
        Extracted text as a single string
    """
    reader = PdfReader(io.BytesIO(file_content))
    text_parts = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text.strip())
            logger.debug(f"Extracted text from page {page_num + 1}")

    full_text = "\n\n".join(text_parts)
    logger.info(f"Extracted {len(full_text)} characters from PDF ({len(reader.pages)} pages)")
    return full_text


def extract_text_from_txt(file_content: bytes) -> str:
    """
    Extract text from a plain text file.

    Args:
        file_content: Raw text file bytes

    Returns:
        Decoded text string
    """
    # Try UTF-8 first, fall back to latin-1
    try:
        text = file_content.decode("utf-8")
    except UnicodeDecodeError:
        text = file_content.decode("latin-1")

    logger.info(f"Extracted {len(text)} characters from text file")
    return text


def extract_text(file_content: bytes, filename: str) -> str:
    """
    Extract text from a file based on its extension.

    Args:
        file_content: Raw file bytes
        filename: Original filename (used to determine format)

    Returns:
        Extracted text content

    Raises:
        ValueError: If file format is not supported
    """
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf(file_content)
    elif lower_name.endswith(".txt"):
        return extract_text_from_txt(file_content)
    else:
        raise ValueError(f"Unsupported file format: {filename}. Only PDF and TXT files are supported.")


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into overlapping chunks using a sliding window.

    Args:
        text: Full text to chunk
        chunk_size: Maximum characters per chunk (default from settings)
        chunk_overlap: Overlap between consecutive chunks (default from settings)

    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.CHUNK_OVERLAP

    # Clean up whitespace
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end, try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence boundary (period, question mark, exclamation)
            for boundary_char in [".\n", ".\r", ". ", "?\n", "? ", "!\n", "! "]:
                boundary = text.rfind(boundary_char, start, end)
                if boundary != -1 and boundary > start + chunk_size // 2:
                    end = boundary + 1
                    break
            else:
                # Fall back to word boundary
                space = text.rfind(" ", start, end)
                if space != -1 and space > start + chunk_size // 2:
                    end = space + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward, accounting for overlap
        start = end - chunk_overlap if end < len(text) else end

    logger.info(f"Split text into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
