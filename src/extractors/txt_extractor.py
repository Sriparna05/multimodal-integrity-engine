"""Text file (.txt) extractor with sentence-aware chunking."""

import logging
from pathlib import Path
from typing import Optional

from src.models import DocumentResult
from src.text_processing import clean_text, chunk_text

logger = logging.getLogger(__name__)


def extract_txt(file_path: Path | str, document_id: Optional[str] = None) -> DocumentResult:
    """Extracts text from a plain .txt file and segments it into clean chunks."""
    path = Path(file_path)
    doc_id = document_id or path.name

    if not path.exists():
        msg = f"TXT file does not exist: {path}"
        logger.error(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="txt",
            status="error",
            error_message=msg,
        )

    if path.stat().st_size == 0:
        msg = f"TXT file is empty (0 bytes): {path.name}"
        logger.warning(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="txt",
            status="error",
            error_message=msg,
        )

    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw_text = path.read_text(encoding="latin-1")
        except Exception as exc:
            msg = f"Failed to read TXT {path.name}: {exc}"
            logger.warning(msg)
            return DocumentResult(
                document_id=doc_id,
                file_path=str(path),
                modality="txt",
                status="error",
                error_message=msg,
            )

    cleaned = clean_text(raw_text)
    if not cleaned:
        msg = f"TXT file contains no text: {path.name}"
        logger.warning(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="txt",
            status="warning",
            error_message=msg,
            chunks=[],
            full_text="",
        )

    chunks = chunk_text(
        text=cleaned,
        document_id=doc_id,
        modality="txt",
        page_number=1,
    )

    return DocumentResult(
        document_id=doc_id,
        file_path=str(path),
        modality="txt",
        status="ok",
        chunks=chunks,
        full_text=cleaned,
    )
