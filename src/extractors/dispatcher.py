"""Extractor dispatcher: routes documents to modality-specific extractors."""

import logging
from pathlib import Path
from typing import Optional

from src.extractors.docx_extractor import extract_docx
from src.extractors.mp4_extractor import extract_mp4
from src.extractors.pdf_extractor import extract_pdf
from src.extractors.txt_extractor import extract_txt
from src.models import DocumentResult

logger = logging.getLogger(__name__)


def extract_document(file_path: Path | str, document_id: Optional[str] = None) -> DocumentResult:
    """Routes a document to the appropriate extractor based on its file extension.

    Supports .pdf, .docx, .mp4, and .txt files. Unrecognized extensions log a warning
    and return an error result without crashing.
    """
    path = Path(file_path)
    doc_id = document_id or path.name
    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_pdf(path, document_id=doc_id)
    elif ext == ".docx":
        return extract_docx(path, document_id=doc_id)
    elif ext == ".mp4":
        return extract_mp4(path, document_id=doc_id)
    elif ext == ".txt":
        return extract_txt(path, document_id=doc_id)
    else:
        msg = f"Unsupported file format '{ext}' for file: {path.name}"
        logger.warning(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality=ext.lstrip(".") or "unknown",
            status="warning",
            error_message=msg,
            chunks=[],
            full_text="",
        )
