"""PDF extractor using PyMuPDF (pymupdf) with page-level and block-level preservation."""

import logging
from pathlib import Path
from typing import Optional

import pymupdf

from src.models import DocumentResult
from src.text_processing import clean_text, chunk_document

logger = logging.getLogger(__name__)


def extract_pdf(file_path: Path | str, document_id: Optional[str] = None) -> DocumentResult:
    """Extracts text page-by-page from a PDF document, preserving 1-indexed page numbers.

    Catches empty, 0-byte, or corrupted PDFs gracefully and returns an error result
    without raising an unhandled exception.
    """
    path = Path(file_path)
    doc_id = document_id or path.name

    if not path.exists():
        msg = f"PDF file does not exist: {path}"
        logger.error(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="pdf",
            status="error",
            error_message=msg,
        )

    # Detect 0-byte or empty file immediately
    if path.stat().st_size == 0:
        msg = f"PDF file is empty (0 bytes): {path.name}"
        logger.warning(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="pdf",
            status="error",
            error_message=msg,
        )

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        msg = f"Failed to parse PDF {path.name}: {exc}"
        logger.warning(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="pdf",
            status="error",
            error_message=msg,
        )

    try:
        if len(doc) == 0:
            msg = f"PDF contains 0 pages: {path.name}"
            logger.warning(msg)
            return DocumentResult(
                document_id=doc_id,
                file_path=str(path),
                modality="pdf",
                status="error",
                error_message=msg,
            )

        all_chunks = []
        page_texts = []
        chunk_offset = 0

        for page_idx, page in enumerate(doc, start=1):
            # Extract blocks to preserve natural paragraph boundaries
            raw_blocks = page.get_text("blocks")
            cleaned_blocks = []
            for b in raw_blocks:
                cleaned_b = clean_text(b[4])
                if cleaned_b:
                    cleaned_blocks.append(cleaned_b)

            page_text = "\n\n".join(cleaned_blocks).strip()
            if page_text:
                page_texts.append(f"[Page {page_idx}]\n{page_text}")
                chunks = chunk_document(
                    text=page_text,
                    document_id=doc_id,
                    modality="pdf",
                    page_number=page_idx,
                    chunk_index_offset=chunk_offset,
                )
                all_chunks.extend(chunks)
                chunk_offset += len(chunks)

        full_text = "\n\n".join(page_texts).strip()

        if not full_text:
            msg = f"PDF contains no extractable text (scanned or blank): {path.name}"
            logger.warning(msg)
            return DocumentResult(
                document_id=doc_id,
                file_path=str(path),
                modality="pdf",
                status="warning",
                error_message=msg,
                chunks=[],
                full_text="",
            )

        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="pdf",
            status="ok",
            chunks=all_chunks,
            full_text=full_text,
        )
    finally:
        doc.close()
