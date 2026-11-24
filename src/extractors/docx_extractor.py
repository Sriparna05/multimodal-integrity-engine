"""DOCX extractor using python-docx with paragraph-aware chunking."""

import logging
from pathlib import Path
from typing import Optional

from docx import Document

from src.models import DocumentResult
from src.text_processing import clean_text, chunk_document

logger = logging.getLogger(__name__)

WORDS_PER_PAGE_ESTIMATE = 350


def extract_docx(file_path: Path | str, document_id: Optional[str] = None) -> DocumentResult:
    """Extracts text from a DOCX document with estimated page numbers (~350 words/page).

    Handles corrupted or empty files gracefully without crashing.
    """
    path = Path(file_path)
    doc_id = document_id or path.name

    if not path.exists():
        msg = f"DOCX file does not exist: {path}"
        logger.error(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="docx",
            status="error",
            error_message=msg,
        )

    if path.stat().st_size == 0:
        msg = f"DOCX file is empty (0 bytes): {path.name}"
        logger.warning(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="docx",
            status="error",
            error_message=msg,
        )

    try:
        doc = Document(str(path))
    except Exception as exc:
        msg = f"Failed to parse DOCX {path.name}: {exc}"
        logger.warning(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="docx",
            status="error",
            error_message=msg,
        )

    # Extract all non-empty paragraphs and table contents
    extracted_paragraphs: list[str] = []
    for p in doc.paragraphs:
        cleaned = clean_text(p.text)
        if cleaned:
            extracted_paragraphs.append(cleaned)

    for table in doc.tables:
        for row in table.rows:
            row_cells = [clean_text(cell.text) for cell in row.cells if clean_text(cell.text)]
            if row_cells:
                extracted_paragraphs.append(" | ".join(row_cells))

    if not extracted_paragraphs:
        msg = f"DOCX contains no extractable text: {path.name}"
        logger.warning(msg)
        return DocumentResult(
            document_id=doc_id,
            file_path=str(path),
            modality="docx",
            status="warning",
            error_message=msg,
            chunks=[],
            full_text="",
        )

    full_text = "\n\n".join(extracted_paragraphs).strip()

    # Group paragraphs into estimated pages based on WORDS_PER_PAGE_ESTIMATE
    page_buckets: list[tuple[int, str]] = []
    current_page_paras: list[str] = []
    current_page_words = 0
    current_page_num = 1

    for p in extracted_paragraphs:
        p_words = len(p.split())
        current_page_paras.append(p)
        current_page_words += p_words

        if current_page_words >= WORDS_PER_PAGE_ESTIMATE:
            page_buckets.append((current_page_num, "\n\n".join(current_page_paras)))
            current_page_num += 1
            current_page_paras = []
            current_page_words = 0

    if current_page_paras:
        page_buckets.append((current_page_num, "\n\n".join(current_page_paras)))

    all_chunks = []
    chunk_offset = 0
    for page_num, page_content in page_buckets:
        chunks = chunk_document(
            text=page_content,
            document_id=doc_id,
            modality="docx",
            page_number=page_num,
            chunk_index_offset=chunk_offset,
        )
        all_chunks.extend(chunks)
        chunk_offset += len(chunks)

    return DocumentResult(
        document_id=doc_id,
        file_path=str(path),
        modality="docx",
        status="ok",
        chunks=all_chunks,
        full_text=full_text,
    )
