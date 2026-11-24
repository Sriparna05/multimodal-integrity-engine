"""Core data models for multimodal submission integrity and plagiarism detection."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextChunk:
    """A granular chunk of extracted text preserving modality and positional metadata."""

    chunk_id: str
    document_id: str
    modality: str  # "pdf" | "docx" | "mp4" | "txt"
    text: str
    page_number: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    token_count: int = 0


@dataclass
class DocumentResult:
    """Unified result of document extraction containing status, metadata, and chunks."""

    document_id: str
    file_path: str
    modality: str
    status: str  # "ok" | "warning" | "error"
    error_message: Optional[str] = None
    chunks: list[TextChunk] = field(default_factory=list)
    full_text: str = ""
