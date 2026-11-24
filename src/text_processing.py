"""Text cleaning, normalization, and paragraph-aware chunking utilities."""

import re
from typing import Optional
import unicodedata

from src.models import TextChunk


_TENS_MAP = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_UNITS_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

_SMALL_NUMBERS_MAP = {
    "zero": "0",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    **{k: str(v) for k, v in _TENS_MAP.items()},
}

_COMPOUND_NUM_RE = re.compile(
    r"\b(" + "|".join(_TENS_MAP.keys()) + r")[-\s]+(" + "|".join(_UNITS_MAP.keys()) + r")\b",
    flags=re.IGNORECASE,
)
_SMALL_NUM_RE = re.compile(
    r"\b(" + "|".join(_SMALL_NUMBERS_MAP.keys()) + r")\b",
    flags=re.IGNORECASE,
)


def normalize_spoken_written(text: str) -> str:
    """Normalizes common spoken-vs-written mismatches (numbers, percentages, hyphenation)."""
    # 1. Compound spelled-out numbers (e.g., "forty-two" -> "42", "sixty-four" -> "64")
    text = _COMPOUND_NUM_RE.sub(
        lambda m: str(_TENS_MAP[m.group(1).lower()] + _UNITS_MAP[m.group(2).lower()]),
        text,
    )
    # 2. Standalone spelled-out numbers (e.g., "eighteen" -> "18")
    text = _SMALL_NUM_RE.sub(
        lambda m: _SMALL_NUMBERS_MAP[m.group(1).lower()],
        text,
    )
    # 3. Spelled-out percentages (e.g., "64 percent" -> "64%")
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s+percent\b", r"\1%", text, flags=re.IGNORECASE)
    # 4. Intra-word hyphenation between letters (e.g., "pro-forma" -> "pro forma")
    text = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", text)
    return text


def clean_text(text: str) -> str:
    """Normalizes whitespace/unicode and strips standalone page markers/headers.

    Replaces single intra-paragraph line breaks with spaces while preserving
    distinct paragraph breaks (\n\n) to eliminate jagged mid-sentence wrapping.
    """
    if not text:
        return ""

    # Normalize unicode representations (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # Standardize whitespace and line endings
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")

    # Strip decorative horizontal divider lines (e.g. ――――, ----, ====)
    text = re.sub(r"^[―\-_=]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove standalone page numbers or headers like "Page 1 of 5", "Page 2", "- 3 -"
    text = re.sub(r"(?im)^\s*(?:page\s+\d+(?:\s*(?:of|\/)\s*\d+)?|\b\d+\s+(?:of|\/)\s+\d+\b|-?\s*\d+\s*-?)\s*$", "", text)
    text = re.sub(r"(?i)\bpage\s+\d+\s*(?:of|\/)\s*\d+\b", "", text)

    # Split into distinct paragraphs, collapsing single intra-paragraph newlines into spaces
    raw_paragraphs = re.split(r"\n\s*\n+", text)
    cleaned_paragraphs = []
    for p in raw_paragraphs:
        # Replace single newline with space within a paragraph
        p_flat = re.sub(r"\s*\n\s*", " ", p)
        # Normalize spoken-vs-written mismatches and collapse horizontal whitespace
        p_flat = normalize_spoken_written(p_flat)
        p_flat = re.sub(r"[ \t]+", " ", p_flat).strip()
        if p_flat:
            cleaned_paragraphs.append(p_flat)

    return "\n\n".join(cleaned_paragraphs)


def estimate_tokens(text: str) -> int:
    """Estimates token count based on whitespace word segmentation."""
    if not text:
        return 0
    return len(text.split())


def split_into_sentences(text: str) -> list[str]:
    """Splits text into sentences cleanly based on punctuation boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    text: str,
    document_id: str,
    modality: str,
    page_number: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    chunk_index_offset: int = 0,
    target_words: int = 100,
    overlap_words: int = 20,
) -> list[TextChunk]:
    """Segments a text block into paragraph-aware chunks (~80-120 words).

    Ensures that shared template/boilerplate headers stay isolated in their own chunk
    rather than merging with student analytical content. Overlaps are applied ONLY
    within long paragraphs, never across distinct paragraphs.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # Split into distinct paragraphs by double newline
    raw_paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        return []

    # Coalesce trivial heading-only paragraphs (< 18 words) with the subsequent paragraph
    coalesced: list[str] = []
    idx = 0
    while idx < len(raw_paragraphs):
        p = raw_paragraphs[idx]
        p_words = len(p.split())

        if p_words < 18 and (idx + 1) < len(raw_paragraphs):
            next_p = raw_paragraphs[idx + 1]
            coalesced.append(f"{p} {next_p}")
            idx += 2
            continue
        coalesced.append(p)
        idx += 1

    chunks: list[TextChunk] = []
    chunk_counter = chunk_index_offset + 1

    for para in coalesced:
        words = para.split()
        p_word_count = len(words)

        # Standard paragraph within ~145 words forms an atomic chunk
        if p_word_count <= 145:
            chunk_id = f"{document_id}_chk_{chunk_counter:03d}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    modality=modality,
                    text=para,
                    page_number=page_number,
                    start_time=start_time,
                    end_time=end_time,
                    token_count=p_word_count,
                )
            )
            chunk_counter += 1
            continue

        # If a single paragraph is long (> 145 words), chunk it internally with overlap
        sentences = split_into_sentences(para)
        curr_sentences: list[str] = []
        curr_words = 0

        for sentence in sentences:
            s_words = len(sentence.split())
            curr_sentences.append(sentence)
            curr_words += s_words

            if curr_words >= target_words:
                chunk_body = " ".join(curr_sentences).strip()
                chunk_id = f"{document_id}_chk_{chunk_counter:03d}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        modality=modality,
                        text=chunk_body,
                        page_number=page_number,
                        start_time=start_time,
                        end_time=end_time,
                        token_count=len(chunk_body.split()),
                    )
                )
                chunk_counter += 1

                # Carry over trailing sentences summing up to ~overlap_words
                overlap_sentences: list[str] = []
                overlap_count = 0
                for prev_s in reversed(curr_sentences):
                    cnt = len(prev_s.split())
                    if overlap_count + cnt > overlap_words and overlap_sentences:
                        break
                    overlap_sentences.insert(0, prev_s)
                    overlap_count += cnt

                curr_sentences = overlap_sentences
                curr_words = overlap_count

        if curr_sentences:
            remaining_text = " ".join(curr_sentences).strip()
            rem_words = len(remaining_text.split())
            if chunks and rem_words < 25:
                # Merge small residual tail into the last chunk of this paragraph
                last_chunk = chunks[-1]
                last_chunk.text = clean_text(last_chunk.text + " " + remaining_text)
                last_chunk.token_count = len(last_chunk.text.split())
            else:
                chunk_id = f"{document_id}_chk_{chunk_counter:03d}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        modality=modality,
                        text=remaining_text,
                        page_number=page_number,
                        start_time=start_time,
                        end_time=end_time,
                        token_count=rem_words,
                    )
                )
                chunk_counter += 1

    return chunks


# Alias for backward compatibility
chunk_text = chunk_document


def chunk_audio_segments(
    segments: list[dict],
    document_id: str,
    modality: str = "mp4",
    target_duration: float = 25.0,
    max_words: int = 140,
) -> list[TextChunk]:
    """Groups Whisper transcription segments into ~20-30s / ~80-150 word windows.

    Preserves tight, accurate start_time and end_time boundaries for cross-modal matching.
    """
    if not segments:
        return []

    chunks: list[TextChunk] = []
    chunk_counter = 1

    current_group: list[dict] = []
    current_words = 0

    for seg in segments:
        seg_text = clean_text(seg.get("text", ""))
        if not seg_text:
            continue

        words_in_seg = len(seg_text.split())
        current_group.append({**seg, "clean_text": seg_text, "words": words_in_seg})
        current_words += words_in_seg

        group_duration = current_group[-1]["end"] - current_group[0]["start"]

        if group_duration >= target_duration or current_words >= max_words:
            chunk_text_str = " ".join(s["clean_text"] for s in current_group).strip()
            chunk_id = f"{document_id}_chk_{chunk_counter:03d}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    modality=modality,
                    text=chunk_text_str,
                    page_number=None,
                    start_time=round(float(current_group[0]["start"]), 2),
                    end_time=round(float(current_group[-1]["end"]), 2),
                    token_count=len(chunk_text_str.split()),
                )
            )
            chunk_counter += 1
            current_group = []
            current_words = 0

    # Flush final segment group
    if current_group:
        chunk_text_str = " ".join(s["clean_text"] for s in current_group).strip()
        chunk_id = f"{document_id}_chk_{chunk_counter:03d}"
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                modality=modality,
                text=chunk_text_str,
                page_number=None,
                start_time=round(float(current_group[0]["start"]), 2),
                end_time=round(float(current_group[-1]["end"]), 2),
                token_count=len(chunk_text_str.split()),
            )
        )

    return chunks
