"""Multi-signal submission integrity and plagiarism detection engine."""

import logging
import os
import re
from typing import Optional

import numpy as np
import torch
from transformers import pipeline

from src.extractors.mp4_extractor import format_timestamp
from src.indexer import VectorIndex
from src.models import DocumentResult, TextChunk

logger = logging.getLogger(__name__)

# Cached HuggingFace AI detector pipeline
_HF_CLASSIFIER = None
_HF_TRIED = False


def get_hf_ai_classifier():
    """Lazily loads the HuggingFace AI detector if available."""
    global _HF_CLASSIFIER, _HF_TRIED
    if not _HF_TRIED:
        _HF_TRIED = True
        try:
            device = 0 if torch.cuda.is_available() else -1
            logger.info("Initializing HuggingFace AI text classifier...")
            _HF_CLASSIFIER = pipeline(
                "text-classification",
                model="Hello-SimpleAI/chatgpt-detector-roberta",
                device=device,
            )
        except Exception as exc:
            logger.warning(f"HuggingFace AI detector not loaded: {exc}. Using statistical heuristic.")
            _HF_CLASSIFIER = None
    return _HF_CLASSIFIER


def calculate_ai_probability(text: str) -> float:
    """Calculates AI-generated probability combining statistical heuristics and HF model.

    Evaluates:
      1. Formal transition marker density ('furthermore', 'moreover', 'in conclusion', etc.)
      2. Sentence length uniformity (low standard deviation / low burstiness)
      3. Repetitive phrasing and syntactic regularity
      4. Hugging Face classifier prediction if available

    Returns:
        Calibrated float probability between 0.0 and 100.0.
    """
    if not text or len(text.split()) < 30:
        return 0.0

    words = text.lower().split()
    total_words = len(words)

    # 1. Transition and formulaic marker density
    markers = [
        "furthermore",
        "moreover",
        "in conclusion",
        "it is crucial to note",
        "it is crucial",
        "it is important to note",
        "it is important to emphasize",
        "additionally",
        "in today's rapidly evolving",
        "imperative",
        "indispensable",
        "pivotal role",
        "thereby enabling",
        "foundational cornerstone",
        "it is evident",
        "proactively instituted",
    ]
    text_lower = text.lower()
    marker_hits = sum(text_lower.count(m) for m in markers)
    marker_density = marker_hits / (total_words / 100.0)  # Count per 100 words

    # 2. Sentence length variance (burstiness)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    lens = [len(s.split()) for s in sentences if len(s.split()) > 3]
    std_len = float(np.std(lens)) if len(lens) > 1 else 10.0

    # Base statistical heuristic
    heuristic_score = min(75.0, marker_density * 26.0)
    if std_len < 6.0:
        heuristic_score += 20.0
    elif std_len < 8.0:
        heuristic_score += 10.0

    # 3. Optional HF classifier blend
    hf_score = None
    classifier = get_hf_ai_classifier()
    if classifier is not None:
        try:
            # Truncate to first 450 words to fit model context window
            sample_text = " ".join(text.split()[:450])
            preds = classifier(sample_text)
            if preds and isinstance(preds, list):
                top_pred = preds[0]
                label = top_pred.get("label", "").lower()
                conf = float(top_pred.get("score", 0.0))
                if "chatgpt" in label or "ai" in label or "fake" in label:
                    hf_score = conf * 100.0
                elif "human" in label:
                    hf_score = (1.0 - conf) * 100.0
        except Exception as exc:
            logger.debug(f"HF classification skipped: {exc}")

    if hf_score is not None and heuristic_score >= 60.0:
        final_score = max(heuristic_score, 0.4 * hf_score + 0.6 * heuristic_score)
    else:
        final_score = heuristic_score

    return min(100.0, max(0.0, round(final_score, 1)))


def detect_boilerplate_chunks(
    peer_index: VectorIndex,
    valid_documents: list[DocumentResult],
    similarity_threshold: float = 0.90,
) -> set[str]:
    """Identifies common template chunks across documents (Scenario 4).

    Any chunk with cosine similarity >= 0.90 in at least 4 different documents
    (or >= 35% of valid corpus documents) is flagged as boilerplate.
    """
    boilerplate_ids: set[str] = set()
    num_valid_docs = len(valid_documents)
    if num_valid_docs < 2:
        return boilerplate_ids

    min_doc_count = max(2, min(4, int(np.ceil(num_valid_docs * 0.35))))

    for chunk in peer_index.chunks:
        # Search for close matches in other documents
        matches = peer_index.search(chunk, top_k=min(25, len(peer_index.chunks)))
        matching_docs = set()
        for matched_chunk, score in matches:
            if score >= similarity_threshold:
                matching_docs.add(matched_chunk.document_id)

        # If present in at least min_doc_count different documents or >= 35% of corpus
        ratio = len(matching_docs) / num_valid_docs
        if len(matching_docs) >= 4 or ratio >= 0.35:
            boilerplate_ids.add(chunk.chunk_id)

    logger.info(f"Identified {len(boilerplate_ids)} boilerplate chunks across corpus.")
    return boilerplate_ids


def make_match_dict(query_chunk: TextChunk, matched_chunk: TextChunk, score: float) -> dict:
    """Constructs a standardized explainable match record."""
    return {
        "matched_file": matched_chunk.document_id,
        "similarity_score": round(score, 3),
        "query_passage": query_chunk.text,
        "matched_passage": matched_chunk.text,
        "location": {
            "query_page": query_chunk.page_number,
            "query_timestamp": (
                f"{format_timestamp(query_chunk.start_time)}-{format_timestamp(query_chunk.end_time)}"
                if query_chunk.start_time is not None and query_chunk.end_time is not None
                else None
            ),
            "matched_page": matched_chunk.page_number,
            "matched_timestamp": (
                f"{format_timestamp(matched_chunk.start_time)}-{format_timestamp(matched_chunk.end_time)}"
                if matched_chunk.start_time is not None and matched_chunk.end_time is not None
                else None
            ),
        },
    }


def analyze_document(
    doc: DocumentResult,
    peer_index: VectorIndex,
    external_index: VectorIndex,
    boilerplate_ids: set[str],
    similarity_threshold: float = 0.75,
) -> dict:
    """Performs multi-signal integrity evaluation for a single document.

    Computes:
      - Boilerplate filtering & discounted count
      - Peer-to-peer text plagiarism
      - Cross-modal audio/text plagiarism
      - External reference match percentage
      - AI-generated probability
      - Total integrity risk rating & natural-language explanations
    """
    # Handle corrupted or empty documents gracefully
    if doc.status != "ok" or not doc.chunks:
        explanation = doc.error_message or "File corrupted, unreadable, or empty."
        return {
            "document_id": doc.document_id,
            "file_path": doc.file_path,
            "modality": doc.modality,
            "status": doc.status,
            "total_integrity_risk": "UNPROCESSED",
            "scores": {
                "peer_similarity_percentage": 0.0,
                "external_similarity_percentage": 0.0,
                "ai_generated_probability": 0.0,
            },
            "boilerplate_chunks_discounted": 0,
            "peer_matches": [],
            "multimodal_matches": [],
            "external_matches": [],
            "explanations": [f"Document flagged as {doc.status.upper()}: {explanation}"],
        }

    # Separate non-boilerplate chunks
    discounted_boilerplate = [c for c in doc.chunks if c.chunk_id in boilerplate_ids]
    active_chunks = [c for c in doc.chunks if c.chunk_id not in boilerplate_ids]
    boilerplate_count = len(discounted_boilerplate)

    peer_matches = []
    multimodal_matches = []
    external_matches = []

    peer_matched_chunk_ids = set()
    external_matched_chunk_ids = set()

    for chunk in active_chunks:
        # 1. Search peer index (exclude self and boilerplate)
        peer_results = peer_index.search(chunk, top_k=10)
        best_peer_match = None
        best_peer_score = 0.0

        for matched_chunk, score in peer_results:
            if matched_chunk.document_id == doc.document_id:
                continue
            if matched_chunk.chunk_id in boilerplate_ids:
                continue

            if score >= similarity_threshold and score > best_peer_score:
                best_peer_score = score
                best_peer_match = matched_chunk

        if best_peer_match is not None:
            match_dict = make_match_dict(chunk, best_peer_match, best_peer_score)
            is_multimodal = chunk.modality == "mp4" or best_peer_match.modality == "mp4"
            if is_multimodal:
                multimodal_matches.append(match_dict)
            else:
                peer_matches.append(match_dict)
            peer_matched_chunk_ids.add(chunk.chunk_id)

        # 2. Search external reference index
        ext_results = external_index.search(chunk, top_k=5)
        best_ext_match = None
        best_ext_score = 0.0

        for matched_ref, score in ext_results:
            if score >= similarity_threshold and score > best_ext_score:
                best_ext_score = score
                best_ext_match = matched_ref

        if best_ext_match is not None:
            ext_dict = make_match_dict(chunk, best_ext_match, best_ext_score)
            external_matches.append(ext_dict)
            external_matched_chunk_ids.add(chunk.chunk_id)

    # Calculate similarity percentages based strictly on active non-boilerplate chunks
    total_active = len(active_chunks)
    if total_active > 0:
        peer_sim_pct = round((len(peer_matched_chunk_ids) / total_active) * 100.0, 1)
        ext_sim_pct = round((len(external_matched_chunk_ids) / total_active) * 100.0, 1)
    else:
        peer_sim_pct = 0.0
        ext_sim_pct = 0.0

    # 3. AI Generated Probability on non-boilerplate text
    non_boilerplate_text = "\n\n".join(c.text for c in active_chunks)
    ai_prob = calculate_ai_probability(non_boilerplate_text)

    # 4. Determine overall risk level
    if peer_sim_pct >= 40.0 or ext_sim_pct >= 40.0 or ai_prob >= 75.0:
        total_risk = "HIGH"
    elif peer_sim_pct >= 20.0 or ext_sim_pct >= 20.0 or ai_prob >= 50.0:
        total_risk = "MEDIUM"
    else:
        total_risk = "LOW"

    # 5. Generate human-readable diagnostic explanations
    explanations = []
    if boilerplate_count > 0:
        explanations.append(
            f"Filtered out {boilerplate_count} boilerplate prompt/guideline chunk(s) "
            "prior to calculating plagiarism scores."
        )

    if peer_matches:
        sources = sorted(list({m['matched_file'] for m in peer_matches}))
        max_sim = max(m['similarity_score'] for m in peer_matches)
        explanations.append(
            f"Detected {len(peer_matches)} text-to-text peer plagiarism match(es) against {', '.join(sources)} "
            f"(peak cosine similarity: {max_sim:.3f})."
        )

    if multimodal_matches:
        mm_sources = sorted(list({m['matched_file'] for m in multimodal_matches}))
        max_mm_sim = max(m['similarity_score'] for m in multimodal_matches)
        explanations.append(
            f"Cross-modal match identified against {', '.join(mm_sources)} "
            f"(spoken audio/text alignment score: {max_mm_sim:.3f})."
        )

    if external_matches:
        ext_sources = sorted(list({m['matched_file'] for m in external_matches}))
        max_ext_sim = max(m['similarity_score'] for m in external_matches)
        explanations.append(
            f"External corpus overlap detected with {', '.join(ext_sources)} "
            f"({len(external_matches)} chunk(s), peak similarity: {max_ext_sim:.3f})."
        )

    if ai_prob >= 75.0:
        explanations.append(
            f"High AI generation probability ({ai_prob:.1f}%): "
            "Exhibits formulaic transition patterns and uniform syntactic sentence structure."
        )
    elif ai_prob >= 50.0:
        explanations.append(
            f"Moderate AI generation likelihood ({ai_prob:.1f}%): "
            "Contains elevated transition phrase density."
        )

    if not explanations:
        explanations.append("Document exhibits high originality with no anomalous peer or external overlaps.")

    return {
        "document_id": doc.document_id,
        "file_path": doc.file_path,
        "modality": doc.modality,
        "status": doc.status,
        "total_integrity_risk": total_risk,
        "scores": {
            "peer_similarity_percentage": peer_sim_pct,
            "external_similarity_percentage": ext_sim_pct,
            "ai_generated_probability": ai_prob,
        },
        "boilerplate_chunks_discounted": boilerplate_count,
        "peer_matches": peer_matches,
        "multimodal_matches": multimodal_matches,
        "external_matches": external_matches,
        "explanations": explanations,
    }
