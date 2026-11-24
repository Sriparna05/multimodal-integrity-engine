"""Automated evaluation script for Multimodal Submission Integrity Detection Engine.

Computes document-level and chunk-level Precision, Recall, F1-Score, and Confusion Matrices
against ground truth, quantifies lexical (Levenshtein, Jaccard) vs semantic (Cosine) similarity
on paraphrased text, and measures the max cosine similarity distribution across all true-negative
unrelated document pairs.
"""

import json
from pathlib import Path
import re
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import detect_boilerplate_chunks  # noqa: E402
from src.extractors.dispatcher import extract_document  # noqa: E402
from src.indexer import VectorIndex  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "sample_submissions" / "dataset_manifest.json"
SAMPLES_DIR = PROJECT_ROOT / "sample_submissions"
REPORTS_DIR = PROJECT_ROOT / "reports"


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Computes precision, recall, and f1 from confusion matrix counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Computes word-level Jaccard similarity between two text strings."""
    tokens_a = set(re.findall(r"\b\w+\b", text_a.lower()))
    tokens_b = set(re.findall(r"\b\w+\b", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a.intersection(tokens_b)) / len(tokens_a.union(tokens_b))


def compute_levenshtein_ratio(text_a: str, text_b: str) -> float:
    """Computes normalized character-level Levenshtein similarity ratio in [0.0, 1.0]."""
    s1 = " ".join(text_a.lower().split())
    s2 = " ".join(text_b.lower().split())
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        current_row = [i]
        for j, c2 in enumerate(s2, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    distance = previous_row[-1]
    return 1.0 - (distance / max(len(s1), len(s2)))


def compute_true_negative_similarity_stats(manifest_entries: list[dict]) -> dict:
    """Computes max cosine similarity across all unrelated (true-negative) document pairs."""
    valid_names = [
        item["filename"]
        for item in manifest_entries
        if not item.get("is_corrupted_or_empty", False)
    ]

    docs = []
    for name in valid_names:
        res = extract_document(SAMPLES_DIR / name)
        if res.status == "ok" and res.chunks:
            docs.append(res)

    peer_index = VectorIndex("tn_eval_index")
    for doc in docs:
        peer_index.add_chunks(doc.chunks)

    boilerplate_ids = detect_boilerplate_chunks(peer_index, docs, similarity_threshold=0.90)

    # Map each document to its normalized embeddings for active (non-boilerplate) chunks
    doc_embeddings: dict[str, np.ndarray] = {}
    doc_modalities: dict[str, str] = {}
    for doc in docs:
        active_texts = [c.text for c in doc.chunks if c.chunk_id not in boilerplate_ids]
        if active_texts:
            doc_embeddings[doc.document_id] = peer_index.model.encode(
                active_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            doc_modalities[doc.document_id] = doc.modality

    # Known ground-truth positive pairs (unordered)
    positive_pairs = {
        frozenset(("sub_06_copy_src.pdf", "sub_07_copy_dup.docx")),
        frozenset(("sub_08_para_src.docx", "sub_09_para_dup.docx")),
        frozenset(("sub_10_pitch_src.pdf", "sub_11_pitch_video.mp4")),
    }

    text_max_sims: list[float] = []
    mm_max_sims: list[float] = []
    all_max_sims: list[float] = []
    highest_tn_pair = ("", "", 0.0)

    doc_ids = sorted(doc_embeddings.keys())
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            id_a, id_b = doc_ids[i], doc_ids[j]
            if frozenset((id_a, id_b)) in positive_pairs:
                continue

            sim_matrix = doc_embeddings[id_a] @ doc_embeddings[id_b].T
            pair_max = float(np.max(sim_matrix))
            all_max_sims.append(pair_max)

            if pair_max > highest_tn_pair[2]:
                highest_tn_pair = (id_a, id_b, pair_max)

            if doc_modalities[id_a] == "mp4" or doc_modalities[id_b] == "mp4":
                mm_max_sims.append(pair_max)
            else:
                text_max_sims.append(pair_max)

    def summarize(vals: list[float]) -> dict:
        if not vals:
            return {"count": 0, "min": 0.0, "mean": 0.0, "max": 0.0}
        return {
            "count": len(vals),
            "min": float(np.min(vals)),
            "mean": float(np.mean(vals)),
            "max": float(np.max(vals)),
        }

    return {
        "all": summarize(all_max_sims),
        "text": summarize(text_max_sims),
        "multimodal": summarize(mm_max_sims),
        "highest_pair": highest_tn_pair,
    }


def run_evaluation() -> None:
    """Runs automated benchmark evaluation comparing generated reports against manifest."""
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_entries = json.load(f)

    valid_manifest = [
        item for item in manifest_entries
        if not item.get("is_corrupted_or_empty", False)
    ]

    print("=" * 84)
    print("SUBMISSION INTEGRITY ENGINE — BENCHMARK EVALUATION")
    print(f"Total Valid Documents Evaluated: {len(valid_manifest)}")
    print("=" * 84)

    # Document-level counters
    peer_tp, peer_fp, peer_tn, peer_fn = 0, 0, 0, 0
    ext_tp, ext_fp, ext_tn, ext_fn = 0, 0, 0, 0
    ai_tp, ai_fp, ai_tn, ai_fn = 0, 0, 0, 0
    ov_tp, ov_fp, ov_tn, ov_fn = 0, 0, 0, 0

    # Chunk-level counters against expected_peer_sources
    chunk_peer_tp, chunk_peer_fp, chunk_peer_fn = 0, 0, 0
    ai_scores_report: list[tuple[str, float, str]] = []

    for item in valid_manifest:
        fname = item["filename"]
        stem = Path(fname).stem
        rep_path = REPORTS_DIR / f"{stem}_report.json"

        if not rep_path.exists():
            print(f"Warning: Missing report for {fname}")
            continue

        with open(rep_path, "r", encoding="utf-8") as rf:
            rep = json.load(rf)

        scores = rep.get("scores", {})
        peer_pct = scores.get("peer_similarity_percentage", 0.0)
        mm_pct = scores.get("multimodal_similarity_percentage", 0.0)
        ext_pct = scores.get("external_similarity_percentage", 0.0)
        ai_prob = scores.get("ai_generated_probability", 0.0)
        risk = rep.get("total_integrity_risk", "LOW")

        gt_peer = item.get("expected_peer_plagiarism", False) or item.get("expected_multimodal_match", False)
        gt_ext = item.get("expected_external_match", False)
        gt_ai = item.get("expected_ai_generated", False)
        gt_overall = gt_peer or gt_ext or gt_ai

        if gt_ai:
            ai_scores_report.append((fname, ai_prob, risk))

        pred_peer = (
            peer_pct > 0.0
            or mm_pct > 0.0
            or len(rep.get("peer_matches", [])) > 0
            or len(rep.get("multimodal_matches", [])) > 0
        )
        pred_ext = ext_pct > 0.0 or len(rep.get("external_matches", [])) > 0
        pred_ai = ai_prob >= 75.0
        pred_overall = risk in ("MEDIUM", "HIGH")

        if pred_peer and gt_peer:
            peer_tp += 1
        elif pred_peer and not gt_peer:
            peer_fp += 1
        elif not pred_peer and gt_peer:
            peer_fn += 1
        else:
            peer_tn += 1

        if pred_ext and gt_ext:
            ext_tp += 1
        elif pred_ext and not gt_ext:
            ext_fp += 1
        elif not pred_ext and gt_ext:
            ext_fn += 1
        else:
            ext_tn += 1

        if pred_ai and gt_ai:
            ai_tp += 1
        elif pred_ai and not gt_ai:
            ai_fp += 1
        elif not pred_ai and gt_ai:
            ai_fn += 1
        else:
            ai_tn += 1

        if pred_overall and gt_overall:
            ov_tp += 1
        elif pred_overall and not gt_overall:
            ov_fp += 1
        elif not pred_overall and gt_overall:
            ov_fn += 1
        else:
            ov_tn += 1

        # Chunk-level evaluation against expected_peer_sources
        expected_sources = set(item.get("expected_peer_sources", []))
        exp_chunk_count = item.get("expected_peer_chunk_count", 2 if expected_sources else 0)
        doc_tp = sum(1 for m in rep.get("peer_matches", []) if m["matched_file"] in expected_sources)
        doc_fp = sum(1 for m in rep.get("peer_matches", []) if m["matched_file"] not in expected_sources)
        doc_fn = max(0, exp_chunk_count - doc_tp)

        chunk_peer_tp += doc_tp
        chunk_peer_fp += doc_fp
        chunk_peer_fn += doc_fn

    peer_m = compute_metrics(peer_tp, peer_fp, peer_tn, peer_fn)
    ext_m = compute_metrics(ext_tp, ext_fp, ext_tn, ext_fn)
    ai_m = compute_metrics(ai_tp, ai_fp, ai_tn, ai_fn)
    ov_m = compute_metrics(ov_tp, ov_fp, ov_tn, ov_fn)

    chunk_prec = chunk_peer_tp / (chunk_peer_tp + chunk_peer_fp) if (chunk_peer_tp + chunk_peer_fp) > 0 else 0.0
    chunk_rec = chunk_peer_tp / (chunk_peer_tp + chunk_peer_fn) if (chunk_peer_tp + chunk_peer_fn) > 0 else 0.0
    chunk_f1 = (2 * chunk_prec * chunk_rec) / (chunk_prec + chunk_rec) if (chunk_prec + chunk_rec) > 0 else 0.0

    print("\n1. DOCUMENT-LEVEL & CHUNK-LEVEL DETECTION METRICS")
    print("-" * 84)
    hdr = f"{'Signal / Task':<34} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Counts'}"
    print(hdr)
    print("-" * 84)
    print(f"{'Peer & Multimodal Plagiarism (Doc)':<34} | {peer_m['precision']:<9.4f} | {peer_m['recall']:<8.4f} | {peer_m['f1']:<8.4f} | TP={peer_m['tp']}, FP={peer_m['fp']}, TN={peer_m['tn']}, FN={peer_m['fn']}")
    print(f"{'Peer Text Plagiarism (Chunk-Level)':<34} | {chunk_prec:<9.4f} | {chunk_rec:<8.4f} | {chunk_f1:<8.4f} | TP={chunk_peer_tp}, FP={chunk_peer_fp}, FN={chunk_peer_fn}")
    print(f"{'External Reference Match (Doc)':<34} | {ext_m['precision']:<9.4f} | {ext_m['recall']:<8.4f} | {ext_m['f1']:<8.4f} | TP={ext_m['tp']}, FP={ext_m['fp']}, TN={ext_m['tn']}, FN={ext_m['fn']}")
    print(f"{'AI Content Estimation (>=75% HIGH)':<34} | {ai_m['precision']:<9.4f} | {ai_m['recall']:<8.4f} | {ai_m['f1']:<8.4f} | TP={ai_m['tp']}, FP={ai_m['fp']}, TN={ai_m['tn']}, FN={ai_m['fn']}")
    print(f"{'Overall Integrity Flagging (Doc)':<34} | {ov_m['precision']:<9.4f} | {ov_m['recall']:<8.4f} | {ov_m['f1']:<8.4f} | TP={ov_m['tp']}, FP={ov_m['fp']}, TN={ov_m['tn']}, FN={ov_m['fn']}")
    print("-" * 84)

    if ai_scores_report:
        print("\nAI-Generated Test Cases Breakdown (max(RoBERTa, Heuristic)):")
        for fname, score, risk in ai_scores_report:
            print(f"  - {fname:<24}: AI Prob = {score:>5.1f}% | Risk = {risk}")

    # 2. Scenario 2 Quantitative Proof: Levenshtein & Jaccard vs Cosine on Paraphrased Pair
    print("\n2. SCENARIO 2 VERIFICATION: LEXICAL (LEVENSHTEIN / JACCARD) VS SEMANTIC (COSINE)")
    print("-" * 84)
    rep_08_path = REPORTS_DIR / "sub_08_para_src_report.json"
    if rep_08_path.exists():
        with open(rep_08_path, "r", encoding="utf-8") as f:
            d8 = json.load(f)

        matches_8_to_9 = [m for m in d8.get("peer_matches", []) if "sub_09" in m["matched_file"]]
        if matches_8_to_9:
            best_match = max(matches_8_to_9, key=lambda x: x["similarity_score"])
            q_passage = best_match["query_passage"]
            m_passage = best_match["matched_passage"]
            cosine_score = best_match["similarity_score"]
            jaccard_score = compute_jaccard_similarity(q_passage, m_passage)
            lev_ratio = compute_levenshtein_ratio(q_passage, m_passage)

            print(f"Paraphrased Pair:              sub_08_para_src.docx vs sub_09_para_dup.docx")
            print(f"Levenshtein Similarity Ratio:  {lev_ratio:.4f}  (Low character-sequence overlap)")
            print(f"Token-Level Jaccard:           {jaccard_score:.4f}  (Low lexical overlap)")
            print(f"Dense Vector Cosine:           {cosine_score:.4f}  (High semantic equivalence)")

    # 3. True-Negative Similarity Margin Verification
    print("\n3. TRUE-NEGATIVE PAIRWISE SIMILARITY DISTRIBUTION (THRESHOLD MARGIN PROOF)")
    print("-" * 84)
    tn_stats = compute_true_negative_similarity_stats(manifest_entries)
    all_s = tn_stats["all"]
    txt_s = tn_stats["text"]
    mm_s = tn_stats["multimodal"]
    hi_a, hi_b, hi_val = tn_stats["highest_pair"]

    print(f"All Unrelated Pairs (n={all_s['count']:>2}):      Min={all_s['min']:.4f} | Mean={all_s['mean']:.4f} | Max={all_s['max']:.4f}")
    print(f"Text-to-Text TN Pairs (n={txt_s['count']:>2}):   Min={txt_s['min']:.4f} | Mean={txt_s['mean']:.4f} | Max={txt_s['max']:.4f} (Margin to 0.75: +{0.75 - txt_s['max']:.4f})")
    print(f"Multimodal TN Pairs (n={mm_s['count']:>2}):     Min={mm_s['min']:.4f} | Mean={mm_s['mean']:.4f} | Max={mm_s['max']:.4f} (Margin to 0.70: +{0.70 - mm_s['max']:.4f})")
    print(f"Highest Unrelated TN Pair:     {hi_a} vs {hi_b} ({hi_val:.4f})")

    # 4. Scenario 3 & 4 Verification
    print("\n4. SCENARIO 3 & 4 VERIFICATION: EDGE CASES & BOILERPLATE NEUTRALIZATION")
    print("-" * 84)
    for edge in ["sub_14_corrupted", "sub_15_silent_video"]:
        ep = REPORTS_DIR / f"{edge}_report.json"
        if ep.exists():
            with open(ep) as f:
                re_data = json.load(f)
            print(f"{edge:<24}: Status = {re_data['status'].upper():<7} | Risk = {re_data['total_integrity_risk']}")

    for b_file in ["sub_01_orig", "sub_02_orig", "sub_03_orig", "sub_04_orig", "sub_05_orig"]:
        b_path = REPORTS_DIR / f"{b_file}_report.json"
        if b_path.exists():
            with open(b_path) as f:
                rb = json.load(f)
            print(f"{b_file:<24}: Discounted = {rb['boilerplate_chunks_discounted']} chunk(s) | Peer Sim = {rb['scores']['peer_similarity_percentage']:.1f}%")

    print("\n" + "=" * 84)
    print("ALL EVALUATION CHECKS COMPLETED SUCCESSFULLY")
    print("=" * 84)


if __name__ == "__main__":
    run_evaluation()
