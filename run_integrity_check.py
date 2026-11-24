"""Main CLI Runner for Multimodal Submission Integrity & Plagiarism Detection Engine.

Usage:
    python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
"""

import argparse
import logging
from pathlib import Path
import sys

from src.detector import analyze_document, detect_boilerplate_chunks
from src.extractors.dispatcher import extract_document
from src.indexer import VectorIndex
from src.reporter import (
    generate_document_json,
    generate_document_markdown,
    generate_summary_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("integrity_engine")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the integrity check pipeline."""
    parser = argparse.ArgumentParser(
        description="Multimodal Submission Integrity & Plagiarism Detection Engine"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./sample_submissions",
        help="Directory containing student/participant submissions (PDF, DOCX, MP4).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./reports",
        help="Directory where individual reports and summary_matrix will be written.",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="./reference_corpus",
        help="Directory containing external reference corpus texts (.txt).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for plagiarism matches (default: 0.75).",
    )
    return parser.parse_args()


def main() -> None:
    """Orchestrates ingestion, vector indexing, integrity analysis, and reporting."""
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ref_dir = Path(args.reference_dir).resolve()

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MULTIMODAL SUBMISSION INTEGRITY & PLAGIARISM DETECTION ENGINE")
    print("=" * 80)
    print(f"Input Directory:     {input_dir}")
    print(f"Reference Corpus:    {ref_dir}")
    print(f"Output Directory:    {output_dir}")
    print(f"Plagiarism Threshold:{args.threshold}")
    print("=" * 80)

    # 1. Ingest Submissions
    print("\n[1/5] Ingesting submissions...")
    submission_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() != ".json"
    ])

    doc_results = []
    for f in submission_files:
        res = extract_document(f)
        doc_results.append(res)
        status_symbol = "✓" if res.status == "ok" else ("!" if res.status == "warning" else "✗")
        print(f"  [{status_symbol}] {f.name:<25} ({res.modality}) -> {len(res.chunks)} chunks, status={res.status}")

    # 2. Ingest Reference Corpus
    print("\n[2/5] Ingesting reference corpus...")
    ref_results = []
    if ref_dir.exists():
        ref_files = sorted([f for f in ref_dir.iterdir() if f.is_file()])
        for rf in ref_files:
            r_res = extract_document(rf)
            ref_results.append(r_res)
            print(f"  [✓] {rf.name:<25} -> {len(r_res.chunks)} chunks")
    else:
        logger.warning(f"Reference corpus directory not found: {ref_dir}. External checks will be skipped.")

    # 3. Vector Indexing
    print("\n[3/5] Building FAISS Vector Indexes...")
    peer_index = VectorIndex("peer_index")
    for d in doc_results:
        if d.status == "ok" and d.chunks:
            peer_index.add_chunks(d.chunks)
    print(f"  Peer Index: {len(peer_index)} chunks indexed.")

    external_index = VectorIndex("external_index")
    for r in ref_results:
        if r.status == "ok" and r.chunks:
            external_index.add_chunks(r.chunks)
    print(f"  External Index: {len(external_index)} chunks indexed.")

    # 4. Boilerplate Filtering & Multi-Signal Detection
    print("\n[4/5] Running Boilerplate Filtering & Multi-Signal Detection...")
    valid_docs = [d for d in doc_results if d.status == "ok" and d.chunks]
    boilerplate_ids = detect_boilerplate_chunks(peer_index, valid_docs, similarity_threshold=0.90)

    reports = []
    for doc in doc_results:
        rep = analyze_document(
            doc=doc,
            peer_index=peer_index,
            external_index=external_index,
            boilerplate_ids=boilerplate_ids,
            similarity_threshold=args.threshold,
        )
        reports.append(rep)

    # 5. Report Generation
    print("\n[5/5] Generating Audit Reports and Summary Matrix...")
    for rep in reports:
        doc_stem = Path(rep["document_id"]).stem
        json_path = output_dir / f"{doc_stem}_report.json"
        md_path = output_dir / f"{doc_stem}_report.md"
        generate_document_json(rep, json_path)
        generate_document_markdown(rep, md_path)

    generate_summary_matrix(reports, output_dir)
    print(f"  Reports generated cleanly in: {output_dir}")

    # Terminal Summary Table
    print("\n" + "=" * 116)
    print("INTEGRITY ASSESSMENT SUMMARY TABLE")
    print("=" * 116)
    header = (
        f"{'Document':<24} | {'Status':<7} | {'Risk':<11} | {'Peer Sim':<8} | "
        f"{'MM Sim':<8} | {'Ext Sim':<8} | {'AI Prob':<8} | {'Boilerplate'}"
    )
    print(header)
    print("-" * 116)
    for r in reports:
        sc = r.get("scores", {})
        print(
            f"{r['document_id']:<24} | "
            f"{r['status'].upper():<7} | "
            f"{r['total_integrity_risk']:<11} | "
            f"{sc.get('peer_similarity_percentage', 0.0):>7.1f}% | "
            f"{sc.get('multimodal_similarity_percentage', 0.0):>7.1f}% | "
            f"{sc.get('external_similarity_percentage', 0.0):>7.1f}% | "
            f"{sc.get('ai_generated_probability', 0.0):>7.1f}% | "
            f"{r.get('boilerplate_chunks_discounted', 0):>6} discounted"
        )
    print("=" * 116)
    print(f"\nAudit complete! Review reports at: {output_dir / 'summary_matrix.md'}")


if __name__ == "__main__":
    main()
