"""Smoke test script for multimodal ingestion, normalization, and chunking pipeline."""

from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.dispatcher import extract_document
from src.extractors.mp4_extractor import format_timestamp


def run_smoke_test() -> None:
    """Executes the ingestion pipeline across all sample submissions and reference corpus."""
    sample_dir = PROJECT_ROOT / "sample_submissions"
    ref_dir = PROJECT_ROOT / "reference_corpus"

    # Filter out manifest json
    sample_files = sorted([f for f in sample_dir.iterdir() if f.is_file() and f.suffix != ".json"])
    ref_files = sorted([f for f in ref_dir.iterdir() if f.is_file()])

    print("=" * 95)
    print("MULTIMODAL INGESTION PIPELINE SMOKE TEST")
    print("=" * 95)

    header = f"{'Filename':<26} | {'Modality':<8} | {'Status':<8} | {'Chunks':<6} | {'Metadata / Details'}"
    print(header)
    print("-" * 95)

    results = {}

    for file_path in sample_files:
        result = extract_document(file_path)
        results[file_path.name] = result

        # Format metadata summary
        if result.status == "error":
            meta_str = f"ERROR: {result.error_message}"
        elif result.status == "warning":
            meta_str = f"WARNING: {result.error_message}"
        else:
            if result.modality == "mp4":
                times = [f"{format_timestamp(c.start_time or 0)}-{format_timestamp(c.end_time or 0)}" for c in result.chunks]
                meta_str = f"Spoken time: {', '.join(times)} ({len(result.full_text.split())} words)"
            elif result.modality in ("pdf", "docx"):
                pages = sorted(list({c.page_number for c in result.chunks if c.page_number is not None}))
                page_info = f"Pages: {min(pages)}-{max(pages)}" if pages else "Pages: N/A"
                meta_str = f"{page_info} ({len(result.full_text.split())} words)"
            else:
                meta_str = f"{len(result.full_text.split())} words"

        print(f"{file_path.name:<26} | {result.modality:<8} | {result.status:<8} | {len(result.chunks):<6} | {meta_str}")

    print("-" * 95)
    print(f"\n--- REFERENCE CORPUS INGESTION ({len(ref_files)} files) ---")
    for ref_file in ref_files:
        ref_res = extract_document(ref_file)
        print(f"{ref_file.name:<26} | {ref_res.modality:<8} | {ref_res.status:<8} | {len(ref_res.chunks):<6} | Words: {len(ref_res.full_text.split())}")

    print("\n" + "=" * 95)
    print("VERIFICATION CHECKS")
    print("=" * 95)

    # Check 1: sub_14_corrupted.pdf handled gracefully
    sub_14 = results.get("sub_14_corrupted.pdf")
    assert sub_14 is not None, "sub_14_corrupted.pdf not found in results"
    assert sub_14.status in ("error", "warning"), f"sub_14 expected error/warning status, got {sub_14.status}"
    assert len(sub_14.chunks) == 0, f"sub_14 expected 0 chunks, got {len(sub_14.chunks)}"
    print("[PASS] Check 1: sub_14_corrupted.pdf handled gracefully (0 chunks, status='error')")

    # Check 2: sub_15_silent_video.mp4 handled gracefully
    sub_15 = results.get("sub_15_silent_video.mp4")
    assert sub_15 is not None, "sub_15_silent_video.mp4 not found in results"
    assert sub_15.status in ("warning", "error"), f"sub_15 expected warning/error status, got {sub_15.status}"
    assert len(sub_15.chunks) == 0, f"sub_15 expected 0 chunks, got {len(sub_15.chunks)}"
    print("[PASS] Check 2: sub_15_silent_video.mp4 handled gracefully (0 chunks, status='warning')")

    # Check 3: sub_11_pitch_video.mp4 produces clean transcript with timestamps
    sub_11 = results.get("sub_11_pitch_video.mp4")
    assert sub_11 is not None, "sub_11_pitch_video.mp4 not found in results"
    assert sub_11.status == "ok", f"sub_11 expected status='ok', got {sub_11.status}"
    assert len(sub_11.chunks) > 0, "sub_11 expected at least 1 chunk"
    first_chunk = sub_11.chunks[0]
    assert first_chunk.start_time is not None and first_chunk.end_time is not None, "sub_11 chunk missing start/end timestamps"
    assert first_chunk.end_time > first_chunk.start_time, "sub_11 end_time must be greater than start_time"
    assert len(first_chunk.text) > 30, "sub_11 chunk text appears empty or too short"
    print(f"[PASS] Check 3: sub_11_pitch_video.mp4 transcribed cleanly:")
    print(f"       Duration: {format_timestamp(first_chunk.start_time)} -> {format_timestamp(first_chunk.end_time)}")
    print(f"       Transcript: \"{first_chunk.text}\"")

    # Display cross-modal text comparison
    sub_10 = results.get("sub_10_pitch_src.pdf")
    if sub_10 and len(sub_10.chunks) >= 2:
        # Find page 2 chunk
        p2_chunks = [c for c in sub_10.chunks if c.page_number == 2]
        if p2_chunks:
            print("\n[INFO] Cross-Modal Source (sub_10_pitch_src.pdf Page 2 Text):")
            print(f"       \"{p2_chunks[0].text[:120]}...\"")

    print("\nALL SMOKE TEST VERIFICATIONS PASSED CLEANLY!")


if __name__ == "__main__":
    run_smoke_test()
