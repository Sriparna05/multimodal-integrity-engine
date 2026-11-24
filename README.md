# Multimodal Submission Integrity & Plagiarism Detection Engine

A modular, lightweight Python engine designed to detect academic and competition submission integrity risks across text, documents, and spoken audio/video.

The system analyzes PDF, DOCX, and MP4 submissions using dense semantic vector search, automated boilerplate neutralizing filters, cross-modal speech transcription, external reference comparison, and AI-generated text estimation. All findings are exported as explainable JSON audit trails and human-readable Markdown reports.

---

## Architecture Overview

```
                                      [ Incoming Submissions ]
                                        (PDF, DOCX, MP4, TXT)
                                                  │
                                                  ▼
                                      [ Multimodal Ingestion ]
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
             [ PyMuPDF ]                   [ python-docx ]             [ FFmpeg + Whisper ]
          Page-by-page text            Paragraph & table text          16kHz mono WAV extraction
         (Preserves page num)        (~350 words/page estimate)       (Preserves start/end times)
                    │                             │                             │
                    └─────────────────────────────┼─────────────────────────────┘
                                                  ▼
                                     [ Text Normalization ]
                        Unicode cleanup, line unwrapping, number/hyphen norm
                                                  │
                                                  ▼
                                   [ Paragraph-Aware Chunker ]
                                  ~80–120 words/chunk (tight)
                                   Preserves page/time metadata
                                                  │
                                                  ▼
                                     [ Dense Vector Encoder ]
                             sentence-transformers/all-MiniLM-L6-v2
                                  (384-dim, L2-normalized)
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
             [ Peer Vector Index ]                            [ External Reference Index ]
               faiss.IndexFlatIP                                   faiss.IndexFlatIP
            (All cohort submissions)                            (Curated reference texts)
                         │                                                 │
                         └────────────────────────┬────────────────────────┘
                                                  ▼
                                   [ Boilerplate Template Filter ]
                                  Identifies shared prompts/rubrics
                            (sim >= 0.90 across >= min_doc_count other docs)
                                  -> Discounted from peer scoring
                                                  │
                                                  ▼
                                   [ Multi-Signal Integrity Engine ]
                     ┌────────────────────────────┼────────────────────────────┐
                     │                            │                            │
             [ Peer Matches ]            [ Cross-Modal Matches ]      [ External Matches ]
            Text-to-Text overlap           Audio-to-Text overlap       Reference corpus overlap
              (sim >= 0.75)                    (sim >= 0.70)                  (sim >= 0.75)
                     │                            │                            │
                     └────────────────────────────┼────────────────────────────┘
                                                  ▼
                                    [ AI Content Estimation ]
                                 max(RoBERTa, Heuristic) Score
                                                  │
                                                  ▼
                                    [ Explainable Reporting ]
                            ├── Per-document JSON audit trail (*_report.json)
                            ├── Human-readable Markdown (*_report.md)
                            └── Cohort summary matrix (summary_matrix.md / .json)
```

---

## Quick Start

### 1. Prerequisites
- **Python >= 3.10** (tested up to Python 3.13 on Windows, Linux, and macOS).
- `ffmpeg` installed on your system PATH (or automatically resolved via `imageio-ffmpeg`).

### 2. Setup Virtual Environment
```bash
# Clone and navigate into the repository
cd multimodal-integrity-engine

# Create virtual environment
python -m venv .venv

# Activate virtual environment (macOS / Linux)
source .venv/bin/activate

# Activate virtual environment (Windows PowerShell / CMD)
.venv\Scripts\activate

# Install pinned core runtime dependencies
pip install -r requirements.txt

# Optional: install dev dependencies only needed to regenerate synthetic data
pip install -r requirements-dev.txt
```

### 3. Generate Synthetic Benchmark Dataset (Optional)
The repository ships with pre-generated sample submissions and reference texts. To regenerate the 16 synthetic submissions and 4 reference corpus articles from scratch (requires `requirements-dev.txt`):
```bash
python scripts/generate_dataset.py
```

### 4. Run Integrity Check Pipeline
Execute the single root command to analyze all submissions and produce explainable audit reports:
```bash
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
```
*(Optional arguments `--reference_dir ./reference_corpus` and `--threshold 0.75` default automatically).*

### 5. Run Automated Benchmark Evaluation
Run the evaluation harness to verify document-level and chunk-level metrics, lexical vs. semantic scores, and true-negative similarity margins:
```bash
python scripts/evaluate.py
```

---

## Project Structure

```
multimodal-integrity-engine/
├── .gitignore
├── EVALUATION.md               # Empirical benchmark report & failure analysis
├── README.md                   # System documentation & scaling architecture
├── requirements.txt            # Pinned core runtime dependencies
├── requirements-dev.txt        # Pinned synthetic dataset generation dependencies
├── run_integrity_check.py      # Main CLI entrypoint
├── reference_corpus/           # Curated external documents for reference overlap check
│   ├── ev_battery_supply_chain.txt
│   ├── quick_commerce_india.txt
│   ├── saas_metrics_guide.txt
│   └── solar_microgrids.txt
├── reports/                    # Generated audit reports & cohort summary matrix
│   ├── summary_matrix.json
│   ├── summary_matrix.md
│   ├── sub_01_orig_report.json
│   ├── sub_01_orig_report.md
│   └── ...
├── sample_submissions/         # Synthetic benchmark dataset (16 files)
│   ├── dataset_manifest.json   # Ground-truth labels & relationships
│   ├── sub_01_orig.pdf         # Original submission with shared prompt header
│   ├── sub_02_orig.docx        # Original submission with shared prompt header
│   ├── sub_03_orig.pdf         # Original submission with shared prompt header
│   ├── sub_04_orig.docx        # Original submission with shared prompt header
│   ├── sub_05_orig.pdf         # Original submission + external copy
│   ├── sub_06_copy_src.pdf     # Exact copy source
│   ├── sub_07_copy_dup.docx    # Exact copy duplicate
│   ├── sub_08_para_src.docx    # Paraphrased source
│   ├── sub_09_para_dup.docx    # Paraphrased duplicate
│   ├── sub_10_pitch_src.pdf    # Cross-modal source (text on page 2)
│   ├── sub_11_pitch_video.mp4  # Cross-modal video (spoken speech)
│   ├── sub_12_orig_video.mp4   # Original video submission
│   ├── sub_13_ai_gen.docx      # Formulaic cliché AI essay
│   ├── sub_14_corrupted.pdf    # 0-byte corrupted PDF edge case
│   ├── sub_15_silent_video.mp4 # Video with no audio track edge case
│   └── sub_16_ai_natural.docx  # Marker-free natural LLM business analysis
├── scripts/
│   ├── evaluate.py             # Automated benchmark evaluation script
│   ├── generate_dataset.py     # Synthetic multimodal dataset generator
│   └── test_ingestion.py       # Ingestion and chunking smoke-test script
└── src/
    ├── __init__.py
    ├── detector.py             # Boilerplate filter & multi-signal scoring engine
    ├── indexer.py              # FAISS vector indexing & sentence-transformers wrapper
    ├── models.py               # TextChunk and DocumentResult dataclasses
    ├── reporter.py             # JSON and Markdown report generation
    ├── text_processing.py      # Text cleaning, normalization & paragraph-aware chunking
    └── extractors/
        ├── __init__.py
        ├── dispatcher.py       # Format router (.pdf, .docx, .mp4, .txt)
        ├── docx_extractor.py   # Word document parser
        ├── mp4_extractor.py    # FFmpeg WAV extraction + faster-whisper transcription
        ├── pdf_extractor.py    # PyMuPDF block parser
        └── txt_extractor.py    # Plaintext parser
```

---

## Detection Signals & Methodology

### 1. Ingestion & Paragraph-Aware Chunking
- **PDF Extraction:** Extracts text by layout blocks using PyMuPDF (`pymupdf`), preserving 1-indexed page numbers. Empty or 0-byte PDFs return `status="error"` cleanly.
- **DOCX Extraction:** Reads paragraphs and table cells using `python-docx`. Approximates page numbers using a ~350 words-per-page model.
- **MP4 Audio/Video Extraction:** Extracts a mono 16kHz WAV track via `imageio-ffmpeg` into a normalized float32 NumPy array and transcribes with `faster-whisper` (`base.en`, falling back to `tiny.en` if needed) using a business/financial `initial_prompt`. Silent or no-audio videos return `status="warning"` safely.
- **Normalization & Chunking:** `clean_text()` unwraps intra-paragraph line breaks and normalizes spoken/written variants (hyphenation like `pro-forma` → `pro forma`, spelled-out numbers like `eighteen` → `18`, and percentages like `sixty-four percent` → `64%`). Documents are split at paragraph boundaries into ~80–120 word chunks, while MP4 transcripts use 20–30 second segment windows.

### 2. Dense Semantic Vector Indexing
- Encodes chunks with `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional L2-normalized vectors) indexed in two `faiss.IndexFlatIP` instances (`peer_index` and `external_index`).

### 3. Scenario 4: Automated Boilerplate Template Filter
- Scans `peer_index` while strictly excluding each chunk's own `document_id`. Any chunk with cosine similarity `>= 0.90` in at least `min_doc_count = max(2, min(4, ceil(N * 0.35)))` other valid documents is marked boilerplate and discounted from similarity calculations.
- Includes a 100%-discount safety guard so tiny 2–3 document cohorts never have all chunks wiped out.

### 4. Scenario 1: Cross-Modal Plagiarism Detection
- Encodes MP4 speech transcripts into the same 384-dim space as written PDF/DOCX passages, using a dedicated `multimodal_threshold = 0.70` to tolerate minor ASR transcription drift.
- Tracks `multimodal_similarity_percentage` independently of `peer_similarity_percentage` and gates `HIGH` risk whenever any single multimodal match `>= 0.85` or `multimodal_similarity_percentage >= 40%`.
- Tested on `sub_10_pitch_src.pdf` vs `sub_11_pitch_video.mp4` (`0.971` cosine similarity, both flagged `HIGH`).

### 5. Scenario 2: Paraphrase Detection (Lexical vs. Semantic)
- Evaluated on `sub_08_para_src.docx` vs `sub_09_para_dup.docx`:
  - **Levenshtein Similarity Ratio:** `0.6455` (misses threshold)
  - **Token-Level Jaccard Similarity:** `0.4697` (misses threshold)
  - **Dense Vector Cosine Similarity:** `0.8990` (flagged cleanly above `0.75`)

### 6. Scenario 3: Corrupted & Empty Files
- `sub_14_corrupted.pdf` (0 bytes) → `ERROR` / `UNPROCESSED`.
- `sub_15_silent_video.mp4` (no audio stream) → `WARNING` / `UNPROCESSED`.

### 7. AI Content Estimation
- Combines sentence-length variance (burstiness) and transition marker density with `Hello-SimpleAI/chatgpt-detector-roberta` (`max(hf_score, heuristic_score)` with explicit `top_k=None` class extraction), falling back to the heuristic if the transformer cannot be loaded.
- Formulaic synthetic essay `sub_13_ai_gen.docx` is flagged **`95.0%` (`HIGH`)**, while marker-free modern LLM prose `sub_16_ai_natural.docx` scores **`20.0%` (`LOW`)** because 2022 HC3-trained RoBERTa classifies clean business prose as human (`~0.04%` ChatGPT), demonstrating the known limits of zero-shot AI detection.

---

## Output Report Formats

The engine writes standardized outputs to `--output_dir`:

### 1. Per-Submission Audit Trail (`<stem>_report.json`)
Relative filenames only (never host absolute paths) and a single joined `explanations` string alongside structured `explanation_details`. Verbatim excerpt from `reports/sub_10_pitch_src_report.json`:

```json
{
  "document_id": "sub_10_pitch_src.pdf",
  "file_path": "sub_10_pitch_src.pdf",
  "modality": "pdf",
  "status": "ok",
  "total_integrity_risk": "HIGH",
  "scores": {
    "peer_similarity_percentage": 0.0,
    "multimodal_similarity_percentage": 25.0,
    "external_similarity_percentage": 0.0,
    "ai_generated_probability": 20.0
  },
  "boilerplate_chunks_discounted": 0,
  "peer_matches": [],
  "multimodal_matches": [
    {
      "matched_file": "sub_11_pitch_video.mp4",
      "similarity_score": 0.971,
      "query_passage": "Our pro forma financial model projects reaching operating break even by month 18, underpinned by an average order value expansion to 42 dollars. We anticipate blended gross margins will stabilize at 64% as centralized procurement volumes lower unit hardware costs. Net working capital requirements will peak during the third fiscal quarter before stabilizing alongside sustained free cash flow.",
      "matched_passage": "Our pro forma financial model projects reaching operating break even by month 18 underpinned by an average order value expansion to $42. We anticipate blended gross margins will stabilize at 64% as centralized procurement volumes lower unit hardware costs. Networking capital requirements will peak during the third fiscal quarter before stabilizing alongside sustained free cash flow.",
      "location": {
        "query_page": 2,
        "query_timestamp": null,
        "matched_page": null,
        "matched_timestamp": "00:00:00-00:00:26"
      }
    }
  ],
  "external_matches": [],
  "explanations": "Cross-modal match identified against sub_11_pitch_video.mp4 (spoken audio/text alignment score: 0.971).",
  "explanation_details": [
    "Cross-modal match identified against sub_11_pitch_video.mp4 (spoken audio/text alignment score: 0.971)."
  ]
}
```

### 2. Human-Readable Audit Report (`<stem>_report.md`)
Formatted for fast review with plain-text risk labels (`HIGH`, `MEDIUM`, `LOW`, `UNPROCESSED`), score breakdown tables, side-by-side matched quotes, and exact page/timestamp citations.

### 3. Cohort Summary Matrix (`summary_matrix.md` & `summary_matrix.json`)
Current cohort output from `reports/summary_matrix.md`:

| Document | Status | Risk Level | Peer Sim (%) | Multimodal Sim (%) | Ext Sim (%) | AI Prob (%) | Boilerplate Filtered | Flagged Sources |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `sub_01_orig.pdf` | OK | LOW | 0.0% | 0.0% | 0.0% | 41.8% | 1 | None |
| `sub_02_orig.docx` | OK | LOW | 0.0% | 0.0% | 0.0% | 10.0% | 1 | None |
| `sub_03_orig.pdf` | OK | LOW | 0.0% | 0.0% | 0.0% | 10.0% | 1 | None |
| `sub_04_orig.docx` | OK | LOW | 0.0% | 0.0% | 0.0% | 20.0% | 1 | None |
| `sub_05_orig.pdf` | OK | MEDIUM | 0.0% | 0.0% | 33.3% | 35.5% | 1 | ev_battery_supply_chain.txt |
| `sub_06_copy_src.pdf` | OK | HIGH | 50.0% | 0.0% | 0.0% | 20.0% | 0 | sub_07_copy_dup.docx |
| `sub_07_copy_dup.docx` | OK | HIGH | 50.0% | 0.0% | 0.0% | 20.0% | 0 | sub_06_copy_src.pdf |
| `sub_08_para_src.docx` | OK | HIGH | 50.0% | 0.0% | 0.0% | 25.1% | 0 | sub_09_para_dup.docx |
| `sub_09_para_dup.docx` | OK | HIGH | 50.0% | 0.0% | 0.0% | 10.0% | 0 | sub_08_para_src.docx |
| `sub_10_pitch_src.pdf` | OK | HIGH | 0.0% | 25.0% | 0.0% | 20.0% | 0 | sub_11_pitch_video.mp4 |
| `sub_11_pitch_video.mp4` | OK | HIGH | 0.0% | 100.0% | 0.0% | 20.0% | 0 | sub_10_pitch_src.pdf |
| `sub_12_orig_video.mp4` | OK | LOW | 0.0% | 0.0% | 0.0% | 20.0% | 0 | None |
| `sub_13_ai_gen.docx` | OK | HIGH | 0.0% | 0.0% | 0.0% | 95.0% | 0 | Synthetic LLM |
| `sub_14_corrupted.pdf` | ERROR | UNPROCESSED | 0.0% | 0.0% | 0.0% | 0.0% | 0 | None |
| `sub_15_silent_video.mp4` | WARNING | UNPROCESSED | 0.0% | 0.0% | 0.0% | 0.0% | 0 | None |
| `sub_16_ai_natural.docx` | OK | LOW | 0.0% | 0.0% | 0.0% | 20.0% | 0 | None |

---

## Benchmark Evaluation Results

Evaluated against `sample_submissions/dataset_manifest.json` using `python scripts/evaluate.py`:

```
====================================================================================
SUBMISSION INTEGRITY ENGINE — BENCHMARK EVALUATION
Total Valid Documents Evaluated: 14
====================================================================================

1. DOCUMENT-LEVEL & CHUNK-LEVEL DETECTION METRICS
------------------------------------------------------------------------------------
Signal / Task                      | Precision | Recall   | F1-Score | Counts
------------------------------------------------------------------------------------
Peer & Multimodal Plagiarism (Doc) | 1.0000    | 1.0000   | 1.0000   | TP=6, FP=0, TN=8, FN=0
Peer Text Plagiarism (Chunk-Level) | 1.0000    | 1.0000   | 1.0000   | TP=8, FP=0, FN=0
External Reference Match (Doc)     | 1.0000    | 1.0000   | 1.0000   | TP=1, FP=0, TN=13, FN=0
AI Content Estimation (>=75% HIGH) | 1.0000    | 0.5000   | 0.6667   | TP=1, FP=0, TN=12, FN=1
Overall Integrity Flagging (Doc)   | 1.0000    | 0.8889   | 0.9412   | TP=8, FP=0, TN=5, FN=1
------------------------------------------------------------------------------------

2. SCENARIO 2 VERIFICATION: LEXICAL (LEVENSHTEIN / JACCARD) VS SEMANTIC (COSINE)
------------------------------------------------------------------------------------
Paraphrased Pair:              sub_08_para_src.docx vs sub_09_para_dup.docx
Levenshtein Similarity Ratio:  0.6455  (Low character-sequence overlap)
Token-Level Jaccard:           0.4697  (Low lexical overlap)
Dense Vector Cosine:           0.8990  (High semantic equivalence)

3. TRUE-NEGATIVE PAIRWISE SIMILARITY DISTRIBUTION (THRESHOLD MARGIN PROOF)
------------------------------------------------------------------------------------
All Unrelated Pairs (n=88):      Min=0.1328 | Mean=0.3332 | Max=0.5920
Text-to-Text TN Pairs (n=64):   Min=0.1328 | Mean=0.3512 | Max=0.5920 (Margin to 0.75: +0.1580)
Multimodal TN Pairs (n=24):     Min=0.1429 | Mean=0.2854 | Max=0.5729 (Margin to 0.70: +0.1271)
Highest Unrelated TN Pair:     sub_01_orig.pdf vs sub_08_para_src.docx (0.5920)
```

See [EVALUATION.md](./EVALUATION.md) for detailed failure mode breakdowns, true-negative margin analysis, and model trade-off comparisons.

---

## Production Architecture Brief: Scaling to 10,000 Concurrent Submissions

Scaling this engine from a local cohort CLI to a production platform handling 10,000 concurrent submissions requires separating ingestion, worker pools, vector storage, and caching.

### 1. High-Throughput Distributed Architecture

```
[ 10,000 Submissions ] ──► [ Cloudflare / API Gateway ]
                                      │
                                      ▼
                             [ Ingestion Service ]
                              (FastAPI Stateless)
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
           [ Object Storage ]                    [ Message Broker ]
           (S3 / GCS Buckets)                   (Kafka / AWS SQS)
           Raw PDF, DOCX, MP4                    Job payload + metadata
                                                         │
                         ┌───────────────────────────────┴───────────────────────────────┐
                         ▼                                                               ▼
               [ CPU Worker Pool ]                                             [ GPU Worker Pool ]
           (Celery / Ray on Kubernetes)                                   (T4 / A10G GPU Instances)
           - PDF Layout parsing (PyMuPDF)                                  - Audio extraction & ASR
           - DOCX parsing (python-docx)                                      (distil-whisper batched)
           - Text cleaning & chunking                                     - Batched Embedding Generation
           - SHA-256 chunk hashing                                          (all-MiniLM-L6-v2 via TensorRT)
                         │                                                               │
                         └───────────────────────────────┬───────────────────────────────┘
                                                         ▼
                                             [ Distributed Vector DB ]
                                            (Qdrant / Milvus / pgvector)
                                            Partitioned by competition_id
                                                         │
                                                         ▼
                                            [ Multi-Signal Evaluator ]
                                           Boilerplate, Peer, Ext, AI
                                                         │
                                                         ▼
                                            [ Result Store & Webhooks ]
                                          PostgreSQL + Redis (Reports)
```

### 2. Sizing Math & Memory Footprint
- **Document Chunk Volume:** 10,000 submissions × ~15 chunks per document = **150,000 chunks**.
- **Vector Storage Size:** 384 floats (FP32) = 1,536 bytes/vector → $150,000 \times 1,536\text{ bytes} \approx \mathbf{230.4\text{ MB}}$.
- **Key Insight:** The entire cohort vector index fits in under 250 MB of RAM. Vector search is **not** the bottleneck—**ASR audio transcription** is. With ~0.5–1.5s GPU transcription per 20-second clip, 8 NVIDIA T4 GPUs clear 10,000 video pitches in ~15–20 minutes using dynamic batching.

### 3. Vector Database Selection & Partitioning Strategy
- **Index Architecture:** Replace in-memory `faiss.IndexFlatIP` with **Qdrant** or **Milvus** using HNSW graphs (<15ms p99 latency).
- **Cohort Partitioning:** Partition indexes via `competition_id` metadata filters so peer comparisons remain isolated within the active tournament.
- **Historical Archive:** Use **IVF-PQ** for multi-year cross-cohort archives (~75% memory reduction).

### 4. Cost & Latency Optimizations
1. **Deduplication:** Hash uploaded files (SHA-256) to skip redundant extraction and ASR on identical uploads.
2. **MinHash / LSH Pre-Filter:** Catch near-exact lexical copies in $O(1)$ before invoking dense neural searches.
3. **Embedding Cache (Redis):** Cache normalized chunk embeddings by SHA-256 chunk hash.
4. **ASR Optimization:** Run `distil-whisper/distil-large-v3` under TensorRT/ONNX Runtime for ~6x faster transcription on GPU workers.

---

## Assumptions & Engineering Constraints

1. **Chunk Size Deviation (~80–120 Words vs. 200–400 Tokens):**
   - Although the prompt guidelines suggested 200–400 tokens per chunk, `sentence-transformers/all-MiniLM-L6-v2` silently truncates inputs beyond **256 word-piece tokens** (~180–200 English words). Furthermore, the shared case competition charter in our benchmark is ~147 words. Using **~80–120 word paragraph-aware chunks** ensures zero token truncation loss and isolates boilerplate headers into dedicated chunks that can be cleanly discounted.
2. **DOCX Physical Page Number Estimation:**
   - Microsoft Word `.docx` files are dynamic flow XML documents without fixed physical page boundaries. Page numbers are estimated using **~350 words per page**.
3. **Reference Corpus Simulation:**
   - External similarity is evaluated against curated texts in `./reference_corpus/`, simulating an external web or academic index (e.g., CrossRef, arXiv, Semantic Scholar).
4. **AI Generation Scoring Limitations:**
   - Statistical burstiness + zero-shot RoBERTa probabilities are advisory screening signals only, never standalone grounds for disciplinary action.
5. **Boilerplate Detection Frequency Threshold:**
   - Assumes cohort sizes `>= 4` for stable statistical prompt detection; tiny 2–3 document cohorts are protected by `min_doc_count` and a 100%-discount guard.

---

## AI Assistant Disclosure

Generative AI assistance was utilized during this project for drafting initial boilerplate scaffolding and creating realistic synthetic case-study texts. Core detection logic, FAISS vector indexing, ASR pipeline fixes, multi-signal scoring formulas, and empirical evaluation metrics were designed, implemented, and verified locally.
