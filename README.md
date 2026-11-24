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
                                 Unicode cleanup, line unwrapping
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
                                (sim ≥ 0.90 in ≥4 docs or ≥35% cohort)
                                  -> Discounted from peer scoring
                                                  │
                                                  ▼
                                   [ Multi-Signal Integrity Engine ]
                     ┌────────────────────────────┼────────────────────────────┐
                     │                            │                            │
             [ Peer Matches ]            [ Cross-Modal Matches ]      [ External Matches ]
            Text-to-Text overlap           Audio-to-Text overlap       Reference corpus overlap
              (sim ≥ 0.75)                    (sim ≥ 0.75)                  (sim ≥ 0.75)
                     │                            │                            │
                     └────────────────────────────┼────────────────────────────┘
                                                  ▼
                                    [ AI Content Estimation ]
                                 Burstiness + RoBERTa Classifier
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
- Python 3.10 to 3.13 on Windows, Linux, or macOS.
- `ffmpeg` installed on your system PATH (or automatically resolved via `imageio-ffmpeg`).

### 2. Setup Virtual Environment
```powershell
# Clone or navigate to the repository
cd d:\integrity-engine

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 3. Generate Synthetic Benchmark Dataset
The test suite includes a generator that creates 15 synthetic business strategy submissions (including shared prompts, exact copies, paraphrased text, video pitch presentations, AI-generated essays, and corrupted/silent edge cases) plus 4 reference corpus files:
```powershell
python scripts/generate_dataset.py
```

### 4. Run Integrity Check Pipeline
Execute the single root command to analyze all submissions and produce explainable audit reports:
```powershell
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
```
*(Optional arguments `--reference_dir ./reference_corpus` and `--threshold 0.75` default automatically).*

### 5. Run Automated Benchmark Evaluation
Run the automated evaluation harness to verify precision, recall, confusion matrices, and lexical vs. semantic similarity measurements:
```powershell
python scripts/evaluate.py
```

---

## Project Structure

```
integrity-engine/
├── .gitignore
├── EVALUATION.md               # Empirical benchmark report & failure analysis
├── README.md                   # System documentation & scaling architecture
├── requirements.txt            # Python dependencies
├── run_integrity_check.py      # Main CLI entrypoint
├── reference_corpus/           # Curated external documents for citation/overlap check
│   ├── ev_battery_supply_chain.txt
│   ├── quick_commerce_india.txt
│   ├── saas_metrics_guide.txt
│   └── solar_microgrids.txt
├── reports/                    # Output directory for audit reports
│   ├── summary_matrix.json
│   ├── summary_matrix.md
│   ├── sub_01_orig_report.json
│   ├── sub_01_orig_report.md
│   └── ...
├── sample_submissions/         # Synthetic benchmark dataset (15 files)
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
│   ├── sub_13_ai_gen.docx      # Synthetic AI-generated essay
│   ├── sub_14_corrupted.pdf    # 0-byte corrupted PDF edge case
│   └── sub_15_silent_video.mp4 # Video with no audio track edge case
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
    ├── text_processing.py      # Text cleaning and paragraph-aware chunking
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
- **PDF Extraction:** Extracts text by layout blocks using PyMuPDF (`pymupdf`), maintaining exact 1-indexed page boundaries. Empty or 0-byte PDFs are caught gracefully.
- **DOCX Extraction:** Reads paragraphs and table cells using `python-docx`. Approximates page numbers using a ~350 words-per-page model.
- **MP4 Audio/Video Extraction:** Extracts a mono 16kHz WAV track via `imageio-ffmpeg` directly into an in-memory float32 NumPy buffer, transcribing spoken words with `faster-whisper` (`tiny.en`). Silent videos and files without audio streams are caught and logged without aborting.
- **Chunking Strategy:** Documents are partitioned at paragraph boundaries into ~80–120 word windows with 20-word overlap only on oversized paragraphs. Spoken video segments are chunked into 20–30 second windows. This keeps shared competition prompts in distinct chunks that can be filtered cleanly.

### 2. Dense Semantic Vector Indexing
- Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) with L2 normalization.
- Built on `faiss.IndexFlatIP` (inner product on normalized vectors yields exact cosine similarity).
- Chunks are segregated into two distinct indexes:
  - `peer_index`: All valid chunks submitted by students in the current cohort.
  - `external_index`: Chunks derived from reference texts and background literature.

### 3. Scenario 4: Automated Boilerplate Template Filter
Case competitions and academic courses frequently provide standard prompt guidelines or submission templates that students paste into their work. Traditional similarity tools flag these headers as plagiarism.
- **Mechanism:** Before calculating peer similarity, the engine scans the peer index. Any chunk exhibiting cosine similarity $\ge 0.90$ across at least 4 distinct documents (or $\ge 35\%$ of the cohort) is flagged as boilerplate.
- **Action:** Boilerplate chunks are discounted from the total chunk count and excluded from peer match percentage calculations.

### 4. Scenario 1: Cross-Modal Plagiarism Detection
- Spoken transcripts from `.mp4` video presentations are encoded into the exact same 384-dimensional embedding space as written `.pdf` and `.docx` text.
- If a student reads a financial or strategic passage directly from a peer's slide deck or written report, the speech transcript chunk matches the written chunk (tested on `sub_10_pitch_src.pdf` vs `sub_11_pitch_video.mp4`, reaching **0.950** cosine similarity).

### 5. Scenario 2: Paraphrase Detection (Semantic vs. Lexical)
- Students replacing vocabulary with synonyms easily bypass token-matching and n-gram overlap algorithms (Turnitin-style lexical hashes).
- Bi-encoder vector representations map paraphrased sentences to adjacent coordinates based on contextual meaning rather than surface string equality.
- Evaluated on `sub_08_para_src.docx` vs `sub_09_para_dup.docx`:
  - **Lexical Jaccard Similarity:** `0.4697` (Bypasses lexical string matching)
  - **Dense Vector Cosine Similarity:** `0.8960` (Caught cleanly above 0.75 threshold)

### 6. Scenario 3: Corrupted and Empty Files
- `sub_14_corrupted.pdf` (0 bytes) is flagged as `ERROR` and assigned risk status `UNPROCESSED`.
- `sub_15_silent_video.mp4` (missing audio stream) is flagged as `WARNING` and assigned risk status `UNPROCESSED`.
- The pipeline processes the rest of the cohort without interruption.

### 7. AI Content Estimation
- Combines sentence-length variance (burstiness) and transition word density with a local sequence classifier (`Hello-SimpleAI/chatgpt-detector-roberta`).
- Flags highly uniform, low-perplexity, synthetic prose (`sub_13_ai_gen.docx` scored **95.0%** AI probability).

---

## Output Report Formats

The engine produces structured audit files in `--output_dir`:

### 1. Per-Submission Audit Trail (`<doc_id>_report.json`)
A machine-readable JSON containing full ingestion metadata, discounted boilerplate count, match lists with passage text and exact location citations, and AI estimation breakdown.

```json
{
  "document_id": "sub_11_pitch_video",
  "file_path": "./sample_submissions/sub_11_pitch_video.mp4",
  "modality": "mp4",
  "status": "ok",
  "overall_risk_level": "HIGH",
  "total_chunks": 1,
  "boilerplate_discounted_chunks": 0,
  "peer_similarity_percentage": 100.0,
  "external_similarity_percentage": 0.0,
  "ai_generated_probability": 0.05,
  "multimodal_matches": [
    {
      "query_chunk_id": "sub_11_pitch_video_chunk_0000",
      "query_passage": "Moving to the pro-former income statement, we project top-line revenues of 14.8 million...",
      "timestamp": "00:00:00 - 00:00:23",
      "matched_document_id": "sub_10_pitch_src",
      "matched_chunk_id": "sub_10_pitch_src_chunk_0001",
      "matched_passage": "Moving to the pro forma income statement, we project top-line revenues of $14.8 million...",
      "page_number": 2,
      "similarity_score": 0.9504,
      "match_type": "multimodal"
    }
  ]
}
```

### 2. Human-Readable Audit Report (`<doc_id>_report.md`)
Formatted for academic evaluators, featuring clean risk badges (`HIGH`, `MEDIUM`, `LOW`, `UNPROCESSED`), side-by-side passage comparison blocks, and citation markers (e.g., `Page 2` or `00:00:00 - 00:00:23`).

### 3. Cohort Summary Matrix (`summary_matrix.md` & `summary_matrix.json`)
An executive summary table indexing all submissions across the cohort:

| Document ID | Modality | Status | Risk Level | Peer Sim | Ext Sim | AI Prob | Discounted Boilerplate | Matches Flagged |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sub_01_orig` | `pdf` | `ok` | `LOW` | 0.0% | 0.0% | 42.8% | 1 | 0 peer, 0 ext |
| `sub_05_orig` | `pdf` | `ok` | `MEDIUM` | 0.0% | 50.0% | 36.0% | 1 | 0 peer, 1 ext |
| `sub_06_copy_src` | `pdf` | `ok` | `HIGH` | 100.0% | 0.0% | 5.0% | 0 | 2 peer, 0 ext |
| `sub_07_copy_dup` | `docx` | `ok` | `HIGH` | 100.0% | 0.0% | 5.0% | 0 | 2 peer, 0 ext |
| `sub_08_para_src` | `docx` | `ok` | `HIGH` | 100.0% | 0.0% | 5.0% | 0 | 2 peer, 0 ext |
| `sub_09_para_dup` | `docx` | `ok` | `HIGH` | 100.0% | 0.0% | 5.0% | 0 | 2 peer, 0 ext |
| `sub_10_pitch_src` | `pdf` | `ok` | `HIGH` | 50.0% | 0.0% | 5.0% | 0 | 1 multi, 0 ext |
| `sub_11_pitch_video` | `mp4` | `ok` | `HIGH` | 100.0% | 0.0% | 5.0% | 0 | 1 multi, 0 ext |
| `sub_13_ai_gen` | `docx` | `ok` | `HIGH` | 0.0% | 0.0% | 95.0% | 0 | 0 peer, 0 ext |
| `sub_14_corrupted` | `pdf` | `error` | `UNPROCESSED` | 0.0% | 0.0% | 0.0% | 0 | Corrupted file |
| `sub_15_silent_video` | `mp4` | `warning` | `UNPROCESSED` | 0.0% | 0.0% | 0.0% | 0 | Silent / No audio |

---

## Benchmark Evaluation Results

The test suite was evaluated against ground truth (`dataset_manifest.json`) using `python scripts/evaluate.py`:

```
================================================================================
          MULTIMODAL INTEGRITY ENGINE: BENCHMARK EVALUATION
================================================================================

1. Peer & Cross-Modal Plagiarism Detection:
   Precision: 1.0000 | Recall: 1.0000 | F1-Score: 1.0000
   Confusion Matrix: TP=6, FP=0, TN=7, FN=0

2. External Reference Match Detection:
   Precision: 1.0000 | Recall: 1.0000 | F1-Score: 1.0000
   Confusion Matrix: TP=1, FP=0, TN=12, FN=0

3. AI-Generated Content Detection:
   Precision: 1.0000 | Recall: 1.0000 | F1-Score: 1.0000
   Confusion Matrix: TP=1, FP=0, TN=12, FN=0

4. Overall Integrity Risk Flagging:
   Precision: 1.0000 | Recall: 1.0000 | F1-Score: 1.0000
   Confusion Matrix: TP=8, FP=0, TN=5, FN=0

Scenario 2 Proof (Paraphrased Pair: sub_08 vs sub_09):
  Token Jaccard Similarity : 0.4697 (low word overlap)
  Dense Cosine Similarity  : 0.8960 (high semantic similarity)
```

See [EVALUATION.md](file:///d:/integrity-engine/EVALUATION.md) for detailed failure mode breakdowns, transcription drift analysis, and model trade-off comparisons.

---

## Production Architecture Brief: Scaling to 10,000 Concurrent Submissions

Scaling this engine from a local cohort script to a production platform handling 10,000 concurrent student submissions requires rearchitecting ingestion, worker pools, vector storage, and caching.

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
- **Document Chunk Volume:**
  - 10,000 submissions $\times$ ~15 chunks per document = **150,000 total chunks**.
- **Vector Storage Size:**
  - Embedding dimension: 384 floats (FP32) = $384 \times 4 \text{ bytes} = 1,536 \text{ bytes per vector}$.
  - Total raw vector RAM: $150,000 \times 1,536 \text{ bytes} \approx \mathbf{230.4 \text{ MB}}$.
- **Key Insight:**
  - The vector index for 10,000 submissions easily fits into standard memory on a single node ($< 250 \text{ MB}$).
  - Vector search is **not** the operational bottleneck. The true system bottleneck is **audio/video transcription (ASR)**.
  - A 20-second video takes ~0.5–1.5 seconds to transcribe on a GPU. Ten thousand video submissions require ~10,000 compute-seconds. A pool of 8 NVIDIA T4 GPUs processes this backlog in ~15–20 minutes using dynamic batching.

### 3. Vector Database Selection & Partitioning Strategy
- **Index Architecture:** Migrate from in-memory `faiss.IndexFlatIP` to a production vector database (**Qdrant** or **Milvus**):
  - **Graph Indexing (HNSW):** Provides $O(\log N)$ query times with $< 15\text{ms}$ p99 search latency.
  - **Cohort Partitioning:** Vectors must be partitioned using a payload tag `competition_id` / `cohort_id`. Peer similarity searches must only search within the active competition cohort, avoiding cross-cohort contamination.
  - **Historical Archive:** For multi-year cross-cohort checks across millions of past submissions, use **IVF-PQ** (Inverted File with Product Quantization) to compress vector footprints by 75% while keeping query times sub-linear.

### 4. Cost & Latency Optimizations
1. **Deduplication & Hash Caching:** Compute SHA-256 hashes for all submitted files upon upload. Identical submissions bypass extraction and ASR entirely.
2. **MinHash / LSH Pre-Filter:** Before triggering dense bi-encoder searches across 150k chunks, run a lightweight MinHash/LSH pass on text chunks. Chunks with $> 95\%$ Jaccard overlap are immediately tagged as exact duplicates, bypassing dense vector search.
3. **Embedding Cache (Redis):** Cache normalized text chunk embeddings by SHA-256 hash. Common phrases and repeated paragraphs avoid redundant neural network forward passes.
4. **ASR Model Optimization:** Replace `tiny.en` with `distil-whisper/distil-large-v3` running under TensorRT-LLM or ONNX Runtime. This gives near-large-v3 accuracy at 6x inference speed.

---

## Assumptions & Engineering Constraints

1. **DOCX Physical Page Number Estimation:**
   - Microsoft Word `.docx` files are OOXML zip archives containing a dynamic flow document; they do not store static page numbers. Page counts are rendered dynamically by the viewing application.
   - We estimate page numbers using a standard academic density of **~350 words per page**.
2. **Reference Corpus Simulation:**
   - The system checks external similarity against a curated reference corpus in `./reference_corpus/`. In a real-world enterprise deployment, this would be backed by a web search crawler or academic publication database (e.g., CrossRef, arXiv, Semantic Scholar).
3. **AI Generation Scoring Limitations:**
   - Automated AI detection is based on stylistic burstiness and a local RoBERTa sequence classifier. Because formal, concise MBA-style case writing can exhibit low perplexity and formal sentence structures, AI probability scores should serve as advisory flags for human review, never as sole proof of academic misconduct.
4. **Boilerplate Detection Frequency Threshold:**
   - The boilerplate detection heuristic assumes cohort sizes $\ge 4$ submissions. Chunks with cosine similarity $\ge 0.90$ across at least 4 documents or $\ge 35\%$ of valid files are marked as template text. For smaller cohorts, an explicit template file should be provided via `--prompt_file`.

---

## AI Assistant Disclosure

Generative AI assistance was utilized during this project for drafting initial boilerplate scaffolding and creating realistic synthetic case-study texts. Core detection logic, FAISS vector indexing, ASR pipeline fixes, multi-signal scoring formulas, and empirical evaluation metrics were designed, implemented, and verified locally.
