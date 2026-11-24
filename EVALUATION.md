# System Evaluation & Benchmark Report

This document reports the empirical benchmark evaluation of the Multimodal Submission Integrity & Plagiarism Detection Engine across 16 synthetic case-study submissions (14 valid submissions plus 2 corrupted/silent edge cases) and 4 reference corpus documents.

All metrics below are generated directly from the automated evaluation suite (`python scripts/evaluate.py`), comparing output reports against the ground-truth manifest (`sample_submissions/dataset_manifest.json`).

---

## 1. Benchmark Summary & Performance Metrics

Evaluation covers all 14 valid submissions (the two deliberately corrupted/silent edge cases are evaluated separately under Scenario 3).

> [!NOTE]
> **Engineering Disclaimer:** The 1.0000 precision and recall scores on peer, chunk-level, and external plagiarism across these synthetic submissions reflect a controlled sanity benchmark designed to validate algorithmic mechanics (bi-encoder semantic projection, cross-modal alignment, boilerplate filtering, and exception handling). They are **not** a claim of 100% real-world operational accuracy. As shown honestly by our AI-detection results below, modern LLM prose and noisy real-world student documents create substantial variance outside controlled test cases.

| Signal / Task | Precision | Recall | F1-Score | Counts / Confusion Matrix |
| :--- | :---: | :---: | :---: | :--- |
| **Peer & Multimodal Plagiarism (Doc-Level)** | **1.0000** | **1.0000** | **1.0000** | TP=6, FP=0, TN=8, FN=0 |
| **Peer Text Plagiarism (Chunk-Level)** | **1.0000** | **1.0000** | **1.0000** | TP=8, FP=0, FN=0 |
| **External Reference Match (Doc-Level)** | **1.0000** | **1.0000** | **1.0000** | TP=1, FP=0, TN=13, FN=0 |
| **AI Content Estimation (>=75% HIGH)** | **1.0000** | **0.5000** | **0.6667** | TP=1, FP=0, TN=12, FN=1 |
| **Overall Integrity Flagging (Doc-Level)** | **1.0000** | **0.8889** | **0.9412** | TP=8, FP=0, TN=5, FN=1 |

### AI Content Estimation Breakdown (`max(RoBERTa, Heuristic)`)

Using `final_score = max(hf_score, heuristic_score)` with explicit `top_k=None` class-label parsing on `Hello-SimpleAI/chatgpt-detector-roberta`:

| Synthetic AI Submission | Style Profile | AI Prob (%) | Risk Level | Honest Observation |
| :--- | :--- | :---: | :---: | :--- |
| `sub_13_ai_gen.docx` | Formulaic clichés & uniform syntax | **`95.0%`** | **`HIGH`** | Heuristic scored `95.0%` and is no longer dragged down by legacy 2022 RoBERTa (`~0.1%`). |
| `sub_16_ai_natural.docx` | ~350-word natural marker-free LLM prose | **`20.0%`** | **`LOW`** | Without cliché transition markers, both the heuristic (`20.0%`) and 2022 HC3-trained RoBERTa (`~0.04%`) rate this clean LLM prose as human. |

Original human submissions remain safely below the `50%` MEDIUM threshold (`sub_01_orig.pdf = 41.8%`, `sub_05_orig.pdf = 35.5%`, others `10.0%–25.1%`), yielding zero false positives (`FP=0`). This demonstrates an honest operational reality: legacy zero-shot RoBERTa checkpoints trained on 2022 conversational Q&A generalize poorly to clean, modern business prose without domain-calibrated retraining.

---

## 2. Scenario 2 Proof: Lexical (Levenshtein & Jaccard) vs. Semantic Similarity

To quantify why traditional lexical plagiarism checkers miss paraphrased submissions, we evaluated the paraphrased pair (`sub_08_para_src.docx` vs `sub_09_para_dup.docx`):

| Similarity Metric | Score | Detection Outcome |
| :--- | :---: | :--- |
| **Levenshtein Similarity Ratio** | `0.6455` | Fails (`< 0.75`, low character-sequence overlap) |
| **Token-Level Jaccard Similarity** | `0.4697` | Fails (`< 0.75`, low lexical word overlap) |
| **Dense Vector Cosine (`all-MiniLM-L6-v2`)** | **`0.8990`** | **Passes cleanly (`>= 0.75`)** |

```
[sub_08 Query Passage]:
"However, shifting from an asset light subscription revenue model to an embedded balance sheet lending structure exposes enterprise software firms to severe macroeconomic interest rate volatility. Without sophisticated debt facility securitization and programmatic loan loss reserve protocols, unexpected spikes in merchant delinquency can rapidly decimate operational cash reserves and undermine primary enterprise valuations."

[sub_09 Matched Passage]:
"However, transitioning from an asset light subscription model to balance sheet loan origination exposes enterprise software companies to significant macroeconomic interest rate fluctuations. In the absence of structured credit facility securitization and automated loan loss reserve protocols, unexpected surges in merchant defaults can quickly deplete cash reserves and erode primary company valuations."
```

---

## 3. True-Negative Pairwise Similarity Distribution (Threshold Margin Proof)

To verify that our `0.75` text-to-text threshold and `0.70` multimodal threshold maintain safe headroom against false positives, we measured the maximum chunk-level cosine similarity across all **88 unrelated (true-negative) document pairs** in the valid cohort (excluding shared boilerplate chunks and the 3 ground-truth positive pairs):

| Pair Category | Pair Count | Min Max-Sim | Mean Max-Sim | Max Max-Sim | Margin to Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **All Unrelated TN Pairs** | `88` | `0.1328` | `0.3332` | `0.5920` | — |
| **Text-to-Text TN Pairs** (vs `0.75`) | `64` | `0.1328` | `0.3512` | `0.5920` | **`+0.1580`** |
| **Multimodal MP4 TN Pairs** (vs `0.70`) | `24` | `0.1429` | `0.2854` | `0.5729` | **`+0.1271`** |

- **Highest Unrelated Pair Observed:** `sub_01_orig.pdf` vs `sub_08_para_src.docx` (`0.5920` — both discuss B2B SaaS business models, yet remain `0.158` below our `0.75` text plagiarism threshold).

---

## 4. Scenario 1, 3 & 4 Verification

- **Scenario 1 (Cross-Modal Audio/Text Alignment):** Upgrading from `tiny.en` to `base.en` with a financial `initial_prompt` and spoken/written text normalization raised cosine similarity between `sub_10_pitch_src.pdf` (page 2) and `sub_11_pitch_video.mp4` (`00:00:00-00:00:26`) from `0.950` to **`0.971`**. Both submissions are flagged `HIGH` risk via our `>= 0.85` multimodal match gate.
- **Scenario 3 (Corrupted & Silent Files):**
  - `sub_14_corrupted.pdf` (0-byte file): Caught by `pdf_extractor.py`, logged as `ERROR`, categorized as `UNPROCESSED`.
  - `sub_15_silent_video.mp4` (5-second video with no audio stream): Caught by `mp4_extractor.py`, logged as `WARNING`, categorized as `UNPROCESSED`.
- **Scenario 4 (Boilerplate Neutralization):**
  - `sub_01_orig` through `sub_05_orig` each carried the shared 147-word competition charter. All 5 had `1` boilerplate chunk discounted (`Peer Sim = 0.0%`, `0` false-positive peer flags).

---

## 5. Failure & Ambiguity Analysis

### Case 1: Remaining ASR Phonetic Drift on Compound Finance Terms
- **Observation:** Upgrading to `faster-whisper` (`base.en`) with domain `initial_prompt` and `normalize_spoken_written()` fixed `"pro forma"`, `"break even"`, `"month 18"`, and `"64%"`, lifting `sub_11_pitch_video.mp4` similarity to **`0.971`**. However, `base.en` still transcribed `"Net working capital"` as `"Networking capital"`.
- **Where this breaks:** On noisy student recordings, heavy accents, or dense domain acronyms (e.g., `"EBITDA"`, `"LTV/CAC"`, `"BESS"`), small CPU Whisper models still introduce compound-word errors.
- **Production Fix:** Run `distil-whisper/distil-large-v3` on GPU workers alongside cohort-specific glossary biasing via `initial_prompt`.

### Case 2: Out-of-Distribution Failure on Modern Marker-Free LLM Prose
- **Observation:** `Hello-SimpleAI/chatgpt-detector-roberta` was trained primarily on 2022–2023 conversational HC3 ChatGPT Q&A text. On clean business strategy prose, it predicted `'Human'` at `~99.9%` on both `sub_13_ai_gen.docx` (`ChatGPT=0.13%`) and `sub_16_ai_natural.docx` (`ChatGPT=0.04%`). Using `max(hf_score, heuristic_score)` lets our heuristic catch formulaic AI (`sub_13_ai_gen.docx = 95.0%`), while marker-free `sub_16_ai_natural.docx` still scores `20.0%` (`LOW`).
- **Root Cause:** Zero-shot RoBERTa checkpoints experience severe domain and generation shift across model families and cannot reliably separate disciplined human consulting prose from clean modern LLM output.
- **Production Fix:** Fine-tune an institution-calibrated classifier on recent domain submissions paired with frontier model generations, and treat AI estimation strictly as a non-punitive advisory triage metric.

### Case 3: Boilerplate Detection in Small Cohorts or Paraphrased Prompts
- **Observation:** Our fixed boilerplate filter excludes self-matches and requires matching at least `min_doc_count = max(2, min(4, ceil(N * 0.35)))` other documents, with a safety guard skipping any document if 100% of its chunks would be discounted.
- **Where this breaks:** In tiny cohorts (`N <= 2`) boilerplate cannot be distinguished statistically from plagiarism; in larger cohorts, if a student slightly paraphrases the shared prompt header below `0.90` cosine similarity, it bypasses the filter.
- **Production Fix:** Add an optional `--prompt_file` CLI flag allowing organizers to supply known template headers directly.

---

## 6. Engineering Trade-off Analysis

### Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- **Why we chose it:** 22.7M parameters (~90 MB footprint, 384-dim dense vectors), encodes ~1,500 words in under 300ms on CPU, and separates paraphrased plagiarism (`0.8990`) cleanly from unrelated business prose (`max 0.5920`).
- **Paragraph Chunk Size (~80–120 words):** `all-MiniLM-L6-v2` truncates inputs past **256 word-piece tokens** (~180–200 English words). Using 200–400 token chunks would silently drop trailing sentences and merge ~150-word prompt headers with student analysis; ~80–120 word paragraph-aware chunks prevent truncation loss and keep boilerplate isolated.

### Vector Search: `faiss.IndexFlatIP` vs. Approximate Nearest Neighbors (ANN)
- **Why we chose it:** At cohort scale (50 chunks now, or ~7,500 chunks for 500 submissions = ~11.5 MB RAM), exhaustive dot product (`IndexFlatIP` on L2-normalized vectors) runs in under 2ms on CPU with exact cosine recall and zero ANN tuning (`nlist`, `nprobe`, `efSearch`).
- **When to switch:** Above ~100,000 chunks (10,000+ submissions or multi-year archives), move to HNSW (`Qdrant` / `Milvus`) or `IVF-PQ`.
