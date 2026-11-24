# System Evaluation & Benchmark Report

This document reports the empirical benchmark evaluation of the Multimodal Submission Integrity & Plagiarism Detection Engine across 15 synthetic case-study submissions and 4 reference corpus documents.

All metrics below are generated directly from the automated evaluation suite (`python scripts/evaluate.py`), comparing output reports against the ground-truth manifest (`sample_submissions/dataset_manifest.json`).

---

## 1. Benchmark Summary & Performance Metrics

Evaluation covers all 13 valid submissions (excluding the two deliberately corrupted/silent edge cases, which are evaluated separately).

> [!NOTE]
> **Engineering Disclaimer:** Achieving an F1-Score of 1.0000 across these 13 synthetic submissions reflects a tightly controlled end-to-end integration and sanity benchmark designed to validate algorithmic mechanics (bi-encoder semantic projection, cross-modal speech alignment, corpus-wide frequency filtering, and exception handling). It is **not** a claim of 100% real-world operational accuracy. In wild deployment, ambiguous student paraphrasing, optical character recognition artifacts, acoustic distortion, and nuanced disciplinary literature will inevitably introduce edge-case variance.

| Evaluation Task / Signal | Precision | Recall | F1-Score | Confusion Matrix (TP / FP / TN / FN) | Primary Ground-Truth Files |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Peer & Multimodal Plagiarism** | **1.0000** | **1.0000** | **1.0000** | TP=6, FP=0, TN=7, FN=0 | `sub_06`, `sub_07`, `sub_08`, `sub_09`, `sub_10`, `sub_11` |
| **External Reference Overlap** | **1.0000** | **1.0000** | **1.0000** | TP=1, FP=0, TN=12, FN=0 | `sub_05` (matches `ev_battery_supply_chain.txt`) |
| **AI Content Estimation (≥75%)** | **1.0000** | **1.0000** | **1.0000** | TP=1, FP=0, TN=12, FN=0 | `sub_13` (Synthetic LLM essay) |
| **Overall Integrity Risk Flagging** | **1.0000** | **1.0000** | **1.0000** | TP=8, FP=0, TN=5, FN=0 | Medium/High Risk vs Clean Original |

### Scenario 2 Proof: Lexical (Jaccard) vs. Semantic (Cosine) Similarity

To test whether vector embeddings catch paraphrased plagiarism that bypasses traditional n-gram matching, we evaluated the paraphrased pair (`sub_08_para_src.docx` vs `sub_09_para_dup.docx`):

- **Token-Level Jaccard Similarity**: `0.4697` (Low lexical overlap)
- **Dense Vector Cosine Similarity (`all-MiniLM-L6-v2`)**: `0.8960` (High semantic equivalence)

```
[sub_08 Query Passage]:
"However, shifting from an asset-light subscription revenue model to an embedded balance-sheet lending structure exposes enterprise software firms to severe macroeconomic interest rate volatility. Without sophisticated debt facility securitization and programmatic loan-loss reserve protocols, unexpected spikes in merchant delinquency can rapidly decimate operational cash reserves and undermine primary enterprise valuations."

[sub_09 Matched Passage]:
"However, transitioning from an asset-light subscription model to balance-sheet loan origination exposes enterprise software companies to significant macroeconomic interest rate fluctuations. In the absence of structured credit facility securitization and automated loan loss reserve protocols, unexpected surges in merchant defaults can quickly deplete cash reserves and erode primary company valuations."
```

Standard string-overlap tools fail here because words like "transitioning", "loan origination", "fluctuations", and "deplete" replace the original vocabulary. The bi-encoder projects both into adjacent coordinates in embedding space, returning a similarity of 0.896 and cleanly exceeding our 0.75 threshold.

### Scenario 3 Proof: Corrupted and Silent Files
- `sub_14_corrupted.pdf` (0-byte file): Handled by `pdf_extractor.py`, logged as an empty PDF, and categorized as `UNPROCESSED` without crashing the pipeline.
- `sub_15_silent_video.mp4` (5-second video with no audio track): Handled by `mp4_extractor.py`, detected missing audio stream, and categorized as `UNPROCESSED` with a clear warning.

### Scenario 4 Proof: Boilerplate Header Neutralization
- 5 separate submissions (`sub_01` to `sub_05`) carried the exact same 147-word competition rules header.
- The engine's corpus-frequency filter flagged the block across all 5 files, discounted 1 chunk from each, and prevented false-positive peer plagiarism flags among the original submissions (`sub_01` through `sub_04` remained at `0.0%` peer similarity, `LOW` risk).

---

## 2. Failure & Ambiguity Analysis

Here are three concrete failure modes and edge cases observed during development and testing:

### Case 1: ASR Transcription Drift on Domain Jargon
- **Observation:** In `sub_11_pitch_video.mp4`, `faster-whisper` (`tiny.en`) transcribed financial jargon with phonetic substitutions:
  - Text: *"pro forma"* → ASR: *"pro-former"*
  - Text: *"Net working capital"* → ASR: *"Networking capital"*
- **Why it still worked here:** Dense vector embeddings map phonetic mis-transcriptions relatively close in semantic space when surrounding context is strong. The chunk still scored `0.950` cosine similarity against `sub_10_pitch_src.pdf` (page 2).
- **Where this breaks:** On recordings with heavy regional accents, background room noise, or low-cost laptop microphones, `tiny.en` often drops technical terminology entirely or hallucinates text. If "Net working capital" became "working software", similarity would drop below 0.75.
- **Production Fix:**
  1. Upgrade from `tiny.en` to `distil-whisper/distil-large-v3` or `medium.en` on GPU workers.
  2. Use Whisper's `initial_prompt` parameter to bias decoding with common business and competition terms (e.g., `"EBITDA, CAC, LTV, pro forma, CapEx, BESS, SaaS"`).

### Case 2: Elevated AI-Probability Scores on Short, Structured Business Prose
- **Observation:** `sub_01_orig.pdf` scored `42.8%` and `sub_05_orig.pdf` scored `36.0%` on AI generation likelihood, despite being original human case studies.
- **Root Cause:** MBA-style case writing uses formal transitions ("Furthermore", "To mitigate", "In conclusion") and maintains disciplined, relatively uniform sentence lengths. Statistical burstiness heuristics and zero-shot RoBERTa classifiers both assign non-trivial AI probabilities to structured, formal prose.
- **Where this breaks:** If an institution sets an aggressive threshold (e.g., flagging anything over 40% as AI-generated), legitimate students who write concisely and formally will be falsely accused.
- **Production Fix:**
  1. Never flag a submission for disciplinary action based on an automated AI score alone.
  2. Enforce a minimum text length requirement (e.g., 300+ words of non-boilerplate text) before computing AI scores.
  3. Calibrate classifier decision thresholds against a verified baseline of domain-specific historical student papers rather than general-purpose web text.

### Case 3: Boilerplate Filter Sensitivity to Reworded Prompts
- **Observation:** The boilerplate filter requires a chunk to have cosine similarity ≥ 0.90 across at least 4 documents or ≥ 35% of the corpus.
- **Where this breaks:**
  1. If a cohort is small (e.g., only 3 students submit), or if fewer than 4 students include the prompt header, the automatic corpus frequency check will not fire.
  2. If a student rewords or abbreviates the prompt header, similarity across peers drops below 0.90, causing the prompt to be treated as original student text and inflating peer plagiarism scores.
- **Production Fix:** Add an optional CLI flag `--prompt_file ./templates/case_prompt.txt`. Chunks matching the organizer-provided prompt template at ≥ 0.85 similarity can be neutralized immediately without relying solely on cohort-wide statistical frequency.

---

## 3. Engineering Trade-off Analysis

### Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- **Why we chose it:**
  - **Memory & Size:** 22.7M parameters (~90 MB disk footprint, ~384-dimensional dense vectors). Runs fast on CPU without requiring an expensive GPU instance.
  - **Inference Speed:** Encodes ~1,500 words in under 300ms on a standard laptop CPU.
  - **Accuracy:** Sufficient semantic resolution for 80–120 word paragraphs, capturing paraphrasing (0.896 on `sub_08`/`sub_09`) and cross-modal alignment (0.950 on `sub_10`/`sub_11`).
- **Rejected Alternatives:**
  - `BAAI/bge-large-en-v1.5` or `text-embedding-3-large`: Larger models (1024-dim, >1.3 GB) provide slightly better MTEB retrieval scores, but increase embedding compute time by 6x and quadruple vector memory consumption without offering noticeable accuracy gains on paragraph-length case analyses.

### Vector Search: `faiss.IndexFlatIP` vs. Approximate Nearest Neighbors (ANN)
- **Why we chose it:**
  - Our test cohort generates ~50 chunks across 15 documents. Even a tournament with 500 submissions yields ~7,500 chunks (~11.5 MB in RAM).
  - On L2-normalized vectors, `IndexFlatIP` calculates exact cosine similarity with $O(N)$ dot products. At 7,500 chunks, an exhaustive search takes under 2 milliseconds on CPU.
  - Using exact search eliminates ANN recall degradation and hyperparameter tuning (e.g., `nlist`, `nprobe`, `efSearch`).
- **When to switch:**
  - Only when scaling past 100,000 chunks (10,000+ submissions or multi-year cross-cohort archives) does ANN (`IndexHNSWFlat` or `IndexIVFFlat`) become necessary to maintain sub-20ms search latencies.
