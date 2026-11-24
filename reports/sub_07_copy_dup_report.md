# Integrity Audit Report: `sub_07_copy_dup.docx`

**Overall Status**: OK | **Risk Assessment**: **HIGH**

## 1. Quantitative Score Breakdown

| Evaluation Metric | Score | Assessment Tier | Benchmark Thresholds |
| :--- | :---: | :---: | :--- |
| **Peer Similarity** | **50.0%** | High | Low < 20.0% ≤ Med < 40.0% ≤ High |
| **Cross-Modal Similarity** | **0.0%** | Low | Low < 20.0% ≤ Med < 40.0% ≤ High (or match ≥ 0.85) |
| **External Reference Match** | **0.0%** | Low | Low < 20.0% ≤ Med < 40.0% ≤ High |
| **AI Generation Probability** | **20.0%** | Low | Low < 50.0% ≤ Med < 75.0% ≤ High |
| **Discounted Boilerplate Chunks** | **0** | Neutralized | Stripped shared tournament header |

## 2. Key Diagnostic Findings

- Detected 2 text-to-text peer plagiarism match(es) against sub_06_copy_src.pdf (peak cosine similarity: 1.000).

## 3. Peer-to-Peer Plagiarism Evidence

### Match #1: against `sub_06_copy_src.pdf` (Cosine Similarity: 1.000)
- **Query Location**: Page 1
- **Matched Location**: Page 1

**Query Passage:**
> "Advanced packaging technologies, notably Chip on Wafer on Substrate (CoWoS) and high density 2.5D interposers, have transitioned from niche high performance computing options into the primary operational bottleneck for modern generative accelerator manufacturing. Front end wafer lithography at sub-3-nanometer nodes yields negligible commercial utility if die interconnect density cannot overcome memory bandwidth wall constraints. Foundry operators are consequently diverting capital allocation from pure substrate fabrication toward localized advanced packaging cleanrooms and thermal dissipation testing facilities."

**Matched Passage:**
> "Advanced packaging technologies, notably Chip on Wafer on Substrate (CoWoS) and high density 2.5D interposers, have transitioned from niche high performance computing options into the primary operational bottleneck for modern generative accelerator manufacturing. Front end wafer lithography at sub-3-nanometer nodes yields negligible commercial utility if die interconnect density cannot overcome memory bandwidth wall constraints. Foundry operators are consequently diverting capital allocation from pure substrate fabrication toward localized advanced packaging cleanrooms and thermal dissipation testing facilities."

### Match #2: against `sub_06_copy_src.pdf` (Cosine Similarity: 1.000)
- **Query Location**: Page 1
- **Matched Location**: Page 1

**Query Passage:**
> "The extreme capital intensity of next generation extreme ultraviolet (EUV) lithography systems creates insurmountable barriers to entry for late stage foundry contenders. Amortizing multi billion dollar cleanroom tooling requires sustained utilization rates exceeding 90%, alongside multi year anchor tenant purchase commitments from hyper scaler clients. When cyclical semiconductor demand dips, foundries without diversified product lines across automotive and industrial microcontrollers face catastrophic margin erosion under fixed depreciation burdens."

**Matched Passage:**
> "The extreme capital intensity of next generation extreme ultraviolet (EUV) lithography systems creates insurmountable barriers to entry for late stage foundry contenders. Amortizing multi billion dollar cleanroom tooling requires sustained utilization rates exceeding 90%, alongside multi year anchor tenant purchase commitments from hyper scaler clients. When cyclical semiconductor demand dips, foundries without diversified product lines across automotive and industrial microcontrollers face catastrophic margin erosion under fixed depreciation burdens."

---
*Report automatically compiled by Multimodal Integrity & Plagiarism Detection Engine.*
