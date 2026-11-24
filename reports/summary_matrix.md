# Multimodal Submission Integrity Assessment — Summary Matrix

**Total Documents Evaluated**: 16

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

### Risk Categorization Policy
- **HIGH RISK**: Peer/Multimodal/External Similarity ≥ 40% OR Single Multimodal Match ≥ 0.85 OR AI Generation Likelihood ≥ 75%
- **MEDIUM RISK**: Peer/Multimodal/External Similarity ≥ 20% OR AI Generation Likelihood ≥ 50%
- **LOW RISK**: Below medium threshold guidelines
- **UNPROCESSED**: Corrupted or empty submission files
