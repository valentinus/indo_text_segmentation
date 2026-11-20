# RDSP 2026 guest lecture: From Comments to Insight

A 90-minute lecture for the Ritsumeikan Data Science Program. It combines the 2025 lecture
(NLP foundations and the TopicDiff-LDA research) with a hands-on case: YouTube comments
about Fushimi Inari Taisha, Kyoto.

## Contents

| Path | What it is |
|---|---|
| `annotation_worksheet.xlsx` | Activity 1 worksheet: codebook, 20 comments, dropdowns for 3 annotators, live Fleiss' κ, hidden `4_AI_labels` sheet for the reveal |
| `prompts.md` | Activity 2 prompt sheet (zero-shot, codebook, disagreement, insight, self-check) |
| `data/comments.py` | The 60-comment dataset (EN/JA/ID) with the LLM labels and the 20-item annotation set |
| `data/fushimi_inari_comments.csv` | The comments without labels (what students and the LLM get) |
| `results/` | LLM labels, all-method labels, topic × sentiment table, `summary.json` |
| `scripts/run_experiments.py` | Seed-word baseline, LDA baseline, Cohen's κ against the LLM labels |
| `scripts/build_worksheet.py` | Rebuilds the worksheet from `data/comments.py` |
| `scripts/collect_youtube_comments.py` | Gets real comments through the YouTube Data API v3 (needs an API key) |
| `deck_source/` | Backup of the slide source (HTML per slide) |

## About the data

The 60 comments were **written for teaching**. They copy the style, topics and language mix
of real comments under travel videos, but they are not taken from YouTube. YouTube was not
reachable from the environment where the materials were prepared. The AI labels were made by
Claude using Prompt B. To use real data, run:

```bash
export YOUTUBE_API_KEY=...
python scripts/collect_youtube_comments.py <VIDEO_ID> --max 300 --out data/real_comments.csv
```

## Results of the reference experiment (n = 60)

| Method vs. LLM labels | Cohen's κ | Agreement |
|---|---|---|
| Seed words, English only (upper bound: the seed lists were written after reading the data) | 0.906 | 92% |
| Seed words, all languages | 0.749 | 78% |
| Seed words, Japanese and Indonesian only | — | 10% |
| LDA, 6 topics, best topic→label mapping | 0.158 | 30% |

## Reproduce

```bash
pip install pandas scikit-learn openpyxl
python scripts/run_experiments.py
python scripts/build_worksheet.py
```
