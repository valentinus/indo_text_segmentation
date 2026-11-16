# Sample human-evaluation packet: Sundanese, Set C = CVS

**Real (from the Sundanese Wikipedia corpus, heading echo removed):**

- `packet.md`: 30 articles, 10 for each CVS error stratum (max(Pk, WD) <= 0.1, <= 0.2, <= 0.3).
  The summaries are still empty, so generate them with `--summarizer hf` or `--summarizer claude`.
- `annotation_sheet.csv`: blank sheet for one annotator. Fill in `annotator` and `rank`
  (1 = best, 4 = worst) for each article, criterion and summary label.
- `key.csv`: which label (S1-S4) is which set (A-D) for each article. Keep it away from annotators.
- `articles.json`: segments of every set, for generating summaries and for checking.

**SIMULATED, not results:** `SIMULATED_rankings.csv` and `SIMULATED_analysis/` come from
random rankings made by `human_eval.py simulate`. They only show the output format of
`human_eval.py analyze`. Never report these numbers.

For the paper, Set C should use the best segmenter (e.g. a sentence encoder run on GPU), not CVS:

    python scripts/human_eval.py prepare --results results/su/<method>/<tag>.pkl --lang su \
        --out human_eval/su --summarizer hf --model csebuetnlp/mT5_multilingual_XLSum
    python scripts/human_eval.py analyze --key human_eval/su/key.csv \
        --sheets "human_eval/su/filled_*.csv" --out human_eval/su/analysis
