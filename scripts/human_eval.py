#!/usr/bin/env python3
"""Human evaluation of segmentation-aware summarization (Sets A-D).

  A  unsegmented article (summarized whole, truncated to the model limit)
  B  fixed-size chunks of N sentences (default 8)
  C  automatic segmentation (predictions saved by main.py for one method)
  D  gold segmentation (section headings)

Sub-commands
------------
prepare   pick articles stratified by the C method's Pk/WD (<0.1, <0.2, <0.3, as in
          the conference study), build the four segmentations, optionally
          summarize them, and write a BLINDED annotation packet + answer key.
analyze   read filled annotation sheets and report average ranks, Friedman test,
          Nemenyi critical difference, Kendall's W, and a CD diagram.
simulate  write a filled sheet with SIMULATED random rankings, only to test
          `analyze` and see the output format. Never report simulated numbers.

Examples
--------
python scripts/human_eval.py prepare --results results/jv/cvs/stripped.pkl --lang jv --out human_eval/jv
python scripts/human_eval.py prepare ... --summarizer hf --model csebuetnlp/mT5_multilingual_XLSum
python scripts/human_eval.py prepare ... --summarizer claude --model claude-opus-5-5
python scripts/human_eval.py analyze --key human_eval/jv/key.csv --sheets human_eval/jv/filled_*.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pickle
import random
import sys
from collections import defaultdict
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SETS = ['A', 'B', 'C', 'D']
SET_NAMES = {'A': 'unsegmented', 'B': 'fixed-size segmentation',
             'C': 'automatic segmentation', 'D': 'gold-standard segmentation'}
CRITERIA = ['informativeness', 'coherence', 'conciseness']
# Studentized range q_0.05 / sqrt(2) for the Nemenyi test (Demsar 2006), by number of systems k.
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}


# ----------------------------------------------------------------- prepare
def split(sentences: List[str], bounds: List[int]) -> List[str]:
    out, last = [], 0
    for b in sorted(bounds):
        out.append(" ".join(sentences[last:b]))
        last = b
    out.append(" ".join(sentences[last:]))
    return [s for s in out if s.strip()]


def fixed_bounds(n: int, size: int) -> List[int]:
    return list(range(size, n, size))


def pick_articles(results, n: int, bins: List[float], seed: int):
    """n articles split evenly over Pk/WD bins (both metrics below the bin's bound)."""
    rng = random.Random(seed)
    per_bin = n // len(bins)
    chosen, lo = [], -1.0
    for hi in bins:
        pool = [i for i, r in enumerate(results)
                if lo < max(r.pk_scores, r.windiff_scores) <= hi and len(r.sentences) <= 120]
        if len(pool) < per_bin:
            print(f'[warn] only {len(pool)} articles with Pk,WD in ({lo}, {hi}]')
        chosen += [(i, f'<={hi}') for i in rng.sample(pool, min(per_bin, len(pool)))]
        lo = hi
    return chosen


class Summarizer:
    def __init__(self, kind: str, model: str, language: str):
        self.kind, self.model, self.language = kind, model, language
        self._pipe = self._client = None

    def __call__(self, text: str) -> str:
        if self.kind == 'none':
            return ''
        if self.kind == 'hf':
            if self._pipe is None:
                from transformers import pipeline
                self._pipe = pipeline('summarization', model=self.model)
            return self._pipe(text, truncation=True, max_length=84, no_repeat_ngram_size=2)[0]['summary_text']
        if self.kind == 'claude':
            import anthropic
            if self._client is None:
                self._client = anthropic.Anthropic()
            resp = self._client.messages.create(
                model=self.model, max_tokens=16000, output_config={"effort": "low"},
                messages=[{"role": "user", "content":
                           f"Summarize the following {self.language} text in {self.language}, "
                           f"in one or two sentences. Reply with the summary only.\n\n{text}"}])
            if resp.stop_reason == 'refusal':
                return '[refused]'
            return next((b.text for b in resp.content if b.type == 'text'), '').strip()
        raise ValueError(self.kind)


def prepare(args) -> None:
    from languages import get_language

    with open(args.results, 'rb') as f:
        results = pickle.load(f)
    bins = [float(x) for x in args.bins.split(',')]
    chosen = pick_articles(results, args.n, bins, args.seed)
    summarize = Summarizer(args.summarizer, args.model, get_language(args.lang).name)
    rng = random.Random(args.seed + 7)
    os.makedirs(args.out, exist_ok=True)

    key_rows, sheet_rows, packet = [], [], []
    articles = []
    for a_idx, (i, stratum) in enumerate(chosen, 1):
        r = results[i]
        segs = {
            'A': [" ".join(r.sentences)],
            'B': split(r.sentences, fixed_bounds(len(r.sentences), args.fixed)),
            'C': split(r.sentences, r.pred_boundaries),
            'D': split(r.sentences, r.true_boundaries),
        }
        summaries = {s: "\n".join(x for x in (summarize(t) for t in segs[s]) if x) for s in SETS}
        labels = SETS[:]
        rng.shuffle(labels)  # blind: S1..S4 map to a random set per article
        art_id = f'{args.lang}-{a_idx:03d}'
        articles.append({'article_id': art_id, 'stratum': stratum, 'result_index': i,
                         'pk_C': r.pk_scores, 'wd_C': r.windiff_scores, 'segments': segs,
                         'summaries': summaries})
        packet.append(f"## Article {art_id}\n\n**Source text**\n\n" + "\n\n".join(segs['D']) + "\n")
        for j, s in enumerate(labels, 1):
            label = f'S{j}'
            key_rows.append({'article_id': art_id, 'label': label, 'set': s, 'stratum': stratum})
            packet.append(f"**Summary {label}**\n\n{summaries[s] or '[summary to be generated]'}\n")
            for c in CRITERIA:
                sheet_rows.append({'annotator': '', 'article_id': art_id, 'criterion': c,
                                   'label': label, 'rank': ''})

    with open(os.path.join(args.out, 'key.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['article_id', 'label', 'set', 'stratum'])
        w.writeheader(); w.writerows(key_rows)
    with open(os.path.join(args.out, 'annotation_sheet.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['annotator', 'article_id', 'criterion', 'label', 'rank'])
        w.writeheader(); w.writerows(sheet_rows)
    with open(os.path.join(args.out, 'packet.md'), 'w', encoding='utf-8') as f:
        f.write(f"# Summary ranking task ({get_language(args.lang).name})\n\n"
                "For each article, rank the four summaries from 1 (best) to 4 (worst) on "
                "informativeness, coherence and conciseness. Use each rank once per criterion.\n\n")
        f.write("\n".join(packet))
    with open(os.path.join(args.out, 'articles.json'), 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=1)
    print(f'Wrote {len(chosen)} articles to {args.out}: packet.md, annotation_sheet.csv, key.csv '
          f'(keep key.csv away from annotators), articles.json')


# ----------------------------------------------------------------- analyze
def _rank(values: List[float]) -> List[float]:
    """Average ranks (1 = smallest) with ties shared."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def kendall_w(rankings: List[List[float]]) -> float:
    """Kendall's W for m raters ranking k items (no tie correction)."""
    m, k = len(rankings), len(rankings[0])
    if m < 2:
        return float('nan')
    sums = [sum(r[i] for r in rankings) for i in range(k)]
    mean = sum(sums) / k
    s = sum((x - mean) ** 2 for x in sums)
    return 12 * s / (m ** 2 * (k ** 3 - k))


def analyze(args) -> None:
    from scipy.stats import friedmanchisquare

    key = {}
    with open(args.key, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key[(row['article_id'], row['label'])] = row['set']
    ranks = defaultdict(dict)  # (article, criterion, annotator) -> {set: rank}
    simulated = False
    for path in sorted(p for pat in args.sheets for p in glob.glob(pat)):
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if not row.get('rank'):
                    continue
                simulated |= row.get('simulated') == '1' or 'SIMULATED' in os.path.basename(path)
                s = key[(row['article_id'], row['label'])]
                ranks[(row['article_id'], row['criterion'], row['annotator'])][s] = float(row['rank'])
    if simulated:
        print('#' * 72 + '\n# SIMULATED RANKINGS - format demo only, not results. Do not report.\n' + '#' * 72)

    articles = sorted({a for a, _, _ in ranks})
    annotators = sorted({n for _, _, n in ranks})
    k = len(SETS)
    report = {}
    for crit in CRITERIA + ['overall']:
        crits = CRITERIA if crit == 'overall' else [crit]
        blocks, ws = [], []
        for a in articles:
            # mean rank of each set over annotators (and criteria for 'overall'), re-ranked per article
            vals = []
            for s in SETS:
                xs = [ranks[(a, c, n)][s] for c in crits for n in annotators if s in ranks.get((a, c, n), {})]
                vals.append(sum(xs) / len(xs))
            blocks.append(_rank(vals))
            for c in crits:
                rk = [[ranks[(a, c, n)][s] for s in SETS] for n in annotators if len(ranks.get((a, c, n), {})) == k]
                if len(rk) >= 2:
                    ws.append(kendall_w(rk))
        N = len(blocks)
        avg = {s: sum(b[i] for b in blocks) / N for i, s in enumerate(SETS)}
        stat, p = friedmanchisquare(*[[b[i] for b in blocks] for i in range(k)])
        cd = Q05[k] * math.sqrt(k * (k + 1) / (6 * N))
        sig = {s: [t for t in SETS if t != s and abs(avg[s] - avg[t]) > cd] for s in SETS}
        report[crit] = {'N_articles': N, 'annotators': len(annotators), 'avg_rank': avg,
                        'friedman_chi2': stat, 'friedman_p': p, 'nemenyi_CD': cd,
                        'significantly_different_from': sig,
                        'kendall_W_mean': sum(ws) / len(ws) if ws else float('nan')}

    for crit, r in report.items():
        print(f"\n== {crit}  (N = {r['N_articles']} articles, {r['annotators']} annotators) ==")
        print(f"Friedman chi2 = {r['friedman_chi2']:.2f}, p = {r['friedman_p']:.2g}; "
              f"Nemenyi CD = {r['nemenyi_CD']:.2f}; Kendall's W (mean over articles) = {r['kendall_W_mean']:.3f}")
        print('| Set | Average rank | Significantly different from (alpha = 0.05) |')
        print('| --- | --- | --- |')
        for s in sorted(SETS, key=lambda s: r['avg_rank'][s]):
            other = ', '.join(f'Set {t}' for t in r['significantly_different_from'][s]) or 'none'
            print(f"| Set {s} ({SET_NAMES[s]}) | {r['avg_rank'][s]:.2f} | {other} |")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, 'human_eval_report.json'), 'w') as f:
            json.dump({'simulated': simulated, **report}, f, indent=2)
        cd_diagram(report['overall'], os.path.join(args.out, 'cd_diagram.png'), simulated)
        print(f"\nWrote {args.out}/human_eval_report.json and cd_diagram.png")


def cd_diagram(r: Dict, path: str, simulated: bool) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    avg, cd = r['avg_rank'], r['nemenyi_CD']
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.set_xlim(1, 4); ax.set_ylim(0, 1); ax.invert_xaxis()
    ax.get_yaxis().set_visible(False)
    for sp in ('left', 'right', 'top'):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel('Average rank (lower is better)')
    order = sorted(SETS, key=lambda s: avg[s])
    for j, s in enumerate(order):
        y = 0.75 - 0.18 * j
        ax.plot([avg[s], avg[s]], [0, y], color='0.3', lw=1)
        # alternate label sides so close ranks do not overlap
        ax.text(avg[s], y + 0.03, f' Set {s} ({avg[s]:.2f}) ', ha='left' if j % 2 == 0 else 'right', fontsize=9)
    # bars joining sets that are NOT significantly different
    for j, s in enumerate(order):
        grp = [t for t in order[j:] if abs(avg[t] - avg[s]) <= cd]
        if len(grp) > 1:
            ax.plot([avg[grp[0]], avg[grp[-1]]], [0.08 + 0.05 * j] * 2, color='tab:blue', lw=3)
    ax.plot([1.05, 1.05 + cd], [0.95, 0.95], color='k', lw=1.5)
    ax.text(1.05 + cd / 2, 0.97, f'CD = {cd:.2f}', ha='center', fontsize=8, va='bottom')
    if simulated:
        ax.text(2.5, 0.5, 'SIMULATED - NOT RESULTS', ha='center', va='center', fontsize=16,
                color='red', alpha=0.35, rotation=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


# ---------------------------------------------------------------- simulate
def simulate(args) -> None:
    """Random rankings with a chosen tendency, ONLY to exercise `analyze`."""
    rng = random.Random(args.seed)
    tendency = {s: float(x) for s, x in zip(SETS, args.tendency.split(','))}
    key = defaultdict(dict)
    with open(args.key, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key[row['article_id']][row['label']] = row['set']
    rows = []
    for n in range(1, args.annotators + 1):
        for art, labels in key.items():
            for c in CRITERIA:
                score = {lab: tendency[s] + rng.gauss(0, args.noise) for lab, s in labels.items()}
                for rank, lab in enumerate(sorted(score, key=score.get), 1):
                    rows.append({'annotator': f'sim{n}', 'article_id': art, 'criterion': c,
                                 'label': lab, 'rank': rank, 'simulated': 1})
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['annotator', 'article_id', 'criterion', 'label', 'rank', 'simulated'])
        w.writeheader(); w.writerows(rows)
    print(f'Wrote SIMULATED sheet {args.out} ({len(rows)} rows). For testing analyze only.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--results', required=True, help='Pickle saved by main.py for the Set C method')
    a.add_argument('--lang', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--n', type=int, default=30)
    a.add_argument('--bins', default='0.1,0.2,0.3', help='Upper bounds on max(Pk, WD) per stratum')
    a.add_argument('--fixed', type=int, default=8, help='Set B chunk size (sentences)')
    a.add_argument('--summarizer', default='none', choices=['none', 'hf', 'claude'])
    a.add_argument('--model', default='csebuetnlp/mT5_multilingual_XLSum')
    a.add_argument('--seed', type=int, default=42)
    b = sub.add_parser('analyze')
    b.add_argument('--key', required=True)
    b.add_argument('--sheets', nargs='+', required=True, help='Filled annotation CSVs (globs allowed)')
    b.add_argument('--out', default=None)
    c = sub.add_parser('simulate')
    c.add_argument('--key', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--annotators', type=int, default=3)
    c.add_argument('--tendency', default='3.0,2.5,2.0,2.0', help='Mean badness per set A,B,C,D')
    c.add_argument('--noise', type=float, default=0.9)
    c.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    {'prepare': prepare, 'analyze': analyze, 'simulate': simulate}[args.cmd](args)


if __name__ == '__main__':
    main()
