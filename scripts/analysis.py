#!/usr/bin/env python3
"""Statistical and error analysis of saved segmentation results.

Reads the per-document pickles written by main.py
(<root>/<lang>/<method>/<tag>.pkl, a list of SegmentResult) and produces:

  significance   per language: mean Pk/WD with 95% bootstrap CIs, and paired
                 Wilcoxon signed-rank tests between all methods (Holm-corrected)
  echo           paired effect of removing the heading echo, on the documents
                 present in both settings (matched by their text)
  errors         error analysis: boundary precision/recall (exact and +-1),
                 near misses, Pk by segment length / document length / segment
                 count, and Spearman correlations
  example        one worked example (gold vs predicted boundaries)

Example
-------
python scripts/analysis.py --roots runs/idruns/results runs/regruns/results \
    --random-root runs/random --langs id jv su --out analysis_out
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils import SegmentResult  # noqa: E402,F401  (needed to unpickle)

METHOD_NAMES = {
    'llm_mock_k_0shot': 'Random (k given)', 'texttiling': 'TextTiling', 'c99': 'C99', 'cvs': 'CVS',
    'topictiling': 'TopicTiling', 'bayesseg': 'BayesSeg', 'bayesseg_md': 'BayesSeg-MD',
    'beamseg': 'BeamSeg', 'bertopic': 'BERTopic', 'bert': 'Sentence encoder',
}


def name(m: str) -> str:
    return METHOD_NAMES.get(m, m)


# ------------------------------------------------------------------ loading
def load_runs(roots: List[str], random_root: str, lang: str, tag: str) -> Dict[str, list]:
    runs = {}
    for root in roots:
        for path in glob.glob(os.path.join(root, lang, '*', f'{tag}.pkl')):
            runs[os.path.basename(os.path.dirname(path))] = pickle.load(open(path, 'rb'))
    if random_root and tag == 'stripped':
        for path in glob.glob(os.path.join(random_root, lang, 'llm_mock_k_0shot', '*.pkl')):
            runs['llm_mock_k_0shot'] = pickle.load(open(path, 'rb'))
    return runs


def check_aligned(runs: Dict[str, list]) -> int:
    sigs = {}
    for m, rs in runs.items():
        h = hashlib.md5('|'.join(r.sentences[0] + r.sentences[-1] + str(len(r.sentences)) for r in rs).encode())
        sigs[m] = (len(rs), h.hexdigest())
    if len(set(sigs.values())) != 1:
        raise SystemExit(f'Runs are not aligned on the same documents: {sigs}')
    return next(iter(sigs.values()))[0]


# ------------------------------------------------------------- statistics
def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals: Dict, alpha: float = 0.05) -> Dict:
    order = sorted(pvals, key=pvals.get)
    m, adj, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def significance(runs: Dict[str, list], metric: str) -> dict:
    from scipy.stats import wilcoxon

    attr = 'pk_scores' if metric == 'pk' else 'windiff_scores'
    X = {m: np.array([getattr(r, attr) for r in rs]) for m, rs in runs.items()}
    summary = {}
    for m, x in X.items():
        lo, hi = bootstrap_ci(x)
        summary[m] = {'mean': float(x.mean()), 'ci95': [lo, hi], 'median': float(np.median(x))}
    raw, diffs = {}, {}
    ms = sorted(X, key=lambda m: X[m].mean())
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            d = X[a] - X[b]
            diffs[(a, b)] = float(d.mean())
            raw[(a, b)] = float(wilcoxon(X[a], X[b], zero_method='zsplit').pvalue) if np.any(d != 0) else 1.0
    adj = holm(raw)
    pairs = [{'a': a, 'b': b, 'mean_diff': diffs[(a, b)], 'p_raw': raw[(a, b)], 'p_holm': adj[(a, b)]}
             for (a, b) in raw]
    return {'summary': summary, 'order': ms, 'pairs': pairs}


# -------------------------------------------------------------- echo pair
_SPACE = re.compile(r'\s+')


def _norm(text: str) -> str:
    return _SPACE.sub('', text)


def echo_key(r) -> str:
    """Text of a cue-kept document with the first sentence of every segment (the echo) removed."""
    starts = set([0] + list(r.true_boundaries))
    return _norm(''.join(s for i, s in enumerate(r.sentences) if i not in starts))


def stripped_key(r) -> str:
    return _norm(''.join(r.sentences))


def echo_pairs(echo_runs: Dict[str, list], stripped_runs: Dict[str, list]) -> dict:
    from scipy.stats import wilcoxon

    out = {}
    for m in sorted(set(echo_runs) & set(stripped_runs)):
        e_idx = {}
        for i, r in enumerate(echo_runs[m]):
            e_idx.setdefault(echo_key(r), i)
        pairs = [(e_idx[k], j) for j, r in enumerate(stripped_runs[m]) if (k := stripped_key(r)) in e_idx]
        if not pairs:
            continue
        e = np.array([echo_runs[m][i].pk_scores for i, _ in pairs])
        s = np.array([stripped_runs[m][j].pk_scores for _, j in pairs])
        ew = np.array([echo_runs[m][i].windiff_scores for i, _ in pairs])
        sw = np.array([stripped_runs[m][j].windiff_scores for _, j in pairs])
        d = s - e
        lo, hi = bootstrap_ci(d)
        out[m] = {
            'n_matched': len(pairs), 'n_echo': len(echo_runs[m]), 'n_stripped': len(stripped_runs[m]),
            'pk_echo': float(e.mean()), 'pk_stripped': float(s.mean()), 'delta_pk': float(d.mean()),
            'delta_pk_ci95': [lo, hi], 'wd_echo': float(ew.mean()), 'wd_stripped': float(sw.mean()),
            'p_wilcoxon': float(wilcoxon(s, e, zero_method='zsplit').pvalue) if np.any(d != 0) else 1.0,
            'share_worse': float((d > 0).mean()), 'share_better': float((d < 0).mean()),
        }
    return out


# ----------------------------------------------------------------- errors
def boundary_prf(results, tol: int):
    tp = fp = fn = 0
    for r in results:
        gold, pred = set(r.true_boundaries), set(r.pred_boundaries)
        matched_g = set()
        for p in sorted(pred):
            hit = next((g for g in range(p - tol, p + tol + 1) if g in gold and g not in matched_g), None)
            if hit is None:
                fp += 1
            else:
                tp += 1
                matched_g.add(hit)
        fn += len(gold) - len(matched_g)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def error_analysis(runs: Dict[str, list]) -> dict:
    from scipy.stats import spearmanr

    any_run = next(iter(runs.values()))
    n_sent = np.array([len(r.sentences) for r in any_run])
    n_seg = np.array([len(r.true_boundaries) + 1 for r in any_run])
    seg_len = n_sent / n_seg
    # share of segments with at most 3 sentences
    short = np.array([np.mean(np.diff([0] + list(r.true_boundaries) + [len(r.sentences)]) <= 3) for r in any_run])
    bins = {'segment length': (seg_len, [0, 5, 8, 12, np.inf], ['<5', '5-8', '8-12', '>=12']),
            'document length': (n_sent, [0, 25, 50, 100, np.inf], ['<25', '25-50', '50-100', '>=100'])}
    out = {'n_docs': len(any_run), 'features': {}}
    for m, rs in runs.items():
        pk = np.array([r.pk_scores for r in rs])
        rec = {}
        for tol in (0, 1, 2):
            p, r_, f = boundary_prf(rs, tol)
            rec[f'tol{tol}'] = {'precision': p, 'recall': r_, 'f1': f}
        exact = rec['tol0']['recall']
        rec['near_miss_share'] = rec['tol1']['recall'] - exact  # gold boundaries hit only within +-1
        rec['by_bin'] = {}
        for label, (x, edges, names) in bins.items():
            idx = np.digitize(x, edges[1:-1])
            rec['by_bin'][label] = {names[b]: {'n': int((idx == b).sum()),
                                              'pk': float(pk[idx == b].mean()) if (idx == b).any() else None}
                                    for b in range(len(names))}
        rec['spearman'] = {f: float(spearmanr(x, pk).statistic) for f, x in
                           [('segment length', seg_len), ('document length', n_sent),
                            ('segment count', n_seg), ('short-segment share', short)]}
        out[m] = rec
    return out


def worked_example(runs: Dict[str, list], method: str, words: int = 8) -> str:
    rs = runs[method]
    pk = np.array([r.pk_scores for r in rs])
    # a typical document: Pk closest to the median, 15-30 sentences for readability
    cand = [i for i, r in enumerate(rs) if 15 <= len(r.sentences) <= 30]
    i = min(cand, key=lambda i: abs(pk[i] - np.median(pk)))
    r = rs[i]
    lines = [f'Document {i}: {len(r.sentences)} sentences, Pk = {r.pk_scores:.3f}, WD = {r.windiff_scores:.3f}',
             'G = gold boundary before this sentence, P = predicted boundary']
    for j, s in enumerate(r.sentences):
        g = 'G' if j in r.true_boundaries else ' '
        p = 'P' if j in r.pred_boundaries else ' '
        lines.append(f'{j:3d} {g}{p} ' + ' '.join(s.split()[:words]) + (' ...' if len(s.split()) > words else ''))
    return '\n'.join(lines)


# -------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--roots', nargs='+', required=True)
    p.add_argument('--random-root', default=None)
    p.add_argument('--langs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--example-method', default='bayesseg_md')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    report = {}
    for lang in args.langs:
        stripped = load_runs(args.roots, args.random_root, lang, 'stripped')
        echo = load_runs(args.roots, None, lang, 'echo')
        if not stripped:
            continue
        n = check_aligned(stripped)
        print(f'\n######## {lang}: {n} documents, methods: {", ".join(name(m) for m in stripped)}')
        rep = {'n_docs': n,
               'significance_pk': significance(stripped, 'pk'),
               'significance_wd': significance(stripped, 'wd'),
               'echo': echo_pairs(echo, stripped),
               'errors': error_analysis({m: r for m, r in stripped.items() if m != 'llm_mock_k_0shot'})}
        report[lang] = rep
        for metric in ('pk', 'wd'):
            sig = rep[f'significance_{metric}']
            print(f'\n-- {metric.upper()}: mean [95% CI]')
            for m in sig['order']:
                s = sig['summary'][m]
                print(f'   {name(m):18s} {s["mean"]:.3f} [{s["ci95"][0]:.3f}, {s["ci95"][1]:.3f}]')
            print(f'   pairs NOT significant after Holm (p >= 0.05):')
            ns = [x for x in sig['pairs'] if x['p_holm'] >= 0.05]
            for x in ns or [{'a': '-', 'b': '-', 'mean_diff': 0, 'p_holm': 0}]:
                if x['a'] != '-':
                    print(f'     {name(x["a"])} vs {name(x["b"])}: diff {x["mean_diff"]:+.4f}, p_holm {x["p_holm"]:.3f}')
                else:
                    print('     none')
        if rep['echo']:
            print('\n-- heading echo, paired on matched documents')
            for m, e in rep['echo'].items():
                print(f'   {name(m):12s} matched {e["n_matched"]}/{e["n_stripped"]}: Pk {e["pk_echo"]:.3f} -> '
                      f'{e["pk_stripped"]:.3f} (delta {e["delta_pk"]:+.3f} [{e["delta_pk_ci95"][0]:+.3f}, '
                      f'{e["delta_pk_ci95"][1]:+.3f}], p {e["p_wilcoxon"]:.1e}; worse on {e["share_worse"]:.0%} of docs)')
        print('\n-- boundary accuracy (precision / recall, exact and +-1 sentence)')
        for m, e in rep['errors'].items():
            if m in ('n_docs', 'features'):
                continue
            t0, t1 = e['tol0'], e['tol1']
            print(f'   {name(m):12s} exact P {t0["precision"]:.2f} R {t0["recall"]:.2f} | +-1 P {t1["precision"]:.2f} '
                  f'R {t1["recall"]:.2f} | near misses {e["near_miss_share"]:.0%} of gold')
        if args.example_method in stripped:
            ex = worked_example(stripped, args.example_method)
            rep['example'] = ex
            with open(os.path.join(args.out, f'example_{lang}_{args.example_method}.txt'), 'w') as f:
                f.write(ex)

    def default(o):
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))
    with open(os.path.join(args.out, 'analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=default)
    print(f'\nWrote {args.out}/analysis.json and worked examples')


if __name__ == '__main__':
    main()
