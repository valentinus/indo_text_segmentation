#!/usr/bin/env python3
"""Corpus statistics per language, and frequent headings to extend skip lists.

Examples
--------
python scripts/corpus_stats.py --lang jv
python scripts/corpus_stats.py --lang id jv su --csv results/corpus_stats.csv
python scripts/corpus_stats.py --lang jv --top-headings 60   # find boilerplate sections
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys
from functools import partial
from multiprocessing import Pool, cpu_count

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

from languages import get_language  # noqa: E402
from main import default_doc_path, list_files_walk  # noqa: E402
from utils import parse_headings_and_content  # noqa: E402

_HEADING = re.compile(r'<h[2-4]>([^<]*)</h[2-4]>')


def count_raw(path: str):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    return text.count('<doc '), collections.Counter(h.strip() for h in _HEADING.findall(text))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lang', nargs='+', required=True)
    p.add_argument('--top-headings', type=int, default=30, help='Show the N most frequent headings')
    p.add_argument('--strip-heading-echo', action='store_true')
    p.add_argument('--csv', help='Write the summary table to this CSV file')
    args = p.parse_args()

    rows = []
    for code in args.lang:
        path = default_doc_path(code)
        files = sorted(list_files_walk(path))
        if not files:
            print(f'[{code}] no files under {path}, skipping')
            continue
        lang = get_language(code)
        fn = partial(parse_headings_and_content, lang=code, strip_heading_echo=args.strip_heading_echo)
        with Pool(cpu_count()) as pool:
            raw = pool.map(count_raw, files, chunksize=4)
            docs = [d for chunk in pool.map(fn, files, chunksize=4) for d in chunk]
        n_articles = sum(r[0] for r in raw)
        headings = sum((r[1] for r in raw), collections.Counter())

        n_sent = [len(d.sentences) for d in docs] or [0]
        n_seg = [len(d.true_boundaries) + 1 for d in docs] or [0]
        seg_len = [l for d in docs for l in d.segment_length] or [0]
        row = {
            'lang': code, 'name': lang.name, 'articles': n_articles, 'usable_docs': len(docs),
            'usable_pct': round(100 * len(docs) / max(1, n_articles), 2),
            'avg_sentences': round(float(np.mean(n_sent)), 1),
            'avg_segments': round(float(np.mean(n_seg)), 1),
            'avg_segment_len': round(float(np.mean(seg_len)), 1),
            'total_tokens': int(sum(d.token_count for d in docs)),
        }
        rows.append(row)
        print(f'\n=== {code} ({lang.name}) ===')
        for k, v in row.items():
            print(f'  {k}: {v}')

        if args.top_headings:
            norm = (lambda h: h) if lang.case_sensitive_headings else str.casefold
            skip = {norm(h) for h in lang.skip_headings}
            print(f'  top {args.top_headings} headings ([skip] = in languages.py boilerplate list):')
            for h, c in headings.most_common(args.top_headings):
                mark = '[skip]' if norm(h) in skip else '      '
                print(f'    {c:7d} {mark} {h}')

    if args.csv and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'\nWrote {args.csv}')


if __name__ == '__main__':
    main()
