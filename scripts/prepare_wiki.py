#!/usr/bin/env python3
"""Download (optional) and extract Wikipedia dumps for Indonesian local languages.

Produces the same format as the Indonesian `text/` folder:
    data/wiki/<lang>/text/AA/wiki_00 ...   (<doc> blocks with <h2>/<h3> headings)

Examples
--------
# Use a dump you already have (e.g. Javanese):
python scripts/prepare_wiki.py --lang jv --dump /path/to/jvwiki-latest-pages-articles.xml.bz2

# Download + extract several languages:
python scripts/prepare_wiki.py --lang jv su min ban

# All registered local languages:
python scripts/prepare_wiki.py --lang all
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from languages import LOCAL_LANGUAGES, get_language  # noqa: E402


def download(url: str, dest: str) -> None:
    """Stream `url` to `dest` using only the standard library."""
    import urllib.request

    if os.path.exists(dest):
        print(f'[skip] {dest} already exists')
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.part'
    req = urllib.request.Request(
        url, headers={'User-Agent': 'indo-text-segmentation/1.0 (research; dump download)'})
    print(f'[download] {url}')
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f'\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB ({100 * done / total:.0f}%)',
                      end='', flush=True)
            else:
                print(f'\r  {done / 1e6:.1f} MB', end='', flush=True)
        print()
    os.replace(tmp, dest)


def extract(dump: str, out_dir: str, processes: int, bytes_per_file: str, overwrite: bool) -> None:
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not overwrite:
            print(f'[skip] {out_dir} is not empty (use --overwrite to re-extract)')
            return
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # --html keeps section headings as <h2>/<h3>, which define the gold boundaries.
    cmd = [sys.executable, '-m', 'wikiextractor.WikiExtractor', '--html',
           '-b', bytes_per_file, '-o', out_dir, dump]
    if processes > 0:
        cmd[4:4] = ['--processes', str(processes)]
    print('[run]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lang', nargs='+', required=True,
                   help=f"Wiki codes, or 'all' for: {' '.join(LOCAL_LANGUAGES)}")
    p.add_argument('--dump', help='Path to an existing *.xml.bz2 dump (only with a single --lang)')
    p.add_argument('--out', default=os.path.join(ROOT, 'data', 'wiki'), help='Output root')
    p.add_argument('--date', default='latest', help='Dump date, e.g. 20261101 (default: latest)')
    p.add_argument('--processes', type=int, default=0, help='wikiextractor worker processes (0 = default)')
    p.add_argument('--bytes', default='1M', help='Max size per output file (same as Indonesian corpus)')
    p.add_argument('--download-only', action='store_true')
    p.add_argument('--overwrite', action='store_true', help='Re-extract even if output exists')
    args = p.parse_args()

    langs = LOCAL_LANGUAGES if args.lang == ['all'] else args.lang
    if args.dump and len(langs) != 1:
        p.error('--dump can only be used with exactly one --lang')

    failed = []
    for code in langs:
        lang = get_language(code)
        lang_dir = os.path.join(args.out, code)
        print(f'=== {code} ({lang.name}) ===')
        try:
            if args.dump:
                dump = os.path.abspath(args.dump)
            else:
                dump = os.path.join(lang_dir, os.path.basename(lang.dump_url(args.date)))
                download(lang.dump_url(args.date), dump)
            if not args.download_only:
                extract(dump, os.path.join(lang_dir, 'text'), args.processes, args.bytes, args.overwrite)
        except Exception as e:  # keep going with other languages
            print(f'[error] {code}: {e}')
            failed.append(code)

    if failed:
        print('Failed:', ' '.join(failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
