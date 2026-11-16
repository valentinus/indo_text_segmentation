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
import re
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


# wikiextractor (<= 3.1.0) crashes with `IndexError: string index out of range`
# at `type = line[l-1]` in extract.compact() when, in --html mode, a list item is
# shorter than the open list nesting (e.g. "* a" / "** b" / "*"). We run a
# patched copy of the installed package instead of editing site-packages.
_BUGGY = re.compile(r'^(?P<ind>[ \t]+)type = line\[l-1\]\n', re.M)
_FIX = (
    '{ind}if l > len(line):  # patched: item shorter than current nesting\n'
    '{ind}    for extra in reversed(listLevel[len(line):]):\n'
    '{ind}        page.append(listClose[extra])\n'
    '{ind}    listLevel = listLevel[:len(line)]\n'
    '{ind}    l = len(line)\n'
    '{ind}type = line[l-1]\n'
)

# wikiextractor 3.1.0 makes --html imply --links, leaving escaped
# `&lt;a href=...&gt;` markup in the text. The Indonesian corpus has none, so
# turn that off to keep the same format.
_HTML_LINKS = re.compile(r'^(?P<ind>[ \t]+)if args\.html:\n(?P=ind)[ \t]+(?P<lhs>(?:Extractor\.)?keepLinks) = True\n', re.M)


def patched_wikiextractor(workdir: str) -> str:
    """Copy the installed wikiextractor into `workdir` with the list bug fixed.

    Returns a directory to prepend to PYTHONPATH (works for fork and spawn)."""
    import wikiextractor

    src = os.path.dirname(wikiextractor.__file__)
    dst_root = os.path.join(workdir, '.wikiextractor_patched')
    dst = os.path.join(dst_root, 'wikiextractor')
    shutil.rmtree(dst_root, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__'))

    def patch(fname, pattern, repl, label):
        path = os.path.join(dst, fname)
        with open(path, encoding='utf-8') as f:
            code = f.read()
        code, n = pattern.subn(repl, code)
        if n:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f'[patch] {label}')
        else:
            print(f'[patch] not needed / not found: {label}')

    patch('extract.py', _BUGGY, lambda m: _FIX.format(ind=m.group('ind')),
          'list-nesting IndexError fix')
    patch('WikiExtractor.py', _HTML_LINKS, '', '--html no longer implies --links')
    return dst_root


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
    env = dict(os.environ)
    patched = patched_wikiextractor(os.path.dirname(os.path.abspath(out_dir)))
    env['PYTHONPATH'] = patched + os.pathsep + env.get('PYTHONPATH', '')
    print('[run]', ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        shutil.rmtree(patched, ignore_errors=True)


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
