#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import json
import random
import time
from functools import partial
from typing import List, Optional, Dict

import numpy as np
import pickle

import nltk

from segmentation.traditional import c99_segment, cvs_segment, run_texttiling
from segmentation.bayes import BayesSeg, BayesSegMD, BeamSeg

from languages import LANGUAGES, get_language
from utils import SegmentDoc, SegmentResult, pk_metric, windowdiff, parse_headings_and_content
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# torch / transformers / gensim / bertopic are imported lazily, only for the
# methods that need them, so classic methods also run on CPU-only machines.

ALL_METHODS = ['texttiling', 'c99', 'cvs', 'topictiling', 'bayesseg', 'bayesseg_md', 'beamseg', 'bert', 'bertopic']
DEFAULT_BERT_MODEL = 'LazarusNLP/congen-indobert-base'
DEFAULT_EMBED_MODEL = 'Qwen/Qwen3-Embedding-0.6B'


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)



def evaluate_on_docs(
    docs: List[SegmentDoc],
    method: str,
    lda_model=None,
    bert_model=None,
    encoder=None,
    llm_model=None,
    window: int = 3,
    penalty_lambda: float = 8.0,
    alpha: float = 1.,
    gamma: float = 0.,
    progress: bool = True,
) -> Dict[str, float]:
    if method == 'bertopic':
        from segmentation.bertopic_segmenter import (
            SentenceEncoder, bertopic_topic_vectors, cohshift_scores, pick_boundaries_from_scores)
        if encoder is None:
            encoder = SentenceEncoder()

    def segment_one(d: SegmentDoc, k_segments: int) -> List[int]:
        if method == 'texttiling':
            return run_texttiling(d.sentences)
        if method == 'c99':
            return c99_segment(d.sentences, window=max(5, 2*window+1), k_segments=k_segments)
        if method == 'cvs':
            return cvs_segment(d.sentences, window=window, k_segments=k_segments)
        if method == 'topictiling':
            assert lda_model is not None
            return lda_model.segment(d.sentences, k_segments=k_segments, window=window)
        if method == 'bayesseg':
            return BayesSeg().segment_k(d.sentences, k_segments=k_segments)
        if method == 'bayesseg_md':
            return BayesSegMD(penalty_lambda=penalty_lambda).segment(d.sentences)
        if method == 'beamseg':
            return BeamSeg(beam_size=8).segment_k(d.sentences, k_segments=k_segments)
        if method == 'bert':
            assert bert_model is not None
            return bert_model.segment(d.sentences, k_segments=k_segments, window=window)
        if method == 'llm':
            assert llm_model is not None
            return llm_model.segment(d.sentences, k_segments=k_segments)
        if method == 'bertopic':
            T = bertopic_topic_vectors(d.sentences, encoder)
            scores, idxs = cohshift_scores(T, windows=None, metric='cos', alpha=alpha, gamma=gamma)
            return pick_boundaries_from_scores(scores, idxs, k_segments=k_segments)
        raise KeyError(method)

    results = []
    n_failed = 0
    for d in tqdm(docs, desc=method, disable=not progress):
        n = len(d.sentences)
        k_segments = len(d.true_boundaries) + 1
        try:
            pred = segment_one(d, k_segments)
        except ValueError:
            # e.g. TF-IDF keeps no terms (min_df/max_df) on a very repetitive document:
            # predict no boundary and count it instead of aborting the run.
            pred = []
            n_failed += 1
        _pk = pk_metric(d.true_boundaries, pred, n)
        _wd = windowdiff(d.true_boundaries, pred, n)
        results.append(SegmentResult(sentences=d.sentences, true_boundaries=d.true_boundaries,
                                     pred_boundaries=pred, pk_scores=_pk, windiff_scores=_wd))
    metrics = _metrics(results)
    metrics['n_failed'] = n_failed
    return metrics, results


def _metrics(results: List[SegmentResult]) -> Dict[str, float]:
    pk = [r.pk_scores for r in results]
    wd = [r.windiff_scores for r in results]
    return {
        'Pk_mean': float(np.mean(pk)),
        'Pk_median': float(np.median(pk)),
        'WinDiff_mean': float(np.mean(wd)),
        'WinDiff_median': float(np.median(wd)),
    }


# Set before the pool forks, so workers share models without pickling them.
_PAR_ARGS: Dict = {}


def _eval_chunk(chunk: List[SegmentDoc]):
    metrics, results = evaluate_on_docs(chunk, progress=False, **_PAR_ARGS)
    return results, metrics['n_failed']


def evaluate_parallel(docs: List[SegmentDoc], method: str, workers: int, **kw):
    """Same output as evaluate_on_docs, with documents split across CPU processes.

    Only for CPU methods; results keep the input document order."""
    global _PAR_ARGS
    if workers <= 1 or method in ('bert', 'bertopic'):
        return evaluate_on_docs(docs, method, **kw)
    _PAR_ARGS = dict(method=method, **kw)
    size = max(1, len(docs) // (workers * 16))
    chunks = [docs[i:i + size] for i in range(0, len(docs), size)]
    results: List[SegmentResult] = []
    n_failed = 0
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers) as pool:
        for part, nf in tqdm(pool.imap(_eval_chunk, chunks), total=len(chunks), desc=f'{method} x{workers}'):
            results.extend(part)
            n_failed += nf
    metrics = _metrics(results)
    metrics['n_failed'] = n_failed
    return metrics, results


def list_files_walk(start_path='.'):
    files_path = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file.startswith('wiki_'):
                files_path.append(os.path.join(root, file))
    return files_path


def default_doc_path(lang: str) -> str:
    # The original Indonesian corpus lives in text/; other languages in data/wiki/<lang>/text.
    if lang == 'id' and os.path.isdir('text'):
        return 'text'
    return os.path.join('data', 'wiki', lang, 'text')


def load_docs(files_path: List[str], parse_kw: Dict) -> List[SegmentDoc]:
    docs: List[SegmentDoc] = []
    fn = partial(parse_headings_and_content, **parse_kw)
    # imap (ordered) keeps the document order deterministic across runs.
    with Pool(processes=cpu_count()) as pool:
        for chunk in tqdm(pool.imap(fn, files_path, chunksize=8),
                          total=len(files_path), desc="Parsing"):
            docs.extend(chunk)
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='id',
                        help=f"Wikipedia language code ({', '.join(LANGUAGES)}; others allowed)")
    parser.add_argument('--doc-path', type=str, default=None,
                        help='wikiextractor output dir (default: text/ for id, data/wiki/<lang>/text otherwise)')
    parser.add_argument('--lda-topics', type=int, default=50)
    parser.add_argument('--lda-passes', type=int, default=5)
    parser.add_argument('--window', type=int, default=3, help='Sliding window size (sentences)')
    parser.add_argument('--method', type=str, default='all', choices=['all'] + ALL_METHODS + ['llm'],
                        help="'all' runs every non-LLM method; 'llm' (paid API calls) must be asked for")
    parser.add_argument('--penalty', type=float, default=8.0, help='MDL penalty (BayesSeg-MD)')
    parser.add_argument('--alpha', type=float, default=1.0, help='coherence exponent')
    parser.add_argument('--gamma', type=float, default=0.0, help='intra-window concentration exponent')
    parser.add_argument('--bert-model', type=str, default=DEFAULT_BERT_MODEL,
                        help='HF encoder for the "bert" method, e.g. LazarusNLP/NusaBERT-base, sentence-transformers/LaBSE')
    parser.add_argument('--embed-model', type=str, default=DEFAULT_EMBED_MODEL,
                        help='HF encoder for the "bertopic" method')
    parser.add_argument('--device', type=str, default=None, help='cuda / cpu (default: auto)')
    # LLM segmentation (segmentation/llm_segmenter.py)
    parser.add_argument('--llm-backend', default='anthropic', choices=['anthropic', 'hf', 'mock'],
                        help="anthropic = Claude API, hf = local open-weight chat model, mock = dry run")
    parser.add_argument('--llm-model', default='claude-opus-5-5',
                        help='Claude model id, or a HF model id with --llm-backend hf')
    parser.add_argument('--llm-mode', default='k', choices=['k', 'free'],
                        help='k = number of sections given (oracle k); free = model decides')
    parser.add_argument('--llm-shots', type=int, default=0, help='Few-shot examples (held out from the evaluated docs)')
    parser.add_argument('--llm-effort', default='medium', choices=['low', 'medium', 'high', 'xhigh', 'max'],
                        help='Claude effort level')
    parser.add_argument('--llm-no-fallback', action='store_true',
                        help='Disable Claude server-side refusal fallback (no answers from another model)')
    parser.add_argument('--llm-concurrency', type=int, default=8)
    parser.add_argument('--llm-4bit', action='store_true',
                        help='hf backend: load the model in 4-bit (bitsandbytes), e.g. 8-9B models on a 16 GB GPU')
    # Document filters (defaults = original Indonesian setup)
    parser.add_argument('--min-headings', type=int, default=3)
    parser.add_argument('--min-boundaries', type=int, default=3)
    parser.add_argument('--min-sentences', type=int, default=11)
    parser.add_argument('--strip-heading-echo', action='store_true',
                        help='Drop the repeated heading sentence ("Sajarah.") wikiextractor puts at each section start')
    parser.add_argument('--skip-headings-file', type=str, default=None,
                        help='Text file, one boilerplate heading per line (overrides languages.py)')
    parser.add_argument('--max-docs', type=int, default=None,
                        help='Evaluate on a seeded random sample of N documents')
    parser.add_argument('--max-files', type=int, default=None, help='Only parse the first N files (quick tests)')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=True, help='Save the results')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--tag', type=str, default=None,
                        help='Name of the saved pickle (default: window-<window>)')
    parser.add_argument('--workers', type=int, default=1,
                        help='CPU processes for per-document evaluation (classic/topic methods only)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)  # used by TextTiling
    except Exception:
        pass

    lang = get_language(args.lang)
    directory_path = args.doc_path or default_doc_path(args.lang)
    if not os.path.isdir(directory_path):
        raise SystemExit(f'No corpus at {directory_path}. Run: python scripts/prepare_wiki.py --lang {args.lang}')
    files_path = sorted(list_files_walk(directory_path))
    if args.max_files:
        files_path = files_path[:args.max_files]

    skip_headings = None
    if args.skip_headings_file:
        with open(args.skip_headings_file, encoding='utf-8') as f:
            skip_headings = [l.strip() for l in f if l.strip()]
    parse_kw = dict(lang=args.lang, skip_headings=skip_headings,
                    min_headings=args.min_headings, min_boundaries=args.min_boundaries,
                    min_sentences=args.min_sentences, strip_heading_echo=args.strip_heading_echo)

    print(f'Language: {args.lang} ({lang.name}) | corpus: {directory_path} | files: {len(files_path)}')
    docs = load_docs(files_path, parse_kw)
    all_docs = docs
    if args.max_docs and len(docs) > args.max_docs:
        docs = random.Random(args.seed).sample(docs, args.max_docs)
    if not docs:
        raise SystemExit('No documents passed the filters (try lowering --min-headings / --min-sentences).')
    print(f'Documents: {len(docs)} | avg sentences: {np.mean([len(d.sentences) for d in docs]):.1f} '
          f'| avg segments: {np.mean([len(d.true_boundaries) + 1 for d in docs]):.1f}')

    results = {}
    save = args.save
    tag = args.tag or f'window-{args.window}'
    out_root = os.path.join(args.results_dir, args.lang)

    methods = ALL_METHODS if args.method == 'all' else [args.method]

    lda = None
    if 'topictiling' in methods:
        from segmentation.lda_segmenter import LdaTopicTiling
        print('Training LDA for topic-based methods...')
        lda = LdaTopicTiling(num_topics=args.lda_topics, passes=args.lda_passes, seed=args.seed)
        lda.fit((d.sentences for d in docs), max_docs=3000)

    bert = None
    if 'bert' in methods:
        from segmentation.bert_segmenter import BertSegmenter
        print(f'Loading BERT encoder {args.bert_model}...')
        bert = BertSegmenter(model_name=args.bert_model, device=args.device)

    encoder = None
    if 'bertopic' in methods:
        from segmentation.bertopic_segmenter import SentenceEncoder
        print(f'Loading sentence encoder {args.embed_model}...')
        encoder = SentenceEncoder(model_name=args.embed_model, device=args.device)

    llm = None
    llm_name = None
    if 'llm' in methods:
        from segmentation.llm_segmenter import LLMSegmenter
        shots = []
        if args.llm_shots:
            in_eval = {id(d) for d in docs}
            pool = [d for d in all_docs if id(d) not in in_eval
                    and 20 <= len(d.sentences) <= 40 and len(d.true_boundaries) >= 3]
            if len(pool) < args.llm_shots:
                raise SystemExit('Not enough held-out documents for few-shot examples (use --max-docs).')
            shots = random.Random(args.seed + 1).sample(pool, args.llm_shots)
        model_label = 'mock' if args.llm_backend == 'mock' else args.llm_model.replace('/', '_')
        llm_name = f"llm_{model_label}_{args.llm_mode}_{args.llm_shots}shot"
        variant = args.llm_effort if args.llm_backend == 'anthropic' else ('4bit' if args.llm_4bit else 'bf16')
        cache = os.path.join(out_root, 'llm_cache', f'{llm_name}_{variant}.jsonl')
        llm = LLMSegmenter(backend=args.llm_backend, model=args.llm_model, mode=args.llm_mode,
                           language=lang.name, shots=shots, effort=args.llm_effort,
                           fallback=not args.llm_no_fallback, cache_path=cache,
                           concurrency=args.llm_concurrency, seed=args.seed, hf_4bit=args.llm_4bit)
        print(f'LLM: {args.llm_backend}/{args.llm_model} mode={args.llm_mode} shots={args.llm_shots} '
              f'effort={args.llm_effort} fallback={not args.llm_no_fallback} cache={cache}')
        llm.prefetch(docs, desc=llm_name)

    def run(m: str):
        print(f'Evaluating {m}...')
        kw = {'lda_model': lda, 'bert_model': bert, 'encoder': encoder, 'llm_model': llm, 'window': args.window,
              'penalty_lambda': args.penalty, 'alpha': args.alpha, 'gamma': args.gamma}
        t0 = time.time()
        res, seg_res = evaluate_parallel(docs, m, 1 if m == 'llm' else args.workers, **kw)
        res['runtime_sec'] = round(time.time() - t0, 1)
        res['workers'] = args.workers if m not in ('bert', 'bertopic') else 1
        if m == 'llm':
            res.update(llm.usage_report(docs))
            m = llm_name
        if save:
            os.makedirs(os.path.join(out_root, m), exist_ok=True)
            with open(os.path.join(out_root, m, f'{tag}.pkl'), 'wb') as file:
                pickle.dump(seg_res, file, protocol=pickle.HIGHEST_PROTOCOL)
        results[m] = res
        print(m + ':', res)

    for m in methods:
        run(m)

    if save:
        os.makedirs(out_root, exist_ok=True)
        summary_path = os.path.join(out_root, f'summary_{tag}.json')
        summary = {'lang': args.lang, 'results': {}}
        if os.path.exists(summary_path):  # merge with earlier single-method runs
            with open(summary_path, encoding='utf-8') as f:
                summary = json.load(f)
        summary.setdefault('results', {}).update(results)
        summary.setdefault('runs', {}).update({m: {'n_docs': len(docs), 'args': vars(args)} for m in results})
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'=== Summary [{args.lang}] (lower is better) ===')
    for name, metrics in results.items():
        print(f'{name}')
        for k, v in metrics.items():
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')


if __name__ == '__main__':
    main()
