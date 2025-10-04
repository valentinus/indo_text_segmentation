#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import random
from typing import List, Optional, Dict

import numpy as np
import pickle

import nltk

import torch

from segmentation.traditional import c99_segment, cvs_segment, run_texttiling
from segmentation.bert_segmenter import BertSegmenter
from segmentation.lda_segmenter import LdaTopicTiling
from segmentation.bayes import BayesSeg, BayesSegMD, BeamSeg
from segmentation.bertopic_segmenter import SentenceEncoder, bertopic_topic_vectors, cohshift_scores, pick_boundaries_from_scores

from utils import SegmentDoc, SegmentResult, pk_metric, windowdiff, parse_headings_and_content
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def evaluate_on_docs(
    docs: List[SegmentDoc],
    method: str,
    lda_model: Optional[LdaTopicTiling] = None,
    bert_model: Optional[BertSegmenter] = None,
    window: int = 3,
    penalty_lambda: float = 8.0,
    alpha: float = 1.,
    gamma: float = 0.
) -> Dict[str, float]:
    encoder: Optional[SentenceEncoder] = None
    if method in ('bertopic'):
        encoder = SentenceEncoder(device='cuda')
    
    pk_scores = []
    wd_scores = []
    results = []
    for d in tqdm(docs, desc=method):
        n = len(d.sentences)
        k_segments = len(d.true_boundaries) + 1
        # try:
        if method == 'texttiling':
            pred = run_texttiling(d.sentences)
        elif method == 'c99':
            pred = c99_segment(d.sentences, window=max(5, 2*window+1), k_segments=k_segments)
        elif method == 'cvs':
            pred = cvs_segment(d.sentences, window=window, k_segments=k_segments)
        elif method == 'topictiling':
            assert lda_model is not None
            pred = lda_model.segment(d.sentences, k_segments=k_segments, window=window)
        elif method == 'bayesseg':
            pred = BayesSeg().segment_k(d.sentences, k_segments=k_segments)
        elif method == 'bayesseg_md':
            pred = BayesSegMD(penalty_lambda=penalty_lambda).segment(d.sentences)
        elif method == 'beamseg':
            pred = BeamSeg(beam_size=8).segment_k(d.sentences, k_segments=k_segments)
        elif method == 'bert':
            assert bert_model is not None
            pred = bert_model.segment(d.sentences, k_segments=k_segments, window=window)
        elif method == 'bertopic':
            T = bertopic_topic_vectors(d.sentences, encoder)
            # pred = boundaries_from_topic_vectors(T, k_segments=k_segments, window=window, metric='cos')
            scores, idxs = cohshift_scores(T, windows=None, metric='cos', alpha=alpha, gamma=gamma)
            pred = pick_boundaries_from_scores(scores, idxs, k_segments=k_segments)
        else:
            raise ValueError(method)
        # except:
        #     pass
        _pk = pk_metric(d.true_boundaries, pred, n)
        _wd = windowdiff(d.true_boundaries, pred, n)
        pk_scores.append(_pk)
        wd_scores.append(_wd)
        results.append(SegmentResult(sentences=d.sentences, true_boundaries=d.true_boundaries, 
                                     pred_boundaries=pred, pk_scores=_pk, windiff_scores=_wd))
    return {
        'Pk_mean': float(np.mean(pk_scores)),
        'Pk_median': float(np.median(pk_scores)),
        'WinDiff_mean': float(np.mean(wd_scores)),
        'WinDiff_median': float(np.median(wd_scores)),
    }, results

def list_files_walk(start_path='.'):
    files_path = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            files_path.append(os.path.join(root, file))
    return files_path

def _process_one(path):
    return parse_headings_and_content(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc-path', type=str, help='Document path for the dataset')
    parser.add_argument('--lda-topics', type=int, default=50)
    parser.add_argument('--lda-passes', type=int, default=5)
    parser.add_argument('--window', type=int, default=3, help='Sliding window size (sentences)')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all','texttiling','c99','cvs','topictiling','bayesseg','bayesseg_md','beamseg','bert','bertopic'])
    parser.add_argument('--penalty', type=float, default=8.0, help='MDL penalty (BayesSeg-MD)')
    parser.add_argument('--alpha', type=float, default=1.0, help='coherence exponent')
    parser.add_argument('--gamma', type=float, default=0.0, help='intra-window concentration exponent')
    parser.add_argument('--save', type=bool, default=True, help='Save the results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except Exception:
        pass
    
    directory_path = args.doc_path
    files_path = sorted(list_files_walk(directory_path))
    docs = []
    with Pool(processes=cpu_count()) as pool:
        for chunk in tqdm(pool.imap_unordered(_process_one, files_path, chunksize=32), 
                          total=len(files_path), desc="Parsing"):
            docs.extend(chunk)
            
    
    results = {}
    save = args.save

    need_lda = args.method in ('all','topictiling',)
    lda = None
    if need_lda:
        print('Training LDA for topic-based methods...')
        lda = LdaTopicTiling(num_topics=args.lda_topics, passes=args.lda_passes)
        lda.fit((d.sentences for d in docs), max_docs=3000)

    need_bert = args.method in ('all','bert')
    bert = None
    if need_bert:
        print('Loading BERT encoder...')
        bert = BertSegmenter()

    def run(m: str):
        print(f'Evaluating {m}...')
        kw = {'lda_model': lda, 'bert_model': bert, 'window': args.window, 'penalty_lambda': args.penalty}
        res, seg_res = evaluate_on_docs(docs, m, **kw)
        if save:
            with open(f'results/{m}/congen_window-9.pkl', 'wb') as file:
                pickle.dump(seg_res, file, protocol=pickle.HIGHEST_PROTOCOL)
        results[m] = res
        print(m + ':', res)

    if args.method == 'all':
        for m in ['texttiling','c99','cvs','topictiling','bayesseg','bayesseg_md','beamseg','bert', 'bertopic']:
            run(m)
    else:
        run(args.method)

    print('=== Summary (lower is better) ===')
    for name, metrics in results.items():
        print(f'{name}')
        for k, v in metrics.items():
            print(f'  {k}: {v:.4f}')


if __name__ == '__main__':
    main()
