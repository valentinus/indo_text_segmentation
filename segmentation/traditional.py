from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize.texttiling import TextTilingTokenizer




def default_tokenize(text: str) -> List[str]:
    return [t.lower() for t in nltk.wordpunct_tokenize(text) if any(c.isalpha() for c in t)]

def tfidf_sentence_matrix(sentences: List[str], max_features: int = 5000) -> Tuple[np.ndarray, TfidfVectorizer]:
    vec = TfidfVectorizer(tokenizer=default_tokenize, lowercase=False,
                          max_features=max_features, norm='l2', min_df=2, max_df=0.5)
    X = vec.fit_transform(sentences).astype(np.float32)
    return X.toarray(), vec

def run_texttiling(sentences: List[str]) -> List[int]:
    if len(" ".join(sentences)) < 150:
        return []
    tt = TextTilingTokenizer()
    text = "\n\n".join(sentences)
    tiles = tt.tokenize(text)

    boundaries = []
    sent_spans: List[Tuple[int,int]] = []
    cur = 0
    for s in sentences:
        start = cur
        cur += len(s) + 2
        sent_spans.append((start, cur))

    tile_starts = []
    pos = 0
    for t in tiles:
        tile_starts.append(pos)
        pos += len(t) + 2

    for ts in tile_starts[1:]:
        idx = 0
        while idx < len(sent_spans) and sent_spans[idx][0] < ts:
            idx += 1
        if 0 < idx < len(sentences):
            boundaries.append(idx)

    boundaries = sorted(set(boundaries))
    return boundaries



def c99_segment(sentences: List[str], window: int = 11, rank_width: int = 11, k_segments: Optional[int] = None) -> List[int]:
    if len(sentences) < 4:
        return []
    X, _ = tfidf_sentence_matrix(sentences, max_features=4000)
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0.0)
    n = S.shape[0]
    R = np.zeros_like(S)
    half = rank_width // 2
    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        for j in range(n):
            j0 = max(0, j - half)
            j1 = min(n, j + half + 1)
            block = S[i0:i1, j0:j1].ravel()
            rank = (block <= S[i, j]).sum()
            R[i, j] = rank
    depth_scores = []
    for b in range(1, n-1):
        left = R[max(0, b-window):b, max(0, b-window):b].mean() if b>0 else 0
        right = R[b:min(n, b+window), b:min(n, b+window)].mean() if b<n else 0
        center = R[max(0, b-window//2):min(n, b+window//2), max(0, b-window//2):min(n, b+window//2)].mean()
        depth = (left + right) / 2.0 - center
        depth_scores.append((depth, b))
    depth_scores.sort(key=lambda x: x[0], reverse=True)
    if k_segments is None or k_segments <= 1:
        vals = np.array([v for v,_ in depth_scores])
        thr = vals.mean() + 0.5*vals.std() if len(vals)>0 else 0
        bounds = [b for v,b in depth_scores if v >= thr]
    else:
        bounds = [b for _, b in depth_scores[:k_segments-1]]
    return sorted(set([b for b in bounds if 0 < b < n]))


def cvs_segment(sentences: List[str], window: int = 3, k_segments: Optional[int] = None) -> List[int]:
    n = len(sentences)
    if n < 2*window + 1:
        return []
    X, _ = tfidf_sentence_matrix(sentences, max_features=5000)
    sims = []
    for i in range(window, n - window):
        a = X[i-window:i].mean(axis=0, keepdims=True)
        b = X[i:i+window].mean(axis=0, keepdims=True)
        sims.append(float(cosine_similarity(a, b)[0,0]))
    sims = np.array(sims)
    loc_min = []
    for j in range(1, len(sims)-1):
        if sims[j] <= sims[j-1] and sims[j] <= sims[j+1]:
            loc_min.append((sims[j], j + window))
    loc_min.sort(key=lambda x: x[0])
    if k_segments is None or k_segments <= 1:
        thr = sims.mean() - 0.5 * sims.std() if len(sims) else -1
        bounds = [idx for val, idx in loc_min if val <= thr]
    else:
        bounds = [idx for _, idx in loc_min[:k_segments-1]]
    return sorted(set([b for b in bounds if 0 < b < n]))