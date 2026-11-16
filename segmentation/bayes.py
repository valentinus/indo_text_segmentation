from __future__ import annotations

import math
from typing import List, Tuple, Optional

import numpy as np
import nltk

from sklearn.feature_extraction.text import CountVectorizer

def default_tokenize(text: str) -> List[str]:
    return [t.lower() for t in nltk.wordpunct_tokenize(text) if any(c.isalpha() for c in t)]

def bow_sentence_matrix(sentences: List[str], vocab: Optional[CountVectorizer] = None, max_features: int = 4000):
    if vocab is None:
        vocab = CountVectorizer(tokenizer=default_tokenize, lowercase=False, max_features=max_features,
                                min_df=2, max_df=0.6)
        X = vocab.fit_transform(sentences)
    else:
        X = vocab.transform(sentences)
    return X.toarray().astype(np.int32), vocab

class BayesSeg:
    def __init__(self, alpha: float = 0.1, max_vocab: int = 3000):
        self.alpha = alpha
        self.max_vocab = max_vocab
        self.vocab: Optional[CountVectorizer] = None

    @staticmethod
    def _lgamma(x: np.ndarray) -> np.ndarray:
        vfunc = np.frompyfunc(math.lgamma, 1, 1)
        return vfunc(x)

    def _seg_log_marginal(self, counts: np.ndarray) -> float:
        V = counts.shape[0]
        alpha = self.alpha
        N = counts.sum()
        if N == 0:
            return 0.0
        a0 = V * alpha
        lgamma = BayesSeg._lgamma
        ll = float(lgamma(a0) - lgamma(a0 + N) + np.sum(lgamma(alpha + counts) - lgamma(alpha)))
        return ll

    def _prepare_counts(self, sentences: List[str]):
        X, vocab = bow_sentence_matrix(sentences, max_features=self.max_vocab)
        self.vocab = vocab
        C = np.zeros((X.shape[0]+1, X.shape[1]), dtype=np.int32)
        C[1:] = np.cumsum(X, axis=0)
        return C

    def _seg_counts(self, C: np.ndarray, i: int, j: int) -> np.ndarray:
        return (C[j] - C[i]).astype(np.int32)

    def _segment_ll(self, sentences: List[str], Lmax: int = 80) -> np.ndarray:
        n = len(sentences)
        C = self._prepare_counts(sentences)
        seg_ll = np.full((n + 1, n + 1), -np.inf)
        for i in range(n):
            for j in range(i + 1, min(n, i + Lmax) + 1):
                seg_ll[i, j] = self._seg_log_marginal(self._seg_counts(C, i, j))
        return seg_ll

    def segment_k(self, sentences: List[str], k_segments: int, Lmax: int = 80) -> List[int]:
        """Exact MAP segmentation into k segments (dynamic programming).

        dp[s, j] = best log marginal likelihood of sentences[0:j] split into s segments.
        """
        n = len(sentences)
        if k_segments <= 1 or n <= 1:
            return []
        k = min(k_segments, n)
        seg_ll = self._segment_ll(sentences, Lmax)
        dp = np.full((k + 1, n + 1), -np.inf)
        prev = np.full((k + 1, n + 1), -1, dtype=int)
        dp[0, 0] = 0.0
        for s in range(1, k + 1):
            for j in range(s, n + 1):
                lo = max(s - 1, j - Lmax)
                cand = dp[s - 1, lo:j] + seg_ll[lo:j, j]
                if cand.size == 0:
                    continue
                b = int(np.argmax(cand))
                dp[s, j] = cand[b]
                prev[s, j] = lo + b
        if not np.isfinite(dp[k, n]):
            return []
        bounds, j = [], n
        for s in range(k, 1, -1):
            j = int(prev[s, j])
            bounds.append(j)
        return sorted(b for b in bounds if 0 < b < n)

    def segment_k_legacy(self, sentences: List[str], k_segments: int) -> List[int]:
        """Original implementation used in the conference paper. It writes each DP step
        into the row it reads from (dp[s-1]), so it does not search over k segments and
        returns about one boundary per document. Kept only to reproduce old numbers."""
        n = len(sentences)
        if k_segments <= 1 or n <= 1:
            return []
        C = self._prepare_counts(sentences)
        V = C.shape[1]
        Lmax = 80
        seg_ll = np.full((n+1, n+1), -1e18)
        for i in range(n):
            for j in range(i+1, min(n, i+Lmax)+1):
                cnt = self._seg_counts(C, i, j)
                seg_ll[i, j] = self._seg_log_marginal(cnt)
        dp = np.full((k_segments, n+1), -1e18)
        prev = np.full((k_segments, n+1), -1, dtype=int)
        dp[0, 0] = 0.0
        for s in range(1, k_segments+1):
            for j in range(1, n+1):
                best = -1e18
                best_i = -1
                for i in range(max(0, j-Lmax), j):
                    cand = dp[s-1, i] + seg_ll[i, j]
                    if cand > best:
                        best = cand
                        best_i = i
                dp[s-1, j] = best
                prev[s-1, j] = best_i
        bounds = []
        j = n
        for s in range(k_segments-1, 0, -1):
            i = prev[s-1, j]
            bounds.append(i)
            j = i
        return sorted([b for b in bounds if 0 < b < n])
    

class BayesSegMD(BayesSeg):
    def __init__(self, alpha: float = 0.1, max_vocab: int = 3000, penalty_lambda: float = 8.0, Kmax: int = 10):
        super().__init__(alpha=alpha, max_vocab=max_vocab)
        self.penalty_lambda = penalty_lambda
        self.Kmax = Kmax

    def segment(self, sentences: List[str]) -> List[int]:
        n = len(sentences)
        if n <= 1:
            return []
        C = self._prepare_counts(sentences)
        Lmax = 80
        seg_ll = np.full((n+1, n+1), -1e18)
        for i in range(n):
            for j in range(i+1, min(n, i+Lmax)+1):
                seg_ll[i, j] = self._seg_log_marginal(self._seg_counts(C, i, j));
        best_score = np.full(n+1, -1e18)
        prev = np.full(n+1, -1, dtype=int)
        segs = np.full(n+1, 0, dtype=int)
        best_score[0] = 0.0
        for j in range(1, n+1):
            for i in range(max(0, j-Lmax), j):
                kprev = segs[i]
                cand = best_score[i] + seg_ll[i, j] - self.penalty_lambda  # pay per boundary
                if cand > best_score[j] and segs[i] + 1 <= self.Kmax:
                    best_score[j] = cand
                    prev[j] = i
                    segs[j] = segs[i] + 1
        bounds = []
        j = n
        while prev[j] != -1 and j > 0:
            i = prev[j]
            bounds.append(i)
            j = i
        return sorted([b for b in bounds if 0 < b < n])
    

class BeamSeg(BayesSeg):
    def __init__(self, alpha: float = 0.1, max_vocab: int = 3000, beam_size: int = 8, Lmax: int = 80):
        super().__init__(alpha=alpha, max_vocab=max_vocab)
        self.beam_size = beam_size
        self.Lmax = Lmax

    def segment_k(self, sentences: List[str], k_segments: int) -> List[int]:
        """Beam search over segment ends for exactly k segments.

        After each step, hypotheses ending at the same sentence are recombined (only
        the best is kept), and the beam keeps the `beam_size` ends with the highest
        log-likelihood per covered sentence, so prefixes of different lengths are
        compared fairly. Segment t may end at most at n - (k - t); the last ends at n.
        With beam_size >= n the search is exact (equal to BayesSeg.segment_k).
        """
        n = len(sentences)
        if k_segments <= 1 or n <= 1:
            return []
        k = min(k_segments, n)
        seg_ll = self._segment_ll(sentences, self.Lmax)
        states = {0: (0.0, [])}  # end position -> (score, boundaries)
        for t in range(1, k + 1):
            nxt = {}
            for last, (score, bounds) in states.items():
                if t < k:
                    # leave >= 1 sentence per remaining segment, and no more than Lmax each
                    lo = max(last + 1, n - (k - t) * self.Lmax)
                    ends = range(lo, min(n - (k - t), last + self.Lmax) + 1)
                else:
                    ends = [n] if n - last <= self.Lmax else []
                for j in ends:
                    sc = score + seg_ll[last, j]
                    if np.isfinite(sc) and (j not in nxt or sc > nxt[j][0]):
                        nxt[j] = (sc, bounds + ([j] if j < n else []))
            if not nxt:
                return []
            keep = sorted(nxt, key=lambda j: nxt[j][0] / j, reverse=True)[:self.beam_size]
            states = {j: nxt[j] for j in keep}
        if n not in states:
            return []
        return sorted(b for b in states[n][1] if 0 < b < n)

    def segment_k_legacy(self, sentences: List[str], k_segments: int) -> List[int]:
        n = len(sentences)
        if k_segments <= 1 or n <= 1:
            return []
        C = self._prepare_counts(sentences)
        seg_ll = {}
        for i in range(n):
            for j in range(i+1, min(n, i+self.Lmax)+1):
                seg_ll[(i,j)] = self._seg_log_marginal(self._seg_counts(C, i, j))
        beam: List[Tuple[float, int, List[int]]] = [(0.0, 0, [])]
        for _ in range(k_segments):
            new_beam: List[Tuple[float, int, List[int]]] = []
            for score, last, bounds in beam:
                for j in range(last+1, min(n, last+self.Lmax)+1):
                    s = score + seg_ll[(last, j)]
                    new_beam.append((s, j, bounds + ([j] if j < n else [])))
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:self.beam_size]
        beam = [(s, j, b) for (s,j,b) in beam if j == n]
        if not beam:
            return []
        best = max(beam, key=lambda x: x[0])
        bounds = [x for x in best[2] if x < n]
        return sorted(set([b for b in bounds if 0 < b < n]))