from __future__ import annotations

from typing import List, Iterable, Optional

import numpy as np
import math

import nltk

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    def _kl(a, b):
        a = np.clip(a, eps, 1)
        b = np.clip(b, eps, 1)
        return float((a * np.log(a / b)).sum())
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


class LdaTopicTiling:
    def __init__(self, num_topics: int = 50, passes: int = 5, seed: int = 42):
        self.num_topics = num_topics
        self.passes = passes
        self.dictionary: Optional[Dictionary] = None
        self.lda: Optional[LdaModel] = None
        self.seed = seed

    @staticmethod
    def simple_tokenize(text: str) -> List[str]:
        return [t.lower() for t in nltk.wordpunct_tokenize(text) if any(c.isalpha() for c in t)]

    def fit(self, docs: Iterable[List[str]], max_docs: int = 5000):
        texts = []
        for i, sents in enumerate(docs):
            if i >= max_docs:
                break
            tokens = []
            for s in sents:
                tokens.extend(self.simple_tokenize(s))
            texts.append(tokens)
        self.dictionary = Dictionary(texts)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=50000)
        corpus = [self.dictionary.doc2bow(t) for t in texts]
        self.lda = LdaModel(corpus=corpus, id2word=self.dictionary, num_topics=self.num_topics,
                            random_state=self.seed, passes=self.passes, iterations=1000)

    def sentence_topic_dist(self, sentence: str) -> np.ndarray:
        bow = self.dictionary.doc2bow(self.simple_tokenize(sentence))
        dense = np.zeros(self.num_topics, dtype=float)
        for k, p in self.lda.get_document_topics(bow, minimum_probability=0.0):
            dense[k] = p
        s = dense.sum()
        return dense / s if s > 0 else dense

    def segment(self, sentences: List[str], k_segments: Optional[int] = None, window: int = 3) -> List[int]:
        if self.dictionary is None or self.lda is None:
            raise RuntimeError("Call fit() before segment().")
        n = len(sentences)
        if n < 2*window + 1:
            return []
        topic_sents = np.stack([self.sentence_topic_dist(s) for s in sentences], axis=0)
        scores = []
        for i in range(window, n - window):
            left = topic_sents[i-window:i].mean(axis=0)
            right = topic_sents[i:i+window].mean(axis=0)
            scores.append(js_divergence(left, right))
        scores = np.array(scores)
        loc_max = []
        for j in range(1, len(scores)-1):
            if scores[j] >= scores[j-1] and scores[j] >= scores[j+1]:
                loc_max.append((scores[j], j + window))
        loc_max.sort(key=lambda x: x[0], reverse=True)

        if k_segments is None or k_segments <= 1:
            thr = scores.mean() + 0.5 * scores.std()
            boundaries = [idx for val, idx in loc_max if val >= thr]
        else:
            boundaries = [idx for _, idx in loc_max[:k_segments-1]]
        boundaries = sorted(set([b for b in boundaries if 0 < b < n]))
        return boundaries
