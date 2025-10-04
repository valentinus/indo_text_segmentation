
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import math

from sklearn.metrics.pairwise import cosine_similarity

import torch

from transformers import AutoTokenizer, AutoModel

try:
    from bertopic import BERTopic
    _HAS_BERTOPIC = True
except Exception:
    print('BERTopic is not installed.')
    _HAS_BERTOPIC = False




def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p / (p.sum() + eps); q = q / (q.sum() + eps)
    m = 0.5*(p+q)
    def _kl(a,b):
        a = np.clip(a, eps, 1); b = np.clip(b, eps, 1)
        return float((a*np.log(a/b)).sum())
    return 0.5*_kl(p,m) + 0.5*_kl(q,m)


def entropy_coherence(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1)
    p = p / p.sum()
    H = -float((p * np.log(p)).sum())
    Hmax = math.log(len(p))
    return 1.0 - (H / Hmax if Hmax > 0 else 0.0)


def intra_window_concentration(T: np.ndarray) -> float:
    if T.shape[0] <= 1:
        return 0.0
    mu = T.mean(axis=0, keepdims=True)
    sims = cosine_similarity(T, mu).ravel()
    return float(np.mean(sims))


def cohshift_scores(T: np.ndarray, windows: List[int], metric: str = 'js', alpha: float = 1.0, gamma: float = 0.0) -> Tuple[np.ndarray, List[int]]:
    n = T.shape[0]
    if not windows:
        windows = [3]
    L = max(windows)
    idxs = list(range(L, n - L)) if n >= 2*L + 1 else []
    if not idxs:
        return np.array([]), []
    all_scores = []
    for i in idxs:
        per_w = []
        for w in windows:
            Lblock = T[i-w:i]
            Rblock = T[i:i+w]
            Lmean = Lblock.mean(axis=0)
            Rmean = Rblock.mean(axis=0)
            if metric == 'cos':
                d = 1.0 - float(cosine_similarity(Lmean.reshape(1,-1), Rmean.reshape(1,-1))[0,0])
            else:
                d = js_divergence(Lmean, Rmean)
            coh_l = entropy_coherence(Lmean)
            coh_r = entropy_coherence(Rmean)
            coh = math.sqrt(coh_l * coh_r)
            if gamma > 0.0:
                intra_l = intra_window_concentration(Lblock)
                intra_r = intra_window_concentration(Rblock)
                intra = math.sqrt(max(0.0, intra_l) * max(0.0, intra_r))
            else:
                intra = 1.0
            per_w.append(d * (coh ** alpha) * (intra ** gamma))
        all_scores.append(float(np.mean(per_w)))
    return np.asarray(all_scores, dtype=float), idxs


def pick_boundaries_from_scores(scores: np.ndarray, idxs: List[int], k_segments: int) -> List[int]:
    if scores.size == 0 or k_segments <= 1:
        return []
    loc = []
    for j in range(1, len(scores)-1):
        if scores[j] >= scores[j-1] and scores[j] >= scores[j+1]:
            loc.append((scores[j], idxs[j]))
    loc.sort(key=lambda x: x[0], reverse=True)
    bounds = [i for _, i in loc[:k_segments-1]]
    return sorted(set(bounds))

class SentenceEncoder:
    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-0.6B', device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: int = 64) -> np.ndarray:
        emb = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(self.device)
            out = self.model(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1)
            pooled = (out*mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            emb.append(pooled.cpu().numpy())
        return np.vstack(emb)


def bertopic_topic_vectors(sentences: List[str], encoder: SentenceEncoder, min_topic_size: int = 5) -> np.ndarray:
    if not _HAS_BERTOPIC:
        raise RuntimeError("BERTopic not installed. Try: pip install bertopic[hdbscan]")
    embeds = encoder.encode(sentences)
    model = BERTopic(min_topic_size=min_topic_size, calculate_probabilities=True, verbose=False)
    topics, probs = model.fit_transform(sentences, embeddings=embeds)
    if probs is None:
        K = max(topics)+1 if topics and max(topics) >= 0 else 1
        probs = np.zeros((len(sentences), K), dtype=float)
        for i, t in enumerate(topics):
            probs[i, max(0,t)] = 1.0
    return np.asarray(probs, dtype=float)
