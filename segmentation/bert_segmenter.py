from __future__ import annotations
from typing import List, Optional

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import AutoTokenizer, AutoModel
class BertSegmenter:
    def __init__(self, model_name: str = 'LazarusNLP/congen-indobert-base', device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: int = 256) -> np.ndarray:
        reps = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(self.device)
            out = self.model(**enc)
            last_hidden = out.last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts
            reps.append(mean_pooled.cpu().numpy())
        return np.vstack(reps)

    def segment(self, sentences: List[str], k_segments: Optional[int] = None, window: int = 3) -> List[int]:
        n = len(sentences)
        if n < 2*window + 1:
            return []
        X = self.encode(sentences)
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
            thr = sims.mean() - 0.5 * sims.std()
            boundaries = [idx for val, idx in loc_min if val <= thr]
        else:
            boundaries = [idx for _, idx in loc_min[:k_segments-1]]
        boundaries = sorted(set([b for b in boundaries if 0 < b < n]))
        return boundaries

