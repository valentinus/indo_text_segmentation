from __future__ import annotations

from bs4 import BeautifulSoup

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


import spacy

print(spacy.require_gpu())

nlp = spacy.blank("id")
nlp.add_pipe("sentencizer")

    
@dataclass
class SegmentDoc:
    sentences: List[str]
    true_boundaries: List[int]
    token_count: int
    segment_length: List[int]
    segment_type: List[str]
    
@dataclass
class SegmentResult:
    sentences: List[str]
    true_boundaries: List[int]
    pred_boundaries: List[int]
    pk_scores: float
    windiff_scores: float



def spacy_sent_tokenize(text: List[str]):
    doc = nlp(text)
    return [token.text for token in doc], [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def parse_headings_and_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')

    all_result = []
    docs = soup.find_all('doc')
    for doc in docs:
        segment, segment_length, segment_count = [], [], 0
        token_count = 0
        headings = doc.find_all(['h2', 'h3'])
        if len(headings) < 3:
            continue
        
        content_temp = list()
        if len(content_temp) > 1:
            tokens, content_temp = spacy_sent_tokenize(str(headings[0].previous_sibling).strip())
            token_count += len(tokens)
            segment_count += len(content_temp)
            segment_length.append(len(content_temp))
            segment.append(segment_count)
        for i, heading in enumerate(headings):
            tag_name = heading.name
            heading_text = heading.get_text(strip=True)
            if heading_text == 'Catatan' or heading_text == 'Referensi' or heading_text == 'Lihat pula' or heading_text == 'Pranala luar' or heading_text == 'Bahan bacaan terkait':
                continue
            content_lines = []
            for sibling in heading.next_siblings:
                if isinstance(sibling, type(heading)) and sibling.name in ('h2', 'h3'):
                    break
                text = ''
                if hasattr(sibling, 'get_text'):
                    text = sibling.get_text(separator=' ', strip=True)
                else:
                    text = str(sibling).strip()
                if text:
                    content_lines.append(text)

            content = "\n\n".join(content_lines).strip()
            if content == "":
                continue
            
            tokens, content = spacy_sent_tokenize(content)
            if len(content) > 1:
                token_count += len(tokens)
                segment_count += len(content)
                segment_length.append(len(content))
                segment.append(segment_count)
                content_temp.extend(content)
            
        if len(segment[:-1]) > 2 and len(content_temp) > 10:
            all_result.append(SegmentDoc(sentences=content_temp, true_boundaries=segment[:-1], 
                                         token_count=token_count, segment_length=segment_length,
                                         segment_type='wiki_indo'))

    return all_result

    
def boundaries_to_segmentation_vector(n_sents: int, boundaries: List[int]) -> List[int]:
    seg = [0]*n_sents
    for b in sorted(boundaries):
        for i in range(b, n_sents):
            seg[i] += 1
    return seg


def pk_metric(true_bounds: List[int], pred_bounds: List[int], n_sents: int) -> float:
    true_seg = boundaries_to_segmentation_vector(n_sents, true_bounds)
    pred_seg = boundaries_to_segmentation_vector(n_sents, pred_bounds)

    all_bounds = [0] + sorted(true_bounds) + [n_sents]
    seg_lengths = [all_bounds[i+1]-all_bounds[i] for i in range(len(all_bounds)-1)]
    avg_len = max(1, int(round(sum(seg_lengths)/len(seg_lengths))))
    k = max(1, avg_len // 2)

    errors = 0
    total = 0
    for i in range(n_sents - k):
        same_true = (true_seg[i] == true_seg[i+k])
        same_pred = (pred_seg[i] == pred_seg[i+k])
        errors += (same_true != same_pred)
        total += 1
    return errors / max(1, total)


def windowdiff(true_bounds: List[int], pred_bounds: List[int], n_sents: int, k: Optional[int] = None) -> float:
    if k is None:
        all_bounds = [0] + sorted(true_bounds) + [n_sents]
        seg_lengths = [all_bounds[i+1]-all_bounds[i] for i in range(len(all_bounds)-1)]
        avg_len = max(1, int(round(sum(seg_lengths)/len(seg_lengths))))
        k = max(1, avg_len // 2)

    true_marks = np.zeros(n_sents, dtype=int)
    pred_marks = np.zeros(n_sents, dtype=int)
    for b in true_bounds:
        if 0 <= b < n_sents:
            true_marks[b] = 1
    for b in pred_bounds:
        if 0 <= b < n_sents:
            pred_marks[b] = 1

    wd = 0
    total = 0
    for i in range(n_sents - k):
        wd += abs(true_marks[i:i+k].sum() - pred_marks[i:i+k].sum()) > 0
        total += 1
    return wd / max(1, total)

