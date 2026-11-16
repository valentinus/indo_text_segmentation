from __future__ import annotations

import re

from bs4 import BeautifulSoup

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


import spacy

from languages import get_language

# The rule-based sentencizer does not need a GPU; prefer_gpu() does not fail
# on CPU-only machines (require_gpu() did).
spacy.prefer_gpu()

# Indonesian tokenizer rules work well for the Latin-script regional languages.
# Add Balinese (U+1B5E/U+1B5F) and Buginese/Lontara (U+1A1E/U+1A1F) full stops.
_EXTRA_PUNCT = ['\u1b5e', '\u1b5f', '\u1a1e', '\u1a1f']
nlp = spacy.blank("id")
_sentencizer = nlp.add_pipe("sentencizer")
_sentencizer.punct_chars = set(_sentencizer.punct_chars) | set(_EXTRA_PUNCT)

    
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



# Corpora extracted with wikiextractor 3.1.0 --html keep wiki links as escaped
# <a href="...">text</a>; keep only the anchor text. (No-op for the Indonesian corpus.)
_LINK_TAG = re.compile(r'</?a\b[^>]*>')


def strip_link_markup(text: str) -> str:
    return _LINK_TAG.sub('', text) if '<a' in text or '</a' in text else text


def spacy_sent_tokenize(text: List[str]):
    doc = nlp(text)
    return [token.text for token in doc], [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def parse_headings_and_content(
    file_path: str,
    lang: str = 'id',
    skip_headings: Optional[Iterable[str]] = None,
    case_sensitive_headings: Optional[bool] = None,
    min_headings: int = 3,
    min_boundaries: int = 3,
    min_sentences: int = 11,
    strip_heading_echo: bool = False,
):
    """Parse a wikiextractor (--html) file into SegmentDocs.

    Gold boundaries are the <h2>/<h3> section starts. Defaults reproduce the
    original Indonesian setup.

    lang:               wiki code (id, jv, su, ...) -> boilerplate headings to skip
    skip_headings:      override the language's boilerplate-heading list
    case_sensitive_headings: override the language's matching rule
                        (Indonesian: exact match, others: case-insensitive)
    strip_heading_echo: wikiextractor repeats each heading as a sentence
                        ("Sajarah.") at the start of its section; drop it so
                        the heading text does not leak into the input.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        html = f.read()

    language = get_language(lang)
    if skip_headings is None:
        skip_headings = language.skip_headings
    if case_sensitive_headings is None:
        case_sensitive_headings = language.case_sensitive_headings
    norm = (lambda h: h.strip()) if case_sensitive_headings else (lambda h: h.strip().casefold())
    skip = {norm(h) for h in skip_headings}

    soup = BeautifulSoup(html, 'html.parser')

    all_result = []
    docs = soup.find_all('doc')
    for doc in docs:
        segment, segment_length, segment_count = [], [], 0
        token_count = 0
        headings = doc.find_all(['h2', 'h3'])
        if len(headings) < min_headings:
            continue
        echoes = {h.get_text(strip=True) + '.' for h in doc.find_all(['h2', 'h3', 'h4'])}

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
            if norm(heading_text) in skip:
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

            content = strip_link_markup("\n\n".join(content_lines)).strip()
            if strip_heading_echo:
                lines = content.split('\n')
                while lines and lines[0].strip() in echoes:
                    lines.pop(0)
                content = '\n'.join(lines).strip()
            if content == "":
                continue

            tokens, content = spacy_sent_tokenize(content)
            if len(content) > 1:
                token_count += len(tokens)
                segment_count += len(content)
                segment_length.append(len(content))
                segment.append(segment_count)
                content_temp.extend(content)

        if len(segment[:-1]) >= min_boundaries and len(content_temp) >= min_sentences:
            all_result.append(SegmentDoc(sentences=content_temp, true_boundaries=segment[:-1],
                                         token_count=token_count, segment_length=segment_length,
                                         segment_type='wiki_indo' if lang == 'id' else f'wiki_{lang}'))

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

