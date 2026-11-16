"""LLM-based text segmentation by prompting.

The model sees the document as numbered sentences and returns the numbers of
the sentences that start a new section. Settings:

  mode='k'     the prompt states how many sections to produce (oracle k, as the
               other methods in main.py receive)
  mode='free'  the model decides how many sections there are
  shots        segmented example documents (few-shot), same language

Backends:
  'anthropic'  Claude via the Anthropic SDK (structured JSON output)
  'hf'         a local Hugging Face chat model (open-weight comparison)
  'mock'       no model: random boundaries, for dry runs and tests

Every answer is cached on disk (JSONL), so interrupted runs resume and reruns
are free. Usage (tokens, served model, latency) is stored with each answer.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are an expert annotator of document structure. You split documents "
    "into topical sections, the way an encyclopedia article is divided under "
    "its section headings. Section headings have been removed from the text."
)

# Approximate USD per 1M tokens (input, output), for cost reporting only.
PRICES = {
    'claude-opus-5-5': (4.00, 20.00),
    'claude-sonnet-5-5': (2.00, 10.00),
    'claude-haiku-5-5': (0.10, 0.50),
    'claude-fable-5-2': (8.00, 40.00),
}

BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "boundaries": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Numbers of the sentences that start a new section, ascending. Never include 1.",
        }
    },
    "required": ["boundaries"],
    "additionalProperties": False,
}


def number_sentences(sentences: Sequence[str]) -> str:
    return "\n".join(f"[{i + 1}] {s.replace(chr(10), ' ')}" for i, s in enumerate(sentences))


def bounds_to_numbers(bounds: Sequence[int]) -> List[int]:
    # A boundary b (0-based) means sentence b starts a new section -> number b+1.
    return [b + 1 for b in bounds]


def numbers_to_bounds(numbers: Sequence[int], n: int) -> List[int]:
    return sorted({int(x) - 1 for x in numbers if 1 < int(x) <= n})


class LLMSegmenter:
    def __init__(
        self,
        backend: str = 'anthropic',
        model: str = 'claude-opus-5-5',
        mode: str = 'k',
        language: str = 'Indonesian',
        shots: Optional[List] = None,          # list of SegmentDoc used as examples
        effort: Optional[str] = 'medium',      # Claude output_config.effort
        fallback: bool = True,                 # Claude server-side refusal fallback
        cache_path: Optional[str] = None,
        concurrency: int = 8,
        max_tokens: int = 16000,
        seed: int = 42,
        hf_4bit: bool = False,
        hf_max_new_tokens: int = 256,
    ):
        assert mode in ('k', 'free'), mode
        self.backend, self.model, self.mode = backend, model, mode
        self.language, self.shots = language, shots or []
        self.effort, self.fallback = effort, fallback
        self.concurrency, self.max_tokens, self.seed = concurrency, max_tokens, seed
        self.cache_path = cache_path
        self.hf_4bit, self.hf_max_new_tokens = hf_4bit, hf_max_new_tokens
        self._cache: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._client = None
        self._hf = None
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._cache[rec['key']] = rec
                    except (json.JSONDecodeError, KeyError):
                        continue

    # ------------------------------------------------------------------ prompt
    def _examples_block(self) -> str:
        if not self.shots:
            return ""
        parts = ["Here are segmented examples in the same language.\n"]
        for j, d in enumerate(self.shots, 1):
            ans = json.dumps({"boundaries": bounds_to_numbers(d.true_boundaries)})
            parts.append(f"Example {j} ({len(d.true_boundaries) + 1} sections):\n"
                         f"{number_sentences(d.sentences)}\nAnswer: {ans}\n")
        return "\n".join(parts)

    def _task_text(self, sentences: Sequence[str], k_segments: Optional[int]) -> str:
        n = len(sentences)
        if self.mode == 'k' and k_segments:
            how_many = (f"The document has exactly {k_segments} sections, so return exactly "
                        f"{k_segments - 1} sentence numbers.")
        else:
            how_many = "Decide how many sections the document has."
        return (
            f"The following {self.language} document has {n} numbered sentences. "
            f"Split it into contiguous topical sections. {how_many}\n"
            f"Return the numbers of the sentences that START a new section, in ascending order "
            f"(never sentence 1), as JSON: {{\"boundaries\": [...]}}.\n\n"
            f"Document:\n{number_sentences(sentences)}"
        )

    def _key(self, sentences: Sequence[str], k_segments: Optional[int]) -> str:
        h = hashlib.sha1()
        effort = str(self.effort) if self.backend == 'anthropic' else ('4bit' if self.hf_4bit else 'bf16')
        for part in (self.backend, self.model, self.mode, effort, SYSTEM_PROMPT,
                     self._examples_block(), self._task_text(sentences, k_segments)):
            h.update(part.encode('utf-8'))
            h.update(b'\x00')
        return h.hexdigest()

    # ---------------------------------------------------------------- backends
    def _call_anthropic(self, sentences, k_segments) -> dict:
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=6)
        system = [{"type": "text", "text": SYSTEM_PROMPT}]
        examples = self._examples_block()
        if examples:
            # Stable prefix (system + examples) is cached across documents.
            system.append({"type": "text", "text": examples, "cache_control": {"type": "ephemeral"}})
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": self._task_text(sentences, k_segments)}],
            output_config={"format": {"type": "json_schema", "schema": BOUNDARY_SCHEMA},
                           **({"effort": self.effort} if self.effort else {})},
        )
        t0 = time.time()
        if self.fallback:
            resp = self._client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = self._client.messages.create(**kwargs)
        latency = time.time() - t0
        text = next((b.text for b in resp.content if b.type == "text"), "")
        numbers, error = [], None
        if resp.stop_reason == "refusal":
            error = "refusal"
        else:
            try:
                numbers = json.loads(text)["boundaries"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                error = f"parse: {e}"
        u = resp.usage
        return {
            "numbers": numbers, "error": error, "stop_reason": resp.stop_reason,
            "served_model": resp.model, "latency_s": round(latency, 2),
            "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }

    def _load_hf(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model)
        kw = dict(device_map='auto')
        if self.hf_4bit:
            from transformers import BitsAndBytesConfig  # needs: pip install bitsandbytes
            kw['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4')
        else:
            kw['torch_dtype'] = torch.bfloat16
        mdl = AutoModelForCausalLM.from_pretrained(self.model, **kw)
        mdl.eval()
        return tok, mdl

    def _hf_prompt_ids(self, tok, sentences, k_segments):
        user = (self._examples_block() + "\n" + self._task_text(sentences, k_segments)).strip()
        variants = [
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            # Gemma-family templates reject a system role: fold it into the user turn.
            [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user}],
        ]
        last_err = None
        for msgs in variants:
            for extra in ({"enable_thinking": False}, {}):  # Qwen3: answer without a thinking phase
                try:
                    out = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                  return_tensors='pt', **extra)
                    return out["input_ids"] if hasattr(out, "keys") else out
                except Exception as e:  # template does not accept this shape
                    last_err = e
        raise last_err

    def _call_hf(self, sentences, k_segments) -> dict:
        import torch

        with self._lock:
            if self._hf is None:
                self._hf = self._load_hf()
            tok, mdl = self._hf
            ids = self._hf_prompt_ids(tok, sentences, k_segments).to(mdl.device)
            t0 = time.time()
            with torch.no_grad():  # one generation at a time on the GPU
                out = mdl.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=self.hf_max_new_tokens,
                                   do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
        numbers, error = [], None
        m = re.search(r'\{[^{}]*"boundaries"[^{}]*\}', text, flags=re.S)
        try:
            if m:
                numbers = [int(x) for x in json.loads(m.group(0))["boundaries"]]
            else:  # fall back to the integers in the answer
                numbers = [int(x) for x in re.findall(r'\d+', text)]
                error = "parse: no JSON (used bare integers)" if numbers else "parse: no answer"
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            error = f"parse: {e}"
        return {"numbers": numbers, "error": error, "served_model": self.model,
                "latency_s": round(time.time() - t0, 2), "raw_answer": text[:500],
                "input_tokens": int(ids.shape[1]), "output_tokens": int(out.shape[1] - ids.shape[1])}

    def _call_mock(self, sentences, k_segments) -> dict:
        n = len(sentences)
        rng = random.Random(hash((n, self.seed)))
        k = (k_segments or max(2, n // 8)) - 1
        nums = sorted(rng.sample(range(2, n + 1), min(k, n - 1))) if n > 1 else []
        prompt_chars = len(SYSTEM_PROMPT) + len(self._examples_block()) + len(self._task_text(sentences, k_segments))
        return {"numbers": nums, "error": None, "served_model": "mock", "latency_s": 0.0,
                "input_tokens": prompt_chars // 4, "output_tokens": 8 * (k + 1)}

    # -------------------------------------------------------------------- API
    def _get(self, sentences, k_segments) -> dict:
        key = self._key(sentences, k_segments)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        call = {'anthropic': self._call_anthropic, 'hf': self._call_hf, 'mock': self._call_mock}[self.backend]
        try:
            rec = call(sentences, k_segments)
        except Exception as e:  # keep the run going; record the failure
            rec = {"numbers": [], "error": f"{type(e).__name__}: {e}"}
            key = None  # do not cache transient failures
        rec['n_sentences'] = len(sentences)
        rec['k_given'] = k_segments if self.mode == 'k' else None
        if key:
            rec['key'] = key
            with self._lock:
                self._cache[key] = rec
                if self.cache_path:
                    os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
                    with open(self.cache_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def prefetch(self, docs: Sequence, desc: str = 'llm') -> None:
        """Query all documents concurrently so segment() later hits the cache."""
        todo = [d for d in docs if self._key(d.sentences, len(d.true_boundaries) + 1) not in self._cache]
        if not todo:
            return
        workers = 1 if self.backend == 'hf' else self.concurrency
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(self._get, d.sentences, len(d.true_boundaries) + 1) for d in todo]
            for _ in tqdm(as_completed(futs), total=len(futs), desc=f'{desc} queries'):
                pass

    def segment(self, sentences: Sequence[str], k_segments: Optional[int] = None) -> List[int]:
        rec = self._get(sentences, k_segments)
        return numbers_to_bounds(rec.get("numbers") or [], len(sentences))

    def usage_report(self, docs: Sequence) -> dict:
        recs = [self._cache.get(self._key(d.sentences, len(d.true_boundaries) + 1)) for d in docs]
        recs = [r for r in recs if r]
        if not recs:
            return {}
        tin = sum(r.get('input_tokens', 0) + r.get('cache_read_tokens', 0) + r.get('cache_write_tokens', 0) for r in recs)
        tout = sum(r.get('output_tokens', 0) for r in recs)
        price = PRICES.get(self.model)
        k_ok = [r for r in recs if r.get('k_given')]
        return {
            'n_answers': len(recs),
            'n_errors': sum(1 for r in recs if r.get('error')),
            'n_fallback_served': sum(1 for r in recs if r.get('served_model') not in (None, self.model, 'mock')),
            'k_exact_rate': (round(sum(1 for r in k_ok if len(set(r.get('numbers') or [])) == r['k_given'] - 1) / len(k_ok), 4)
                             if k_ok else None),
            'input_tokens': tin, 'output_tokens': tout,
            'est_cost_usd': round((tin * price[0] + tout * price[1]) / 1e6, 2) if price else None,
            'mean_latency_s': round(sum(r.get('latency_s', 0) for r in recs) / len(recs), 2),
        }
