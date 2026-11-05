"""LLM-WER: ask a language model whether each word error changes meaning.

For every utterance with errors, the judge sees the reference, the
hypothesis and the list of aligned error pairs, and returns one verdict per
pair: same content (spelling, script, number format) or a real error.
LLM-WER = (errors judged real) / (reference words).

Verdicts are cached on disk, keyed by a hash of the prompt, so rerunning the
report costs nothing and the exact verdicts behind a number can be inspected.

Any OpenAI-compatible chat endpoint works. Sarvam's own chat API, a local
vLLM server and OpenAI all fit.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from .data import read_manifest
from .failures import classify
from .metrics import UttScore

PROMPT_VERSION = "v1"

SYSTEM = (
    "You check speech recognition errors in Indian languages. For each pair, "
    "decide whether the hypothesis token(s) express the same spoken content as "
    "the reference token(s). Count as SAME: spelling variants, nukta or "
    "chandrabindu differences, the same word in Latin vs native script, the same "
    "number written as digits or words, and word-boundary differences. Count as "
    "DIFFERENT: any other word, a missing word, or an extra word. "
    'Reply with JSON only: {"verdicts": [{"i": 1, "same": true}, ...]}'
)


def _pairs(score: UttScore) -> list[dict]:
    out = []
    for bucket, ref, hyp, n_ops in classify(score.alignment):
        out.append({"bucket": bucket, "ref": ref or "", "hyp": hyp or "", "ops": n_ops})
    return out


def _prompt(ref: str, hyp: str, pairs: list[dict]) -> str:
    lines = [f"Reference: {ref}", f"Hypothesis: {hyp}", "Pairs:"]
    for i, p in enumerate(pairs, 1):
        lines.append(f"{i}. ref=[{p['ref'] or '<none>'}] hyp=[{p['hyp'] or '<none>'}]")
    return "\n".join(lines)


class Judge:
    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "JUDGE_API_KEY", cache: str = "results/judge_cache.jsonl"):
        self.model = model
        self.base_url = (base_url or os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.key = os.environ.get(api_key_env, "")
        self.cache_path = Path(cache)
        self.cache: dict[str, list[bool]] = {}
        if self.cache_path.exists():
            for row in read_manifest(self.cache_path):
                self.cache[row["key"]] = row["same"]

    def _key(self, prompt: str) -> str:
        return hashlib.sha1(f"{PROMPT_VERSION}|{self.model}|{prompt}".encode()).hexdigest()

    def _call(self, prompt: str, n: int) -> list[bool]:
        import httpx

        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        }
        r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers={"Authorization": f"Bearer {self.key}"}, timeout=120)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        verdicts = json.loads(m.group(0))["verdicts"] if m else []
        same = [False] * n
        for v in verdicts:
            i = int(v.get("i", 0)) - 1
            if 0 <= i < n:
                same[i] = bool(v.get("same"))
        # Pairs the judge skipped stay False, i.e. counted as real errors.
        return same

    def judge(self, ref: str, hyp: str, score: UttScore) -> tuple[list[dict], list[bool]]:
        pairs = _pairs(score)
        if not pairs:
            return pairs, []
        prompt = _prompt(ref, hyp, pairs)
        key = self._key(prompt)
        if key not in self.cache:
            self.cache[key] = self._call(prompt, len(pairs))
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "same": self.cache[key]}) + "\n")
        return pairs, self.cache[key]


def llm_errors(pairs: list[dict], same: list[bool]) -> int:
    return sum(p["ops"] for p, s in zip(pairs, same) if not s)
