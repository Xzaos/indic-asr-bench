"""Sarvam speech-to-text over REST.

Requests run concurrently under a semaphore, with exponential backoff on
429 and 5xx. Check the current parameter names at docs.sarvam.ai before a
real run; the model string and mode are CLI flags for that reason.
"""
from __future__ import annotations

import asyncio
import os
import random

from ..data import append_hyp, read_hyps

URL = "https://api.sarvam.ai/speech-to-text"
LANG_CODES = {"hi": "hi-IN", "ta": "ta-IN", "te": "te-IN", "bn": "bn-IN", "mr": "mr-IN", "en": "en-IN"}


class SarvamBackend:
    # mode is one of transcribe, translate, verbatim, translit, codemix
    # (saaras:v3 and v4). Clips must be under 30 s for the REST endpoint.
    def __init__(self, model_id: str = "saaras:v4", mode: str | None = "transcribe", concurrency: int = 4, max_retries: int = 5):
        self.key = os.environ.get("SARVAM_API_KEY")
        if not self.key:
            raise RuntimeError("set SARVAM_API_KEY")
        self.model_id, self.mode = model_id, mode
        self.concurrency, self.max_retries = concurrency, max_retries

    async def _one(self, client, sem, row, out_path, lock):
        import httpx

        data = {"model": self.model_id, "language_code": LANG_CODES[row["lang"]]}
        if self.mode:
            data["mode"] = self.mode
        async with sem:
            for attempt in range(self.max_retries):
                with open(row["audio"], "rb") as f:
                    files = {"file": (os.path.basename(row["audio"]), f.read(), "audio/wav")}
                try:
                    resp = await client.post(URL, data=data, files=files, headers={"api-subscription-key": self.key})
                except httpx.TransportError:
                    resp = None
                if resp is not None and resp.status_code == 200:
                    body = resp.json()
                    async with lock:
                        append_hyp(out_path, row["id"], body.get("transcript", "").strip(), request_id=body.get("request_id"))
                    return
                if resp is not None and resp.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"{row['id']}: HTTP {resp.status_code} {resp.text[:200]}")
                await asyncio.sleep(2**attempt + random.random())
            raise RuntimeError(f"{row['id']}: gave up after {self.max_retries} tries")

    async def _run(self, rows, out_path):
        import httpx

        sem, lock = asyncio.Semaphore(self.concurrency), asyncio.Lock()
        async with httpx.AsyncClient(timeout=60) as client:
            results = await asyncio.gather(
                *(self._one(client, sem, r, out_path, lock) for r in rows), return_exceptions=True
            )
        failed = [e for e in results if isinstance(e, Exception)]
        for e in failed[:5]:
            print("failed:", e)
        if failed:
            print(f"{len(failed)} requests failed; rerun the same command to retry them")

    def run(self, rows: list[dict], out_path: str) -> None:
        done = read_hyps(out_path)
        asyncio.run(self._run([r for r in rows if r["id"] not in done], out_path))
