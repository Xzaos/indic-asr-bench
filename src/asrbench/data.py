"""Manifests: one JSON object per line with id, audio (path to a 16 kHz wav),
text (reference) and lang (ISO code like "hi").

Everything downstream only reads manifests, so a new test set means writing
one prepare function and nothing else.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterator

SR = 16_000


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_hyps(path: str | Path) -> dict[str, str]:
    if not Path(path).exists():
        return {}
    return {r["id"]: r["hyp"] for r in read_manifest(path)}


def append_hyp(path: str | Path, uid: str, hyp: str, **extra) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": uid, "hyp": hyp, **extra}, ensure_ascii=False) + "\n")


def _save_wav(path: Path, array, sr: int) -> None:
    import numpy as np
    import soundfile as sf

    audio = np.asarray(array, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import torch
        import torchaudio.functional as AF

        audio = AF.resample(torch.from_numpy(audio), sr, SR).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SR)


def prepare_hf(
    dataset: str,
    config: str | None,
    split: str,
    lang: str,
    text_field: str,
    out_dir: str,
    n: int,
    seed: int = 0,
    tag: str | None = None,
) -> Path:
    """Sample n utterances from a Hugging Face audio dataset into wavs + a
    manifest. Sampling is seeded so the test set is reproducible from the
    command line alone."""
    from datasets import Audio, load_dataset

    # FLEURS still ships as a loading script, which needs datasets<4 and
    # trust_remote_code. Parquet datasets ignore the flag.
    ds = load_dataset(dataset, config, split=split, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(len(ds)), min(n, len(ds))))

    tag = tag or f"{dataset.split('/')[-1]}_{lang}"
    root = Path(out_dir) / tag
    rows = []
    for k in picks:
        ex = ds[k]
        uid = f"{tag}_{k:06d}"
        wav = root / "wav" / f"{uid}.wav"
        _save_wav(wav, ex["audio"]["array"], ex["audio"]["sampling_rate"])
        rows.append({"id": uid, "audio": str(wav), "text": ex[text_field], "lang": lang})
    manifest = root / "manifest.jsonl"
    write_jsonl(manifest, rows)
    return manifest


def prepare_folder(folder: str, lang: str, out: str, tsv: str = "transcripts.tsv") -> Path:
    """For hand-collected sets (e.g. Hinglish clips): a folder of wavs plus a
    tab-separated file of filename<TAB>reference."""
    folder_p = Path(folder)
    rows = []
    with open(folder_p / tsv, encoding="utf-8") as f:
        for name, text in csv.reader(f, delimiter="\t"):
            rows.append({"id": Path(name).stem, "audio": str(folder_p / name), "text": text, "lang": lang})
    write_jsonl(out, rows)
    return Path(out)


def iter_batches(rows: list[dict], size: int) -> Iterator[list[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]
