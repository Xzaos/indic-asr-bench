"""ASR backends. Each one takes manifest rows and appends {"id", "hyp"} lines
to a results file, skipping ids that are already there, so a crashed run
(or an exhausted API quota) resumes where it stopped."""
from __future__ import annotations


def get_backend(name: str, **kwargs):
    if name.startswith("whisper"):
        from .whisper_hf import WhisperBackend

        return WhisperBackend(**kwargs)
    if name == "indicconformer":
        from .indicconformer import IndicConformerBackend

        return IndicConformerBackend(**kwargs)
    if name.startswith("saaras") or name.startswith("saarika"):
        from .sarvam_api import SarvamBackend

        return SarvamBackend(**kwargs)
    raise ValueError(f"unknown backend {name}")
