from __future__ import annotations

from tqdm import tqdm

from ..data import append_hyp, read_hyps


class IndicConformerBackend:
    """AI4Bharat's 600M multilingual Conformer, loaded through its own
    remote code on the Hub. Clips go through one at a time because the
    wrapper takes a single waveform."""

    def __init__(self, model_id: str = "ai4bharat/indic-conformer-600m-multilingual", decoder: str = "ctc", device: str | None = None):
        import torch
        from transformers import AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        self.decoder = decoder  # "ctc" or "rnnt"

    def run(self, rows: list[dict], out_path: str) -> None:
        import soundfile as sf
        import torch

        done = read_hyps(out_path)
        for r in tqdm([r for r in rows if r["id"] not in done], desc=f"indicconformer-{self.decoder}"):
            wav, _ = sf.read(r["audio"], dtype="float32")
            wav = torch.from_numpy(wav).unsqueeze(0)
            with torch.inference_mode():
                text = self.model(wav, r["lang"], self.decoder)
            append_hyp(out_path, r["id"], str(text).strip())
