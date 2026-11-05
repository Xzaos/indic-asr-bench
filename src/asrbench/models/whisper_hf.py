from __future__ import annotations

from tqdm import tqdm

from ..data import append_hyp, iter_batches, read_hyps

# Whisper's language tokens use full names in the HF generate() API.
WHISPER_LANG = {"hi": "hindi", "ta": "tamil", "te": "telugu", "bn": "bengali", "mr": "marathi", "en": "english"}


class WhisperBackend:
    def __init__(self, model_id: str = "openai/whisper-large-v3", batch_size: int = 16, device: str | None = None, peft_adapter: str | None = None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()
        if peft_adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, peft_adapter).merge_and_unload().eval()
        self.dtype = dtype
        self.batch_size = batch_size

    def run(self, rows: list[dict], out_path: str) -> None:
        import soundfile as sf
        import torch

        done = read_hyps(out_path)
        todo = [r for r in rows if r["id"] not in done]
        for batch in tqdm(list(iter_batches(todo, self.batch_size)), desc="whisper"):
            audio = [sf.read(r["audio"], dtype="float32")[0] for r in batch]
            # All test clips are under 30 s; anything longer is truncated by
            # the feature extractor, so check durations when adding a dataset.
            feats = self.processor(audio, sampling_rate=16_000, return_tensors="pt").input_features
            lang = WHISPER_LANG[batch[0]["lang"]]
            with torch.inference_mode():
                ids = self.model.generate(
                    feats.to(self.device, self.dtype),
                    language=lang,
                    task="transcribe",
                    num_beams=1,
                    max_new_tokens=225,
                )
            texts = self.processor.batch_decode(ids, skip_special_tokens=True)
            for r, t in zip(batch, texts):
                append_hyp(out_path, r["id"], t.strip())
