#!/usr/bin/env bash
# Full benchmark run. Needs a GPU for Whisper/IndicConformer (a Kaggle T4 is
# fine), SARVAM_API_KEY for Saaras, and JUDGE_API_KEY + JUDGE_BASE_URL for
# LLM-WER. Every step resumes, so rerun after a crash.
set -euo pipefail

N=${N:-150}

# Test sets. FLEURS test split, seeded sample. Swap in Kathbath or your own
# folder with `asrbench prepare --folder`.
asrbench prepare --dataset google/fleurs --config hi_in --split test \
  --text-field raw_transcription --lang hi --n "$N" --tag fleurs_hi
asrbench prepare --dataset google/fleurs --config ta_in --split test \
  --text-field raw_transcription --lang ta --n "$N" --tag fleurs_ta
cat data/fleurs_hi/manifest.jsonl data/fleurs_ta/manifest.jsonl > data/test.jsonl

asrbench transcribe whisper --manifest data/test.jsonl --model-id openai/whisper-large-v3 --name whisper-large-v3
asrbench transcribe whisper --manifest data/test.jsonl --model-id openai/whisper-large-v3-turbo --name whisper-turbo
asrbench transcribe indicconformer --manifest data/test.jsonl --decoder ctc --name indicconformer-ctc
asrbench transcribe saaras:v3 --manifest data/test.jsonl --mode transcribe --name saaras-v3
asrbench transcribe saaras:v4 --manifest data/test.jsonl --mode transcribe --name saaras-v4
# codemix writes English words in Latin: the case the translit score is for
asrbench transcribe saaras:v4 --manifest data/test.jsonl --mode codemix --name saaras-v4-codemix

asrbench report --manifest data/test.jsonl \
  --judge-model "${JUDGE_MODEL:-sarvam-m}" --judge-base-url "${JUDGE_BASE_URL:-https://api.sarvam.ai/v1}"

# Judge check: label the 50 sampled pairs by hand, then score.
asrbench label-export --manifest data/test.jsonl --judge-model "${JUDGE_MODEL:-sarvam-m}" \
  --judge-base-url "${JUDGE_BASE_URL:-https://api.sarvam.ai/v1}" --n 50
echo "now fill results/judge_labels.csv and run: asrbench label-score results/judge_labels.csv"
