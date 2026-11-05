# indic-asr-bench

How much of an Indic ASR model's word error rate is the model mishearing, and how much is the scorer disagreeing with a spelling? This repo scores Whisper, AI4Bharat's IndicConformer and Sarvam's Saaras on Hindi and Tamil with that question in mind.

Headline result: _ (fill in from results/report.md after the run)

## Why plain WER isn't enough here

Hindi has several correct ways to write the same word. "हँसना" and "हंसना" are both fine. "क़ानून" and "कानून" differ only in a nukta. A number can be "500", "५००" or "पांच सौ". And in code-mixed speech a model can reasonably write "mobile" where the reference says "मोबाइल". Standard WER counts all of these as errors, so two models with the same WER can be making very different mistakes.

Sarvam makes the same argument in their [ASR evaluation post](https://www.sarvam.ai/blogs/evaluating-indian-language-asr), and the [Voice of India](https://voiceofindiablogs.ai.joshtalks.com/) benchmark scores against multiple accepted spellings for the same reason. I wanted to measure the effect myself on public data, with error bars.

## What it computes

Every system is scored four ways on the same utterances:

| mode | what changes |
|---|---|
| raw | whitespace only |
| basic | NFC, punctuation removed, Latin lowercased |
| indic | basic, plus nukta folding, chandrabindu to anusvara, native digits to ASCII, zero-width characters removed |
| indic+translit | indic, plus a Latin word counts as correct if it matches the Devanagari reference word by consonant skeleton (`mobile` and `मोबाइल` both reduce to `mbl`) |

On top of that:

* Corpus WER and CER with 95% bootstrap intervals, resampling utterances.
* A paired bootstrap test between every system and the best one, so "model A beats model B" comes with a p-value instead of two overlapping numbers.
* Error buckets from the alignment: orthographic, script mix, number format, word-boundary split or merge, near miss, lexical, deletion, insertion. These show which part of a WER number a better normaliser would erase and which part needs a better model.
* LLM-WER: an LLM judges each aligned error pair as "same content" or "real error". The judge's verdicts are cached and inspectable.
* Judge validation. `label-export` samples 50 error pairs, stratified by bucket, into a CSV for hand labelling, and `label-score` reports agreement and Cohen's kappa against my labels. An LLM-WER number without this check is just another model's opinion.

Missing hypotheses (a timed-out API call, a crashed batch) are scored as all deletions, never dropped. Dropping them quietly makes a flaky system look better.

## Results

Test set: FLEURS test split, seeded sample of _ utterances per language. Full tables with every interval are in [results/report.md](results/report.md).

| system | hi WER basic | hi WER indic+translit | hi LLM-WER | ta WER basic | ta WER indic |
|---|---|---|---|---|---|
| whisper-large-v3 | _ | _ | _ | _ | _ |
| whisper-large-v3-turbo | _ | _ | _ | _ | _ |
| indicconformer (CTC) | _ | _ | _ | _ | _ |
| saaras:v3 | _ | _ | _ | _ | _ |
| saaras:v4 | _ | _ | _ | _ | _ |
| saaras:v4, codemix mode | _ | _ | _ | _ | _ |

The codemix row is the one to watch: that mode writes English words in Latin on purpose, so its basic WER should be much worse than its translit-aware WER. If the gap is small, the translit matcher is missing cases.

Judge agreement with my labels: _ / 50, kappa _.

What I read from these: _

## Running it

```bash
pip install -e ".[models,dev]"
pytest -q

export SARVAM_API_KEY=...          # for Saaras
export JUDGE_API_KEY=...           # for LLM-WER
export JUDGE_BASE_URL=https://api.sarvam.ai/v1   # any OpenAI-compatible endpoint
bash scripts/run_all.sh
```

Every step resumes from where it stopped, so a crash or an exhausted API quota just means running the same command again. Whisper and IndicConformer need a GPU for reasonable speed; a free Kaggle T4 is enough for 150 clips per language.

Individual steps:

```bash
asrbench prepare --dataset google/fleurs --config hi_in --split test --text-field raw_transcription --lang hi --n 150 --tag fleurs_hi
asrbench transcribe whisper --manifest data/fleurs_hi/manifest.jsonl --model-id openai/whisper-large-v3 --name whisper-large-v3
asrbench report --manifest data/fleurs_hi/manifest.jsonl
asrbench label-export --manifest data/fleurs_hi/manifest.jsonl --judge-model sarvam-m --judge-base-url https://api.sarvam.ai/v1
asrbench label-score results/judge_labels.csv
```

For your own clips (a Hinglish set, say), put the wavs in a folder with a `transcripts.tsv` of `filename<TAB>reference` and run `asrbench prepare --folder that_folder --lang hi --out data/mine.jsonl`.

## Design choices

Corpus-level WER, not the mean of per-utterance WER. A three-word utterance with one error would otherwise count as much as a thirty-word one.

CER is computed with spaces removed, so a word-boundary disagreement like "कर के" vs "करके" doesn't get charged at both the word and character level.

The alignment is a plain Levenshtein DP written out in `align.py` rather than jiwer, because the error buckets need the aligned pairs and the translit mode needs a custom match function.

The translit match is a heuristic. It romanises Devanagari with ITRANS, folds consonant pairs that English spelling and Hindi romanisation disagree on (ph/f, w/v, z/j), drops vowels and compares skeletons. Skeletons shorter than two consonants never match, which avoids most false positives on short words. It can still match two different words with the same skeleton, which is one reason the judge check exists.

Within a run of consecutive errors, the aligner's choice of which token is a deletion and which is a substitution is arbitrary. The bucketer looks at the run as two token lists, pulls out split/merge pairs first, and re-aligns the rest. Bucket op counts always sum to the alignment's error count, so LLM-WER can never exceed WER.

## Limitations

FLEURS is read speech. It says little about conversational or telephone audio, where the gap between these systems is likely larger.

The number-format bucket only detects that digits are involved; it doesn't verify that "500" and "पांच सौ" are the same number.

The translit matcher only handles Devanagari against Latin. Tamil code-mixing goes unmatched for now.

With 150 utterances per language the intervals are wide. The paired test is the fairer comparison at this size.
