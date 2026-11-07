# indic-asr-bench

How much of an Indic ASR model's word error rate is the model mishearing, and how much is the scorer disagreeing with a spelling? This repo scores Whisper, AI4Bharat's IndicConformer and Sarvam's Saaras on Hindi and Tamil with that question in mind.

Headline result: saaras-v4 has the lowest error on both languages. The table is under Results.

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

Test set: FLEURS test split, a seeded sample of 150 Hindi and 148 Tamil utterances. WER and CER are in %, with 95% bootstrap intervals over utterances. WER is indic+translit. CER uses indic normalisation, which is the CER column the report gives. Full tables with every mode are in [results/report.md](results/report.md).

Hindi (3417 reference words):

| system | WER | CER |
|---|---|---|
| saaras-v4 | 5.52 [4.55, 6.51] | 2.73 [2.10, 3.42] |
| saaras-v4-codemix | 7.61 [6.41, 8.87] | 15.67 [13.25, 18.21] |
| indicconformer-ctc | 9.67 [8.14, 11.24] | 4.77 [3.80, 5.78] |
| whisper-large-v3 | 26.51 [23.74, 29.97] | 12.96 [10.81, 15.54] |
| whisper-turbo | 28.63 [26.43, 31.02] | 13.69 [11.84, 15.69] |

Tamil (2426 reference words):

| system | WER | CER |
|---|---|---|
| saaras-v4 | 21.66 [18.75, 24.63] | 6.58 [4.49, 8.93] |
| indicconformer-ctc | 27.81 [25.07, 30.74] | 9.03 [7.21, 11.17] |
| whisper-large-v3 | 54.01 [50.29, 58.00] | 19.40 [15.90, 23.41] |
| whisper-turbo | 71.38 [64.79, 81.75] | 23.16 [19.98, 27.15] |

saaras-v4 is best on both languages, with paired bootstrap p<0.001 against every other system.

The codemix CER is high because indic normalisation has no translit matching, so English words written in Latin count as character errors. The WER column is the fairer one for that row.

### Why normalisation matters

Raw WER mixes model errors with spelling and formatting choices. For indicconformer-ctc on Hindi, raw WER is 21.69 and indic WER is 9.73. More than half of its raw errors are punctuation, nukta, digit and similar differences, not misheard words.

The saaras-v4-codemix case shows the translit match at work. That mode writes English words in Latin on purpose, and the bucketer counts 165 script_mix errors for it, against 3 for saaras-v4. Indic WER is 12.40. With the translit match it drops to 7.61. Plain WER penalises output that is correct, just in a different script from the reference.

### LLM-WER and how far to trust it

For each substitution pair from the alignment, the judge (sarvam-105b) decides whether the two tokens carry the same spoken content or are a real error. Same content is not counted. Hindi LLM-WER ranges from 2.38 for saaras-v4 to 15.45 for whisper-turbo (full table in the report).

I checked the judge against my own labels on 50 Hindi pairs, stratified by error bucket. I was the only annotator and the judge's verdicts were hidden while I labelled. Raw output is in [results/judge_validation.txt](results/judge_validation.txt) and the pairs are in [results/judge_labels.csv](results/judge_labels.csv).

* Agreement is 0.80 and Cohen's kappa is 0.58.
* The judge disagreed with me 10 times. It was lenient 6 times (it said same content where I said real error) and strict 4 times (the reverse).
* split_merge and script_mix were 6/6 each. number_format was 3/6.

The number weakness has a clear cause. A spelled-out number gets split across several token pairs in the alignment, and each pair makes no sense alone. The judge sees "400" against "सौ" and has no way to know the rest of the phrase. A judge that saw the whole span would likely do better here, but I haven't tried it.

Because the judge is more often lenient than strict, LLM-WER is probably optimistic. It is also validated on Hindi only.

### Where models fail

Whisper's errors are mostly near_miss and lexical. On Hindi, whisper-large-v3 has 404 near_miss and 320 lexical errors, against 60 orthographic and 12 script_mix. The gap to the other systems is acoustic, not formatting, and no normaliser will close it.

Tamil has high split_merge counts for every system (96 for indicconformer-ctc, 114 for saaras-v4, against 20 to 34 on Hindi). Tamil is agglutinative, so a word boundary is often a judgement call. CER is the fairer metric there.

The wide Tamil interval for whisper-turbo (64.79 to 81.75) comes from a runaway output on fleurs_ta_000487. The hypothesis is a long run of repeated digits instead of a transcript.

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
asrbench label-export --manifest data/fleurs_hi/manifest.jsonl --judge-model sarvam-105b --judge-base-url https://api.sarvam.ai/v1
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

Whisper truncates audio at 30 s and the Sarvam REST API rejects anything longer, so clips over 30 s are dropped from the manifest (`asrbench filter`, or `--max-sec` on `prepare`) and are excluded for every system.

FLEURS is read speech. It says little about conversational or telephone audio, where the gap between these systems is likely larger.

The test set is 150 Hindi and 148 Tamil utterances. The judge validation rests on one annotator (me) and 50 pairs. There is no LLM-WER for Tamil, and the judge check covers Hindi only.

The number-format bucket only detects that digits are involved; it doesn't verify that "500" and "पांच सौ" are the same number.

The translit matcher only handles Devanagari against Latin. Tamil code-mixing goes unmatched for now.

With 150 utterances per language the intervals are wide. The paired test is the fairer comparison at this size.
