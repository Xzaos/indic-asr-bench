# Results

Manifest: `data/test.jsonl`, 298 utterances.

## hi (150 utterances, 3417 reference words)

| system | WER raw | WER basic | WER indic | WER indic+translit | CER indic | LLM-WER |
|---|---|---|---|---|---|---|
| indicconformer-ctc | 21.69 [19.86, 23.48] | 12.08 [10.43, 13.71] | 9.73 [8.19, 11.32] | 9.67 [8.14, 11.24] | 4.77 [3.80, 5.78] | 4.79 [3.60, 6.06] |
| saaras-v4-codemix | 21.63 [19.81, 23.53] | 14.23 [12.56, 15.96] | 12.40 [10.74, 14.12] | 7.61 [6.41, 8.87] | 15.67 [13.25, 18.21] | 3.05 [2.26, 4.01] |
| saaras-v4 | 15.22 [13.69, 16.72] | 7.96 [6.83, 9.13] | 5.60 [4.64, 6.60] | 5.52 [4.55, 6.51] | 2.73 [2.10, 3.42] | 2.38 [1.64, 3.19] |
| whisper-large-v3 | 36.35 [33.47, 39.81] | 28.66 [25.86, 32.10] | 26.86 [24.06, 30.29] | 26.51 [23.74, 29.97] | 12.96 [10.81, 15.54] | 14.49 [11.69, 17.93] |
| whisper-turbo | 38.86 [36.64, 41.28] | 30.89 [28.76, 33.27] | 29.15 [26.92, 31.55] | 28.63 [26.43, 31.02] | 13.69 [11.84, 15.69] | 15.45 [13.39, 17.79] |

WER in %, 95% bootstrap interval over utterances in brackets.

Paired bootstrap vs `saaras-v4` (indic+translit WER):

| system | ΔWER | p |
|---|---|---|
| indicconformer-ctc | 4.15 [2.94, 5.48] | 0.000 |
| saaras-v4-codemix | 2.09 [1.25, 3.07] | 0.000 |
| whisper-large-v3 | 20.99 [18.25, 24.35] | 0.000 |
| whisper-turbo | 23.11 [20.99, 25.37] | 0.000 |

Error buckets (basic normalisation, word ops):

| system | orthographic | script_mix | number_format | split_merge | near_miss | lexical | deletion | insertion |
|---|---|---|---|---|---|---|---|---|
| indicconformer-ctc | 78 | 2 | 39 | 32 | 128 | 84 | 20 | 33 |
| saaras-v4-codemix | 63 | 165 | 26 | 20 | 38 | 143 | 13 | 22 |
| saaras-v4 | 81 | 3 | 22 | 22 | 69 | 50 | 14 | 13 |
| whisper-large-v3 | 60 | 12 | 16 | 32 | 404 | 320 | 59 | 84 |
| whisper-turbo | 57 | 18 | 15 | 34 | 410 | 394 | 53 | 83 |

## ta (148 utterances, 2426 reference words)

| system | WER raw | WER basic | WER indic | WER indic+translit | CER indic | LLM-WER |
|---|---|---|---|---|---|---|
| indicconformer-ctc | 41.14 [38.38, 44.19] | 27.85 [25.11, 30.78] | 27.81 [25.07, 30.74] | 27.81 [25.07, 30.74] | 9.03 [7.21, 11.17] | n/a |
| saaras-v4 | 29.31 [26.25, 32.43] | 21.70 [18.79, 24.67] | 21.66 [18.75, 24.63] | 21.66 [18.75, 24.63] | 6.58 [4.49, 8.93] | n/a |
| whisper-large-v3 | 60.51 [56.79, 64.51] | 54.14 [50.31, 58.18] | 54.14 [50.31, 58.18] | 54.01 [50.29, 58.00] | 19.40 [15.90, 23.41] | n/a |
| whisper-turbo | 73.95 [71.11, 76.76] | 71.42 [64.79, 81.79] | 71.42 [64.79, 81.79] | 71.38 [64.79, 81.75] | 23.16 [19.98, 27.15] | n/a |

WER in %, 95% bootstrap interval over utterances in brackets.

Paired bootstrap vs `saaras-v4` (indic+translit WER):

| system | ΔWER | p |
|---|---|---|
| indicconformer-ctc | 6.14 [4.24, 8.02] | 0.000 |
| whisper-large-v3 | 32.35 [28.55, 36.00] | 0.000 |
| whisper-turbo | 49.71 [42.67, 60.87] | 0.000 |

Error buckets (basic normalisation, word ops):

| system | orthographic | script_mix | number_format | split_merge | near_miss | lexical | deletion | insertion |
|---|---|---|---|---|---|---|---|---|
| indicconformer-ctc | 1 | 0 | 39 | 96 | 263 | 142 | 48 | 91 |
| saaras-v4 | 1 | 0 | 18 | 114 | 205 | 79 | 36 | 77 |
| whisper-large-v3 | 0 | 2 | 21 | 72 | 537 | 476 | 90 | 124 |
| whisper-turbo | 0 | 1 | 60 | 70 | 647 | 614 | 145 | 207 |
