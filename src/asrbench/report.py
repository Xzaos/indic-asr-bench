"""Build results/report.md from a manifest and a folder of hypothesis files.

One table per language: WER under each scoring mode with 95% bootstrap CIs,
CER, LLM-WER if a judge is given (only for the languages asked for), error buckets, and a paired test of
every system against the best one.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .data import read_hyps, read_manifest
from .failures import bucket_counts
from .metrics import SCORING_MODES, score_all
from .stats import bootstrap_ci, paired_bootstrap

BUCKETS = ["orthographic", "script_mix", "number_format", "split_merge", "near_miss", "lexical", "deletion", "insertion"]


def _systems(results_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(results_dir.glob("hyp_*.jsonl"))}


def build(manifest: str, results_dir: str = "results", judge=None, n_boot: int = 2000, judge_langs=None) -> str:
    rows = read_manifest(manifest)
    rdir = Path(results_dir)
    systems = _systems(rdir)
    if not systems:
        raise SystemExit(f"no hyp_*.jsonl files in {rdir}")

    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)

    out = [f"# Results\n\nManifest: `{manifest}`, {len(rows)} utterances.\n"]
    for lang, lrows in sorted(by_lang.items()):
        refs = {r["id"]: r["text"] for r in lrows}
        out.append(f"## {lang} ({len(lrows)} utterances, {sum(len(t.split()) for t in refs.values())} reference words)\n")
        header = "| system | " + " | ".join(f"WER {m}" for m in SCORING_MODES) + " | CER indic |" + (" LLM-WER |" if judge else "")
        out += [header, "|" + "---|" * (len(SCORING_MODES) + 2 + bool(judge))]

        primary = {}
        bucket_rows = []
        fallbacks = []
        for name, path in systems.items():
            hyps = {k: v for k, v in read_hyps(path).items() if k in refs}
            if not hyps:
                continue
            cells = []
            for mode in SCORING_MODES:
                sc = score_all(refs, hyps, mode)
                ci = bootstrap_ci([s.word_errors for s in sc], [s.ref_words for s in sc], n_boot=n_boot)
                cells.append(ci.fmt())
                if mode == "indic+translit":
                    primary[name] = sc
                if mode == "indic":
                    cer = bootstrap_ci([s.char_errors for s in sc], [s.ref_chars for s in sc], n_boot=n_boot)
                if mode == "basic":
                    bucket_rows.append((name, bucket_counts([s.alignment for s in sc])))
                    basic = sc
            row = f"| {name.removeprefix('hyp_')} | " + " | ".join(cells) + f" | {cer.fmt()} |"
            if judge and judge_langs and lang not in judge_langs:
                row += " n/a |"
            elif judge:
                from .judge import llm_errors

                res = judge.judge_many([(refs[s.id], hyps.get(s.id, ""), s) for s in basic])
                errs = [llm_errors(pairs, same) for pairs, same in res]
                n_fb = sum(None in same for _, same in res)
                if n_fb:
                    fallbacks.append((name.removeprefix("hyp_"), n_fb))
                row += f" {bootstrap_ci(errs, [s.ref_words for s in basic], n_boot=n_boot).fmt()} |"
            out.append(row)

        out.append("\nWER in %, 95% bootstrap interval over utterances in brackets.\n")
        if fallbacks:
            total = sum(n for _, n in fallbacks)
            per = ", ".join(f"{k}: {n}" for k, n in fallbacks)
            out.append(f"Judge fell back to unknown (counted as errors) for {total} utterances ({per}).\n")

        if len(primary) > 1:
            best = min(primary, key=lambda k: sum(s.word_errors for s in primary[k]))
            out += [f"Paired bootstrap vs `{best.removeprefix('hyp_')}` (indic+translit WER):\n", "| system | ΔWER | p |", "|---|---|---|"]
            b = primary[best]
            for name, sc in primary.items():
                if name == best:
                    continue
                res = paired_bootstrap([s.word_errors for s in sc], [s.word_errors for s in b], [s.ref_words for s in b])
                out.append(f"| {name.removeprefix('hyp_')} | {res.diff.fmt()} | {res.p_value:.3f} |")
            out.append("")

        out += ["Error buckets (basic normalisation, word ops):\n", "| system | " + " | ".join(BUCKETS) + " |", "|" + "---|" * (len(BUCKETS) + 1)]
        for name, c in bucket_rows:
            out.append(f"| {name.removeprefix('hyp_')} | " + " | ".join(str(c.get(b, 0)) for b in BUCKETS) + " |")
        out.append("")

    text = "\n".join(out)
    (rdir / "report.md").write_text(text, encoding="utf-8")
    return text
