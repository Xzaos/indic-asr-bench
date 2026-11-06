from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_prepare(a):
    from .data import prepare_folder, prepare_hf

    if a.folder:
        print(prepare_folder(a.folder, a.lang, a.out, max_sec=a.max_sec))
    else:
        print(prepare_hf(a.dataset, a.config, a.split, a.lang, a.text_field, a.out, a.n, a.seed, a.tag, a.max_sec))


def cmd_filter(a):
    from .data import filter_long, read_manifest, write_jsonl

    rows = read_manifest(a.manifest)
    kept, dropped = filter_long(rows, a.max_sec)
    write_jsonl(a.out, kept)
    for lang in sorted({r["lang"] for r in rows}):
        print(f"{lang}: dropped {dropped[lang]} of {sum(r['lang'] == lang for r in rows)} clips longer than {a.max_sec:g} s")
    print(a.out)


def cmd_transcribe(a):
    from .data import read_manifest
    from .models import get_backend

    rows = read_manifest(a.manifest)
    if a.lang:
        rows = [r for r in rows if r["lang"] == a.lang]
    kwargs = {}
    if a.backend.startswith("whisper"):
        kwargs = {"model_id": a.model_id or "openai/whisper-large-v3", "batch_size": a.batch_size, "peft_adapter": a.adapter}
    elif a.backend == "indicconformer":
        kwargs = {"decoder": a.decoder}
    else:
        kwargs = {"model_id": a.model_id or a.backend, "mode": a.mode, "concurrency": a.concurrency}
    name = a.name or a.backend.replace(":", "-")
    out = Path(a.results) / f"hyp_{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    get_backend(a.backend, **kwargs).run(rows, str(out))
    print(out)


def _judge(a):
    if not a.judge_model:
        return None
    from .judge import Judge

    return Judge(a.judge_model, a.judge_base_url, cache=str(Path(a.results) / "judge_cache.jsonl"), workers=a.judge_workers, reasoning=a.judge_reasoning)


def cmd_report(a):
    from .report import build

    langs = None if "all" in a.judge_lang else set(a.judge_lang)
    print(build(a.manifest, a.results, judge=_judge(a), n_boot=a.n_boot, judge_langs=langs))


def cmd_label_export(a):
    from .data import read_hyps, read_manifest
    from .judge import _pairs
    from .metrics import score_all
    from .validate import sample_pairs, write_sample

    refs = {r["id"]: r["text"] for r in read_manifest(a.manifest) if not a.lang or r["lang"] == a.lang}
    records = []
    for path in sorted(Path(a.results).glob("hyp_*.jsonl")):
        hyps = read_hyps(path)
        for s in score_all(refs, {k: v for k, v in hyps.items() if k in refs}, "basic"):
            for idx, p in enumerate(_pairs(s)):
                records.append({"system": path.stem.removeprefix("hyp_"), "id": s.id, "bucket": p["bucket"], "ref_tok": p["ref"], "hyp_tok": p["hyp"], "ref": refs[s.id], "hyp": hyps.get(s.id, ""), "judge_same": "", "_score": s, "_idx": idx})
    picked = sample_pairs(records, n=a.n, seed=a.seed)
    judge = _judge(a)
    if judge:
        utts = {(r["system"], r["id"]): r for r in picked}
        res = judge.judge_many([(r["ref"], r["hyp"], r["_score"]) for r in utts.values()])
        verdicts = dict(zip(utts, (same for _, same in res)))
        for r in picked:
            v = verdicts[(r["system"], r["id"])][r["_idx"]]
            r["judge_same"] = "" if v is None else int(v)
    write_sample(picked, a.out)
    print(f"wrote {a.out}; fill human_same with 1 or 0")


def cmd_label_score(a):
    from .validate import score_labels

    res = score_labels(a.csv)
    dis = res.pop("disagreements")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    for r in dis[:15]:
        print(f"[{r['bucket']}] ref={r['ref_tok']!r} hyp={r['hyp_tok']!r} human={r['human_same']} judge={r['judge_same']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="asrbench")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("prepare", help="build a manifest from a HF dataset or a folder")
    s.add_argument("--dataset")
    s.add_argument("--config")
    s.add_argument("--split", default="test")
    s.add_argument("--text-field", default="transcription")
    s.add_argument("--folder")
    s.add_argument("--lang", required=True)
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tag")
    s.add_argument("--max-sec", type=float, default=30)
    s.add_argument("--out", default="data")
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("filter", help="drop clips longer than --max-sec from a manifest")
    s.add_argument("--manifest", required=True)
    s.add_argument("--max-sec", type=float, default=30)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_filter)

    s = sub.add_parser("transcribe")
    s.add_argument("backend", help="whisper | indicconformer | saaras:v3 | saarika:v2.5")
    s.add_argument("--manifest", required=True)
    s.add_argument("--lang")
    s.add_argument("--model-id")
    s.add_argument("--adapter", help="PEFT adapter dir for a fine-tuned Whisper")
    s.add_argument("--name", help="system name in the report")
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--decoder", default="ctc")
    s.add_argument("--mode", default="transcribe")
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--results", default="results")
    s.set_defaults(fn=cmd_transcribe)

    for name, fn in (("report", cmd_report), ("label-export", cmd_label_export)):
        s = sub.add_parser(name)
        s.add_argument("--manifest", required=True)
        s.add_argument("--results", default="results")
        s.add_argument("--judge-model")
        s.add_argument("--judge-base-url")
        s.add_argument("--judge-workers", type=int, default=8)
        s.add_argument("--judge-reasoning", default="low", choices=["none", "low", "high", "max"])
        s.add_argument("--n-boot", type=int, default=2000)
        if name == "report":
            s.add_argument("--judge-lang", nargs="+", default=["all"], help="languages to compute LLM-WER for")
        if name == "label-export":
            s.add_argument("--out", default="results/judge_labels.csv")
            s.add_argument("--n", type=int, default=50)
            s.add_argument("--lang", help="only sample this language")
            s.add_argument("--seed", type=int, default=0)
        s.set_defaults(fn=fn)

    s = sub.add_parser("label-score")
    s.add_argument("csv")
    s.set_defaults(fn=cmd_label_score)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
