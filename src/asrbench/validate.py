"""Check the judge against human labels before trusting LLM-WER.

export: sample error pairs (stratified by bucket, so rare buckets show up)
        into a CSV with an empty human_same column.
score:  after filling human_same with 1/0, report agreement, Cohen's kappa
        and where the judge and the human disagree.
"""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict


def export_sample(records: list[dict], out_csv: str, n: int = 50, seed: int = 0) -> None:
    by_bucket = defaultdict(list)
    for r in records:
        by_bucket[r["bucket"]].append(r)
    rng = random.Random(seed)
    # Round-robin across buckets so lexical errors (the majority) don't crowd
    # out the script and number cases the judge is most likely to get wrong.
    pools = {b: rng.sample(v, len(v)) for b, v in by_bucket.items()}
    picked = []
    while len(picked) < n and any(pools.values()):
        for b in sorted(pools):
            if pools[b] and len(picked) < n:
                picked.append(pools[b].pop())
    rng.shuffle(picked)
    fields = ["system", "id", "bucket", "ref_tok", "hyp_tok", "ref", "hyp", "judge_same", "human_same"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in picked:
            w.writerow({k: r.get(k, "") for k in fields} | {"human_same": ""})


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score_labels(csv_path: str) -> dict:
    with open(csv_path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human_same"].strip() in ("0", "1")]
    if not rows:
        raise ValueError("no labelled rows; fill human_same with 1 (same content) or 0 (real error)")
    j = [int(r["judge_same"] in ("1", "True", "true")) for r in rows]
    h = [int(r["human_same"]) for r in rows]
    confusion = Counter(zip(h, j))
    per_bucket = defaultdict(lambda: [0, 0])
    for r, x, y in zip(rows, h, j):
        per_bucket[r["bucket"]][0] += x == y
        per_bucket[r["bucket"]][1] += 1
    return {
        "n": len(rows),
        "agreement": sum(x == y for x, y in zip(h, j)) / len(rows),
        "kappa": cohen_kappa(h, j),
        # The judge being too lenient (calling real errors "same") is the
        # failure that makes LLM-WER look better than it should.
        "judge_lenient": confusion[(0, 1)],
        "judge_strict": confusion[(1, 0)],
        "per_bucket": {b: f"{c}/{t}" for b, (c, t) in sorted(per_bucket.items())},
        "disagreements": [r for r, x, y in zip(rows, h, j) if x != y],
    }
