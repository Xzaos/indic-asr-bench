import json

import numpy as np
import soundfile as sf

from asrbench.cli import main
from asrbench.data import prepare_folder, read_manifest


def _wav(path, sec):
    sf.write(path, np.zeros(int(16000 * sec), dtype="float32"), 16000)
    return str(path)


def _manifest(tmp_path):
    rows = [
        {"id": "a", "audio": _wav(tmp_path / "a.wav", 5), "text": "x", "lang": "hi"},
        {"id": "b", "audio": _wav(tmp_path / "b.wav", 31), "text": "x", "lang": "hi"},
        {"id": "c", "audio": _wav(tmp_path / "c.wav", 40), "text": "x", "lang": "ta"},
        {"id": "d", "audio": _wav(tmp_path / "d.wav", 30), "text": "x", "lang": "ta"},
    ]
    man = tmp_path / "m.jsonl"
    man.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return man


def test_filter_drops_long_and_reports_per_language(tmp_path, capsys):
    man = _manifest(tmp_path)
    out = tmp_path / "f.jsonl"
    main(["filter", "--manifest", str(man), "--max-sec", "30", "--out", str(out)])
    assert [r["id"] for r in read_manifest(out)] == ["a", "d"]
    text = capsys.readouterr().out
    assert "hi: dropped 1 of 2" in text and "ta: dropped 1 of 2" in text


def test_prepare_folder_applies_max_sec(tmp_path):
    _wav(tmp_path / "s.wav", 2)
    _wav(tmp_path / "l.wav", 45)
    (tmp_path / "transcripts.tsv").write_text("s.wav\tone\nl.wav\ttwo\n", encoding="utf-8")
    out = tmp_path / "o.jsonl"
    prepare_folder(str(tmp_path), "hi", str(out))
    assert [r["id"] for r in read_manifest(out)] == ["s"]
    prepare_folder(str(tmp_path), "hi", str(out), max_sec=60)
    assert len(read_manifest(out)) == 2
