"""Go/no-go benchmark on REAL ID photos, end to end through the same IdCardReader the API uses.

    python spike/bench.py --init spike/samples/     # write a labels.csv template for the photos
    python spike/bench.py spike/samples/            # run the benchmark

labels.csv columns: file,id_number,name_khmer,name_latin,date_of_birth,sex,expiry_date
(values in the normalised form the API returns: ISO dates, M/F, uppercase Latin name).

Prints per-field exact-match rate and character error rate (CER), plus latency percentiles.
It never prints field values, so the output is safe to paste into a ticket.
spike/samples/ is gitignored: real IDs are personal data and never enter the repo.

Gate (from the build plan): id_number exact match >= 95% and p95 latency < 1500 ms on the
target CPU. Exit code 1 if the gate fails, so this can run in CI against a private sample set.
"""

import csv
import statistics
import sys
import time
from pathlib import Path

from khmer_kyc_ocr.config import Settings
from khmer_kyc_ocr.pipeline.kiri import KiriRecognizer
from khmer_kyc_ocr.schemas import FieldName, IdCardResult
from khmer_kyc_ocr.service import PARSERS, IdCardReader

GATE_ID_EXACT = 0.95
GATE_P95_MS = 1500.0
FIELDS: list[FieldName] = list(PARSERS)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def find_photos(sample_dir: Path) -> list[Path]:
    return sorted(p for p in sample_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def init_labels(sample_dir: Path) -> int:
    """Write labels.csv with one row per photo and empty answer columns to fill in by hand."""
    labels = sample_dir / "labels.csv"
    if labels.exists():
        print(f"{labels} already exists; not overwriting your answers")
        return 1
    photos = find_photos(sample_dir)
    if not photos:
        print(f"no photos in {sample_dir}: add front-of-card .jpg/.png/.webp files first")
        return 1
    with labels.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", *FIELDS])
        writer.writerows([p.name, *([""] * len(FIELDS))] for p in photos)
    print(f"wrote {labels} with {len(photos)} rows. Fill each row in the API's format:")
    print("  id_number 9 digits | names as printed (Latin UPPERCASE) | dates YYYY-MM-DD | sex M/F")
    return 0


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main(sample_dir: Path) -> int:
    labels = sample_dir / "labels.csv"
    if not labels.exists():
        print(f"missing {labels}. Add card photos, then run:")
        print(f"  python spike/bench.py --init {sample_dir}")
        return 1
    with labels.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    unfilled = [r["file"] for r in rows if not r.get("id_number", "").strip()]
    if not rows or unfilled:
        print(f"labels.csv has {len(rows)} rows, {len(unfilled)} without answers: fill them first")
        return 1

    # The token is irrelevant offline; the reader is called directly, not over HTTP.
    reader = IdCardReader(KiriRecognizer(), Settings.model_validate({"api_token": "bench"}))
    latencies: list[float] = []
    exact = dict.fromkeys(FIELDS, 0)
    edits = dict.fromkeys(FIELDS, 0)
    chars = dict.fromkeys(FIELDS, 0)
    rejected: dict[str, int] = {}
    conf_right: list[float] = []
    conf_wrong: list[float] = []

    for row in rows:
        data = (sample_dir / row["file"]).read_bytes()
        started = time.perf_counter()
        result = reader.read(data)
        latencies.append((time.perf_counter() - started) * 1000)
        if not isinstance(result, IdCardResult):
            rejected[result.reason] = rejected.get(result.reason, 0) + 1
            for name in FIELDS:  # a rejection counts as every character wrong
                edits[name] += len(row[name])
                chars[name] += len(row[name])
            continue
        for name in FIELDS:
            got = result.fields[name].value or ""
            (conf_right if got == row[name] else conf_wrong).append(result.fields[name].confidence)
            exact[name] += got == row[name]
            edits[name] += levenshtein(got, row[name])
            chars[name] += len(row[name])

    n = len(rows)
    print(f"samples: {n}   rejected: {sum(rejected.values())} {rejected or ''}")
    print(f"{'field':<15}{'exact':>8}{'CER':>8}")
    for name in FIELDS:
        cer = edits[name] / max(chars[name], 1)
        print(f"{name:<15}{exact[name] / n:>8.1%}{cer:>8.1%}")
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[-1] if n >= 2 else latencies[0]
    print(f"latency ms: p50={p50:.0f} p95={p95:.0f} max={max(latencies):.0f}")

    # The review threshold must sit above most wrong reads: pick it from this, not by feel.
    if len(conf_right) > 1 and len(conf_wrong) > 1:
        right_p10 = statistics.quantiles(conf_right, n=10)[0]
        wrong_p90 = statistics.quantiles(conf_wrong, n=10)[-1]
        print(
            f"confidence  right: p10={right_p10:.2f} median={statistics.median(conf_right):.2f}"
            f"   wrong: median={statistics.median(conf_wrong):.2f} p90={wrong_p90:.2f}"
        )

    id_exact = exact["id_number"] / n
    passed = id_exact >= GATE_ID_EXACT and p95 < GATE_P95_MS
    print(
        f"GATE {'PASS' if passed else 'FAIL'}: id_number exact {id_exact:.1%} "
        f"(need {GATE_ID_EXACT:.0%}), p95 {p95:.0f}ms (need <{GATE_P95_MS:.0f})"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--init"] and len(args) == 2:
        sys.exit(init_labels(Path(args[1])))
    if len(args) != 1 or args[0].startswith("-"):
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(args[0])))
