"""Check a submission file (SPEC 09, E4.6).

    uv run python tools/check_submission.py <submission.csv> [--sample data/sample_submission.csv]

Exits 0 and prints one `RESULT ok` line only if the header is exactly `id,Exited`, there are 10,000
rows, the ids equal those of the sample file in the same order, and every `Exited` is a number in
[0, 1] (no NaN, no infinity). Otherwise it prints `RESULT fail: <the first violation>` and exits 1.

Both files are data and stay out of git. Standard library only.
"""
import argparse
import csv
import math
import sys

HEADER = ["id", "Exited"]
EXPECTED_ROWS = 10_000


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def first_violation(path, sample_path):
    """The first violation as a string, or None if the file is valid; also the (min, max) of Exited."""
    try:
        rows = read_csv(path)
    except OSError as e:
        return f"cannot read {path}: {e.strerror}", None
    try:
        sample = read_csv(sample_path)
    except OSError as e:
        return f"cannot read the sample {sample_path}: {e.strerror}", None

    if not rows:
        return "the file is empty", None
    if rows[0] != HEADER:
        return f"the header is {','.join(rows[0])!r}, expected 'id,Exited'", None
    data, sample_ids = rows[1:], [r[0] for r in sample[1:]]
    if len(sample_ids) != EXPECTED_ROWS:
        return f"the sample has {len(sample_ids)} rows, expected {EXPECTED_ROWS}", None
    if len(data) != EXPECTED_ROWS:
        return f"there are {len(data)} rows, expected {EXPECTED_ROWS}", None

    low, high = math.inf, -math.inf
    for n, (row, expected_id) in enumerate(zip(data, sample_ids), start=1):
        if len(row) != 2:
            return f"row {n} has {len(row)} column(s), expected 2", None
        if row[0] != expected_id:
            return f"row {n} has id {row[0]!r}, the sample has {expected_id!r}", None
        try:
            value = float(row[1])
        except ValueError:
            return f"row {n} (id {row[0]}) has Exited {row[1]!r}, which is not a number", None
        if not math.isfinite(value):
            return f"row {n} (id {row[0]}) has Exited {row[1]!r}, which is not finite", None
        if not 0 <= value <= 1:
            return f"row {n} (id {row[0]}) has Exited {value}, outside [0, 1]", None
        low, high = min(low, value), max(high, value)
    return None, (low, high)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    parser.add_argument("--sample", default="data/sample_submission.csv")
    args = parser.parse_args(argv)

    violation, span = first_violation(args.submission, args.sample)
    if violation:
        print(f"RESULT fail: {violation}")
        return 1
    print(f"RESULT ok: {EXPECTED_ROWS} rows, ids equal to the sample, Exited in [0, 1] "
          f"(min {span[0]:.6f}, max {span[1]:.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
