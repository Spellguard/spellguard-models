# /// script
# requires-python = ">=3.11"
# ///
"""Gather the fetched sets into one items file and check it against the items the published results used.

    uv run items.py <fetched-dir> <items.jsonl>

Writes one row per item: benchmark, set, id, text, pair (the reply being judged, or null), label and language.
Then compares every set with items-manifest.json (item count, positives, and a SHA-256 over each item's id,
label, text and reply, in order). Exits 1 if any set differs: the scores would then be on different items.
"""

import hashlib
import json
import sys
from pathlib import Path

BENCHMARKS = ["aa-guardrail", "content-safety", "horizon-pi", "jailbreakbench", "phtest"]
HERE = Path(__file__).resolve().parent


def build(fetched: Path) -> list[dict]:
    rows = []
    for bench in BENCHMARKS:
        files = sorted((fetched / bench).glob("*.jsonl"))
        if not files:
            sys.exit(f"no sets for {bench} under {fetched}: run fetch.py first")
        for f in files:
            for line in f.open(encoding="utf-8"):
                it = json.loads(line)
                rows.append({"benchmark": bench, "set": f.stem, "id": it["id"], "text": it["text"],
                             "pair": it.get("pair"), "label": it.get("label"), "lang": it.get("lang")})
    return rows


def manifest(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        k = f"{r['benchmark']}|{r['set']}"
        m = out.setdefault(k, {"items": 0, "positives": 0, "_h": hashlib.sha256()})
        m["items"] += 1
        m["positives"] += int(r["label"])
        m["_h"].update((json.dumps([r["id"], r["label"], r["text"], r["pair"]], ensure_ascii=False) + "\n").encode())
    return {k: {"items": v["items"], "positives": v["positives"], "sha256": v["_h"].hexdigest()} for k, v in out.items()}


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: items.py <fetched-dir> <items.jsonl>")
    rows = build(Path(sys.argv[1]))
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    got, want = manifest(rows), json.load(open(HERE / "items-manifest.json"))["sets"]
    bad = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
    for k in bad:
        print(f"DIFFERS {k}: got {got.get(k)} want {want.get(k)}")
    print(f"{len(rows)} items in {len(got)} sets; {len(got) - len(bad)} of {len(want)} sets match the published items")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
