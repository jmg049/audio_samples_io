#!/usr/bin/env python3
"""Load Criterion + C benchmark data and print performance comparison tables."""

import json
import csv
from pathlib import Path

CRITERION_DIR = Path("target/criterion")
C_CSV = Path("benches/C/flac_bench.csv")


def load_criterion() -> list[dict]:
    rows = []
    if not CRITERION_DIR.exists():
        return rows
    for group_dir in CRITERION_DIR.iterdir():
        if not group_dir.is_dir():
            continue
        group = group_dir.name
        for bench_dir in group_dir.iterdir():
            if not bench_dir.is_dir():
                continue
            bench_id = bench_dir.name
            for case_dir in bench_dir.iterdir():
                if not case_dir.is_dir():
                    continue
                case_label = case_dir.name
                estimates_path = case_dir / "new" / "estimates.json"
                benchmark_path = case_dir / "new" / "benchmark.json"
                if not estimates_path.exists():
                    continue
                with open(estimates_path) as f:
                    est = json.load(f)
                throughput_bytes = None
                if benchmark_path.exists():
                    with open(benchmark_path) as f:
                        bm = json.load(f)
                    tp = bm.get("throughput")
                    if tp and isinstance(tp, dict):
                        throughput_bytes = tp.get("bytes") or tp.get("Bytes")
                    elif tp and isinstance(tp, list):
                        for entry in tp:
                            if isinstance(entry, dict) and ("bytes" in entry or "Bytes" in entry):
                                throughput_bytes = entry.get("bytes") or entry.get("Bytes")
                                break
                mean_ns = est["mean"]["point_estimate"]
                std_ns = est["std_dev"]["point_estimate"]
                mbps = None
                if throughput_bytes:
                    mbps = (throughput_bytes / (mean_ns / 1e9)) / 1e6
                rows.append({
                    "source": "rust",
                    "group": group,
                    "bench_id": bench_id,
                    "case_label": case_label,
                    "mean_ns": mean_ns,
                    "std_ns": std_ns,
                    "throughput_bytes": throughput_bytes,
                    "mbps": mbps,
                })
    return rows


def load_c() -> list[dict]:
    rows = []
    if not C_CSV.exists():
        return rows
    with open(C_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mbps = None
            mean_ns = float(row["mean_ns"])
            tb = float(row.get("throughput_bytes") or 0)
            if tb > 0 and mean_ns > 0:
                mbps = (tb / (mean_ns / 1e9)) / 1e6
            rows.append({
                "source": "c",
                "group": row["group"],
                "bench_id": row["bench_id"],
                "case_label": row["case_label"],
                "mean_ns": mean_ns,
                "std_ns": float(row["stdev_ns"]),
                "throughput_bytes": tb or None,
                "mbps": mbps,
            })
    return rows


def format_ns(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.1f} ns"
    if ns < 1_000_000:
        return f"{ns/1_000:.1f} µs"
    return f"{ns/1_000_000:.1f} ms"


def print_table(title: str, rows: list[dict], group: str):
    group_rows = [r for r in rows if r["group"] == group]
    if not group_rows:
        print(f"  (no data for group '{group}')\n")
        return

    cases = sorted(set(r["case_label"] for r in group_rows))
    bench_ids = sorted(set(r["bench_id"] for r in group_rows))

    # Index by (bench_id, case_label)
    idx = {(r["bench_id"], r["case_label"]): r for r in group_rows}

    col_w = 22
    id_w = 30

    print(f"\n{'─'*10} {title} {'─'*10}")
    header = f"  {'bench':<{id_w}}" + "".join(f"  {c:>{col_w}}" for c in cases)
    print(header)
    print(f"  {'':<{id_w}}" + "".join(f"  {'mean (MBps)':>{col_w}}" for _ in cases))
    print("  " + "─" * (id_w + (col_w + 2) * len(cases)))

    for bid in bench_ids:
        cells = []
        for case in cases:
            r = idx.get((bid, case))
            if r is None:
                cells.append(f"{'—':>{col_w}}")
            elif r["mbps"] is not None:
                cells.append(f"{r['mbps']:>{col_w}.1f}")
            else:
                cells.append(f"{format_ns(r['mean_ns']):>{col_w}}")
        print(f"  {bid:<{id_w}}" + "".join(f"  {c}" for c in cells))

    # Ratio table vs libFLAC (if c data exists)
    c_libflac = {r["case_label"]: r for r in group_rows if r["bench_id"] == "libFLAC"}
    if c_libflac:
        print(f"\n  {'ratio vs libFLAC (lower=faster)':<{id_w + 2}}")
        print("  " + "─" * (id_w + (col_w + 2) * len(cases)))
        for bid in bench_ids:
            if bid == "libFLAC":
                continue
            cells = []
            for case in cases:
                r = idx.get((bid, case))
                ref = c_libflac.get(case)
                if r and ref and r.get("mbps") and ref.get("mbps"):
                    ratio = ref["mbps"] / r["mbps"]
                    marker = " ✓" if ratio <= 1.0 else "  "
                    cells.append(f"{ratio:>{col_w-2}.2f}x{marker}")
                else:
                    cells.append(f"{'—':>{col_w}}")
            print(f"  {bid:<{id_w}}" + "".join(f"  {c}" for c in cells))

    print()


def main():
    rust_rows = load_criterion()
    c_rows = load_c()
    all_rows = rust_rows + c_rows

    if not all_rows:
        print("No benchmark data found.")
        print(f"  Criterion: {CRITERION_DIR} ({'exists' if CRITERION_DIR.exists() else 'missing'})")
        print(f"  C CSV:     {C_CSV} ({'exists' if C_CSV.exists() else 'missing'})")
        return

    print(f"\nLoaded {len(rust_rows)} Criterion entries, {len(c_rows)} C entries")

    groups = sorted(set(r["group"] for r in all_rows))
    for group in groups:
        print_table(group, all_rows, group)


if __name__ == "__main__":
    main()
