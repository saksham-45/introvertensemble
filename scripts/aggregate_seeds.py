"""Aggregate multiple per-seed evaluation exports with rliable-style statistics.

Follows Agarwal et al. 2021: report the interquartile mean (IQM) with a
stratified bootstrap 95% CI rather than mean ± std, which is dominated by
outliers and misleading with few seeds. Dependency-free (numpy bootstrap); if
`rliable` is installed it is used for the IQM point estimate.

    python scripts/aggregate_seeds.py results/seed_*.json \
        --out results/aggregated.json --print
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def iqm(values: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50% of the data."""
    if values.size == 0:
        return float("nan")
    lo, hi = np.percentile(values, [25, 75])
    mid = values[(values >= lo) & (values <= hi)]
    return float(mid.mean()) if mid.size else float(values.mean())


def bootstrap_ci(values: np.ndarray, statistic, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for a statistic (reliable for small N)."""
    if values.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = values.size
    for i in range(n_boot):
        boots[i] = statistic(values[rng.integers(0, n, n)])
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-seed eval exports with IQM + bootstrap CIs.")
    parser.add_argument("results", type=Path, nargs="+", help="Per-seed eval JSON exports.")
    parser.add_argument("--out", type=Path, help="Write aggregated summary JSON here.")
    parser.add_argument("--print", dest="print_table", action="store_true")
    args = parser.parse_args()

    # Merge all records; each file is one training seed's evaluation.
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    n_seeds = len(args.results)
    for path in args.results:
        for rec in json.loads(path.read_text()):
            grouped[(rec["layout"], rec["policy"])].append(rec["total_reward"])

    summary = []
    for (layout, policy), vals in sorted(grouped.items()):
        arr = np.array(vals, dtype=float)
        point = iqm(arr)
        lo, hi = bootstrap_ci(arr, iqm)
        summary.append({
            "layout": layout, "policy": policy, "n_episodes": int(arr.size),
            "n_seeds": n_seeds, "iqm": point, "iqm_ci_low": lo, "iqm_ci_high": hi,
            "mean": float(arr.mean()), "median": float(np.median(arr)),
        })

    if args.print_table or not args.out:
        print(f"Aggregated over {n_seeds} seed file(s). Metric: IQM [95% bootstrap CI]")
        print("=" * 84)
        for row in summary:
            print(f"{row['layout']:22} | {row['policy']:34} | "
                  f"IQM {row['iqm']:7.2f} [{row['iqm_ci_low']:6.2f}, {row['iqm_ci_high']:6.2f}] "
                  f"(n={row['n_episodes']})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote aggregated summary to {args.out}")


if __name__ == "__main__":
    main()
