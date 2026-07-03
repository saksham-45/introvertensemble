"""Render the README results table from an evaluation JSON export.

This makes the headline numbers a build artifact, not hand-typed prose: the old
README bolded PPO as the winner on layouts where it lost to no-op (B1). The table
is written between the markers

    <!-- results:begin --> ... <!-- results:end -->

so `make_results_table.py` fully owns that block. Under-powered runs (too few
episodes per cell) are refused unless --force, so a 4-episode smoke test can
never masquerade as a result (B6).

Usage:
    python scripts/evaluate_agent.py --episodes 50 --num-seeds 5 \
        --eval-layouts library_v1 library_v2_riverside --export-json results.json
    python scripts/make_results_table.py --results results.json --readme README.md
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"


def _mean_ci(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * math.sqrt(var) / math.sqrt(n)


def build_table(records: list[dict], min_episodes: int, force: bool) -> str:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["layout"], r["policy"])].append(r)

    counts = {k: len(v) for k, v in grouped.items()}
    underpowered = {k: n for k, n in counts.items() if n < min_episodes}
    if underpowered and not force:
        detail = ", ".join(f"{layout}/{policy}={n}" for (layout, policy), n in underpowered.items())
        raise SystemExit(
            f"Refusing to write an under-powered results table: {detail} "
            f"(need >= {min_episodes} episodes/cell). Re-run eval with more "
            f"--episodes/--num-seeds, or pass --force to override."
        )

    modes = sorted({r.get("reward_mode", "?") for r in records})
    spawns = sorted({r.get("spawn", "?") for r in records})
    caption = (
        f"Reward mode: `{', '.join(modes)}` · spawn: `{', '.join(spawns)}` · "
        f"{min(counts.values())}–{max(counts.values())} episodes/cell · "
        f"total reward, mean ± 95% CI. Higher is better."
    )

    lines = [
        BEGIN,
        "",
        caption,
        "",
        "| Layout | Policy | Total reward (mean ± 95% CI) | Moves/ep |",
        "|--------|--------|------------------------------|----------|",
    ]
    for (layout, policy) in sorted(grouped):
        recs = grouped[(layout, policy)]
        mean, ci = _mean_ci([r["total_reward"] for r in recs])
        moves = sum(r["moves"] for r in recs) / len(recs)
        lines.append(f"| {layout} | {policy} | {mean:.2f} ± {ci:.2f} | {moves:.2f} |")
    lines += ["", END]
    return "\n".join(lines)


def inject(readme: Path, table: str) -> None:
    text = readme.read_text()
    if BEGIN in text and END in text:
        pre = text[: text.index(BEGIN)]
        post = text[text.index(END) + len(END):]
        readme.write_text(pre + table + post)
    else:
        # Append a results section if the markers are absent.
        readme.write_text(text.rstrip() + "\n\n## Evaluation results\n\n" + table + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render README results table from an eval JSON export.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--readme", type=Path, default=Path(__file__).resolve().parents[1] / "README.md")
    parser.add_argument("--min-episodes", type=int, default=30,
                        help="Minimum episodes per (layout, policy) cell (B6).")
    parser.add_argument("--force", action="store_true", help="Write even if under-powered.")
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="Print the table instead of writing the README.")
    args = parser.parse_args()

    records = json.loads(args.results.read_text())
    table = build_table(records, args.min_episodes, args.force)
    if args.print_only:
        print(table)
    else:
        inject(args.readme, table)
        print(f"Wrote results table to {args.readme}")


if __name__ == "__main__":
    main()
