"""Generate disjoint train / val / test pools of procedural layouts.

Follows the ProcGen protocol (docs/TRAINING_BEST_PRACTICES.md): the agent trains
on the train pool and is measured on the never-seen test pool. Seed ranges are
disjoint so no layout leaks across splits.

    python scripts/generate_layouts.py --out assets/generated \
        --n-train 128 --n-val 16 --n-test 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introvertensemble.layout_generator import write_layout
from introvertensemble.loader import load_layout
from introvertensemble.simulation import LibrarySimulation, SimulationConfig
from introvertensemble.world import LibraryWorld


def _validate_layout(layout_dir: Path) -> tuple[int, int]:
    """Load + run a short simulation to prove the layout is well-formed."""
    spec = load_layout(layout_dir)
    world = LibraryWorld(spec)  # precomputes all-pairs shortest paths -> connectivity check
    sim = LibrarySimulation(world, config=SimulationConfig(events_enabled=True), seed=0)
    for _ in range(20):
        sim.step()
    return len(spec.seats), len(spec.zones)


def _make_split(out_root: Path, split: str, seeds: range, difficulty_fn) -> list[str]:
    names = []
    for seed in seeds:
        name = f"{split}_{seed:04d}"
        layout_dir = out_root / split / name
        write_layout(layout_dir, seed=seed, difficulty=difficulty_fn(seed))
        n_seats, n_zones = _validate_layout(layout_dir)
        names.append(name)
        print(f"  {name}: {n_seats} seats, {n_zones} zones  OK")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/val/test layout pools.")
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "generated")
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-val", type=int, default=16)
    parser.add_argument("--n-test", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=1000)
    args = parser.parse_args()

    # Disjoint seed ranges guarantee no layout is shared across splits.
    train_seeds = range(args.seed_base, args.seed_base + args.n_train)
    val_seeds = range(args.seed_base + 10_000, args.seed_base + 10_000 + args.n_val)
    test_seeds = range(args.seed_base + 20_000, args.seed_base + 20_000 + args.n_test)

    # Train/val span the full difficulty range; test does too (measures general
    # competence, not a single difficulty).
    def spread_difficulty(seed: int) -> float:
        return 0.2 + 0.6 * ((seed % 7) / 6.0)

    print(f"Generating {args.n_train} train layouts -> {args.out}/train")
    train = _make_split(args.out, "train", train_seeds, spread_difficulty)
    print(f"Generating {args.n_val} val layouts -> {args.out}/val")
    val = _make_split(args.out, "val", val_seeds, spread_difficulty)
    print(f"Generating {args.n_test} test layouts -> {args.out}/test")
    test = _make_split(args.out, "test", test_seeds, spread_difficulty)

    index = {
        "seed_base": args.seed_base,
        "train": train, "val": val, "test": test,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
    }
    (args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "splits.json").write_text(json.dumps(index, indent=2))
    print(f"\nWrote split index to {args.out / 'splits.json'}")
    print(f"train={len(train)} val={len(val)} test={len(test)} (disjoint seeds, all validated)")


if __name__ == "__main__":
    main()
