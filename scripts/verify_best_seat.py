from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo src is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introvertensemble.env import LibraryEnv
from introvertensemble.scoring import SeatScorer
from introvertensemble.simulation import SimulationConfig


def main():
    # Create a config that lets the simulation place the focal agent using its heuristic
    config = SimulationConfig(
        focal_agent_external_control=False,
        focal_agent_enabled=True,
        events_enabled=True,
    )
    env = LibraryEnv(seed=42, config=config)
    # Reset returns observation and info dict; info contains the focal seat ID after initial placement
    obs_vec, info = env.reset(seed=123)
    focal_id = env.sim.focal_agent_id
    focal_agent = env.sim.agents[focal_id]
    current_seat = focal_agent.current_seat_id
    print(f"Focal agent initially seated at: {current_seat}")

    # Compute scores for every seat using SeatScorer
    scorer = SeatScorer(env.world)
    seat_scores = {seat_id: scorer.score_seat(focal_agent, seat_id).total for seat_id in env.spec.seats}
    # Identify the seat with the highest total utility
    best_seat = max(seat_scores, key=seat_scores.get)
    print(f"Best seat according to scoring: {best_seat} (score {seat_scores[best_seat]:.4f})")
    # Show top 5 seats
    top_n = 5
    print(f"Top {top_n} seats:")
    for sid in sorted(seat_scores, key=seat_scores.get, reverse=True)[:top_n]:
        print(f"{sid}: {seat_scores[sid]:.4f}")

if __name__ == "__main__":
    main()
