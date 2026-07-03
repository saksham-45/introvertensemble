# Decisions (ADR-lite)

Short, append-only record of irreversible modeling/engineering choices. Each entry:
context → decision → consequence. Newest last.

## D1 — Split NPC utility from RL reward (2026-07)
- **Context:** the hand-crafted `SeatScorer` doubles as NPC behavior model and RL
  reward. Its familiarity (+0.25/seat-visit) and dwell bonuses accrue on cost-free
  moves, so a learner is rewarded for seat-hopping (reward gaming).
- **Decision:** introduce `LibraryEnv.reward_mode`. `legacy` keeps reward == raw
  score (reproducibility + reward-hacking case study). `environment` strips
  familiarity + dwell from the *reward* (they remain in NPC utility) and charges a
  realized movement cost on the step a move happens.
- **Consequence:** habit must now *emerge* from the movement-cost/anticipation
  tradeoff rather than being paid for directly; legacy results stay reproducible.

## D2 — Movement cost model (2026-07)
- **Context:** need to penalize relocation without a full transit-time model yet.
- **Decision:** scalar tax — `reward -= move_cost_scale * normalized_path_cost`
  (`move_cost_scale = 1.5`) only on the move step. Multi-step transit is deferred.
- **Consequence:** one-line, testable; a cross-library move costs ≈1–2 median-score
  steps. Revisit if policies exploit "free teleport within cheap-cost radius".

## D3 — Withhold true score from observations (2026-07)
- **Context:** the observation included each candidate's true score (== reward),
  making the task a near-bandit.
- **Decision:** `score_in_obs` flag (default off for new work) zeroes the score
  channels; kept configurable for the ablation arm.
- **Consequence:** policy must infer quality from raw features; obs dim stays 141
  (score channels become constant-zero) so saved-model shapes are unaffected.

## D4 — Random focal spawn for evaluation (2026-07)
- **Context:** the focal agent spawned at the scoring argmax, so Greedy and the
  "perfect-info oracle" never moved and were byte-identical; No-op looked strong.
- **Decision:** `focal_agent_random_spawn` — arrive at a uniformly random free seat
  during evaluation/training.
- **Consequence:** policies are scored on *finding* good seats. No-op is no longer
  trivially strong; baselines separate. Scored-spawn remains available.

## D5 — "Best-seat" reference is not an upper bound (2026-07)
- **Context:** wanted a strong reference above greedy.
- **Decision:** all-seats, no-cooldown, move-cost-aware myopic policy. Labeled a
  *reference*, not an oracle: choosing the pre-step argmax can be hurt by post-step
  crowding, so it does not dominate every policy.
- **Consequence:** resolves the greedy≡oracle defect honestly. A clairvoyant /
  H-step MPC oracle (feasible since utilities are closed-form) is future work.

## D6 — Results table is a generated artifact (2026-07)
- **Context:** README bolded PPO as winner where it lost to No-op; numbers were
  hand-typed and disagreed with the table below them.
- **Decision:** `make_results_table.py` renders the table from an eval JSON export
  between `<!-- results:begin/end -->` markers, and refuses under-powered runs.
- **Consequence:** headline numbers can no longer contradict the data; a smoke test
  cannot masquerade as a result.
