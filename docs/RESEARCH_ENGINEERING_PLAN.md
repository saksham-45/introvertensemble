# introvertensemble — Research & Engineering Upgrade Plan

**Status:** Draft v1 · **Date:** 2026-07-02
**Goal:** Turn introvertensemble from a promising prototype with untrustworthy numbers into a defensible research project with industry-grade engineering. Every claim in the README should survive a hostile reviewer.

---

## 0. Executive summary

The simulator is sound; the *experiment* around it is not. Three structural defects currently invalidate the headline results:

1. **The reward function is gameable** (familiarity + dwell bonuses accrue on free, instant moves), so the single-layout PPO "win" is plausibly reward hacking, not skill.
2. **The baselines are degenerate** (the focal agent spawns at the argmax of the same scoring function, so Greedy and the "Perfect-info oracle" never move and produce identical numbers).
3. **The evaluation is statistically void** (2 seeds × 2 episodes per cell) and the README bolds PPO as the winner on layouts where it loses to No-op by 25–30 points.

This plan fixes those in order: **hygiene → bug corrections → environment redesign → rigorous evaluation → retraining → honest writeup → MARL extension.** Each phase has acceptance criteria; do not start a phase until the previous one's criteria pass.

The silver lining: the reward-hacking behavior you accidentally produced is itself a publishable case study (reward specification gaming in a congestion environment). Don't delete the old model — it's evidence.

---

## 1. Research framing

### 1.1 What field is this?

This project sits at the intersection of four established literatures. Positioning it explicitly is what turns "I built a library sim" into a research question:

- **Congestion games** (Rosenthal 1973): seats are resources whose utility degrades with crowding. Pure-strategy Nash equilibria are guaranteed to exist — measurable ground truth for the MARL phase.
- **The El Farol Bar problem / minority games** (Arthur 1994; Challet & Zhang 1997): agents choosing whether/where to go under crowding externalities. Your "introvert" utility is a spatial, multi-resource El Farol.
- **Habit formation & switching costs in sequential decision-making**: your familiarity/dwell mechanics are a habit model. When is habit *rational* (amortizing search cost) vs. *exploitative* (gaming the utility model)?
- **Reward specification gaming** (Amodei et al. 2016, "Concrete Problems in AI Safety"; Skalse et al. 2022, "Defining and Characterizing Reward Gaming"): what your PPO agent appears to be doing today.

### 1.2 Research questions (pick RQ1 + one other; do not do all four at once)

**RQ1 (core, single-agent):**
*Under partial observability and non-stationary crowd dynamics, can a learned policy outperform myopic greedy seat selection — and is the gain attributable to anticipation of future arrivals?*
- **Hypothesis:** With the true seat score removed from observations and forecastable arrival waves added, PPO beats 1-step greedy on raw features but loses to a model-based lookahead oracle; the gap to the oracle measures how much dynamics-anticipation the policy learned.
- **Falsifiable prediction:** If arrival dynamics are made i.i.d. (unforecastable), the PPO-vs-greedy gap collapses to zero. This ablation is your causal evidence.
- **Why it's legit:** it's only answerable after fixing B4/B5 below; today's environment makes greedy trivially near-optimal.

**RQ2 (reward hacking case study — cheapest path to a real writeup):**
*How does a PPO agent exploit a hand-crafted utility model with visit-count familiarity bonuses and cost-free relocation, and which minimal reward corrections eliminate the exploit?*
- **Method:** Keep the current (buggy) environment as `library_v1_legacy`. Show, with per-component reward decomposition logging (the `SeatScoreBreakdown` you already have makes this nearly free), that PPO's excess return over No-op is concentrated in `familiarity` + `dwell_bonus`; then show the exploit disappears under each of the reward fixes in §3.
- **Deliverable:** workshop-paper / strong blog post. This is genuinely interesting and you have already done the hard part by accident.

**RQ3 (MARL, later):**
*When N introvert agents learn concurrently, do spatial equilibria emerge, and how far are they from the social optimum?*
- **Metrics:** price of anarchy (learned joint welfare ÷ centrally-optimized welfare), zone-occupancy Gini coefficient, seat "territory" stability across episodes, convergence to Nash (no agent gains by unilateral reseat — checkable exactly since utilities are known).
- **Prereq:** RQ1 environment must be sound first. Do not scale to 200 learners on a leaky reward.

**RQ4 (generalization, stretch):**
*Does domain randomization over procedurally generated layouts produce layout-invariant seat-selection policies?* Requires a layout generator (§4.5) so you can hold out a *distribution* of layouts, not just `library_v4`.

### 1.3 Reading list (minimum viable related-work section)

| Topic | Read |
|---|---|
| Congestion games | Rosenthal (1973); Roughgarden, *Selfish Routing and the Price of Anarchy* (ch. 1–2) |
| El Farol / minority games | Arthur (1994); Challet & Zhang (1997) |
| Reward gaming | Amodei et al. (2016); Skalse et al. (2022); Krakovna's specification-gaming examples list |
| MARL environments & social dilemmas | Leibo et al. (2017); PettingZoo paper (Terry et al. 2021); Melting Pot (Leibo et al. 2021) |
| Statistical practice in deep RL | **Agarwal et al. (2021), "Deep RL at the Edge of the Statistical Precipice" — mandatory**; Henderson et al. (2018) |
| Domain randomization | Tobin et al. (2017); Cobbe et al. (2020, ProcGen) for the held-out-distribution protocol |

### 1.4 Contribution statement (draft, to be earned)

> "We present a lightweight, fully-inspectable congestion environment for studying seat selection under crowding externalities and habit formation. Because agent utilities are known in closed form, exact oracles, Nash checks, and per-component reward decomposition are available — enabling (i) a controlled case study of reward specification gaming, and (ii) attribution of learned-policy gains to dynamics anticipation."

The *known-utility, exact-oracle* property is your actual differentiator versus ProcGen/Melting Pot-style environments. Lean on it.

---

## 2. Bug register

Severity: 🔴 invalidates results · 🟠 misleads readers · 🟡 hygiene/robustness.
Every fix ships with the listed acceptance test; a fix without its test is not done.

| ID | Sev | Location | Defect | Root cause | Fix | Acceptance test |
|---|---|---|---|---|---|---|
| **B1** | 🟠 | `README.md` results section | PPO bolded as winner while losing to No-op by 25–30 pts on 3 of 4 layouts; "generalized policies that perform" claim contradicted by own table | Aspirational writing | Rewrite results section from auto-generated table only (§4.6); state the negative result plainly | CI step regenerates the table; a docs test greps README for stale hardcoded reward numbers |
| **B2** | 🟠 | `scripts/evaluate_agent.py:381` | Column printed as `Steps:` actually contains mean **per-step reward** (`means`), propagated into README as "Mean Steps" | Copy-paste label | Print both `mean_step_reward` and true `steps`; rename headers everywhere | Unit test on the summary-row builder asserts field names ↔ values |
| **B3** | 🔴 | `scripts/evaluate_agent.py:98-116` + `simulation.py::_choose_arrival_seat` | "Perfect-info oracle" ≡ Greedy to the decimal; both make ~0 moves | Focal spawn seat is chosen by `_best_available_seat` using the *same scorer*, so the current seat is already argmax; oracle only moves when strictly better and respects cooldown | Oracle must (a) re-evaluate **every step**, (b) ignore cooldown, (c) optionally do H-step lookahead using the known arrival schedule (this is exact MPC — utilities are closed-form). Additionally add a *random-spawn* eval mode so policies must find good seats, not inherit them | Assert oracle ≥ every other policy per paired seed; assert oracle ≠ greedy trajectories on ≥1 seed per layout |
| **B4** | 🔴 | `scoring.py:60-79` + `env.py:179-194` | Reward is gameable: `record_seat` (+0.25/seat-visit, +0.12/zone-visit familiarity) fires on **every** move; dwell bonus stacks; relocation is **instant and free in realized reward** (movement penalty only exists in *candidate* scoring — from your own seat to itself the cost is 0) | Behavioral utility model (for NPCs) reused verbatim as RL reward | Full spec in §3.1–3.3: charge realized movement cost on the move step, make familiarity require sustained dwell + decay, and split *RL reward* from *NPC utility* | Regression test: a scripted ping-pong policy (alternate two seats every cooldown expiry) must earn **strictly less** than No-op on every layout |
| **B5** | 🔴 | `env.py:290, seat_to_features` | Observation includes each candidate's **true score** — the reward itself — reducing the task to a near-bandit | Convenience | Remove scalar `score` from candidate features (keep raw features); keep a `score_in_obs=True` flag for the ablation arm | Ablation runs in eval harness; test asserts obs dim changes with flag |
| **B6** | 🔴 | README "Sample Evaluation Results" | 2 seeds × 2 episodes per cell; ± values are noise | Rushed eval | Protocol in §4.1: ≥5 training seeds × ≥50 paired eval episodes, IQM + bootstrap CIs | Eval harness refuses to write a README table from < the minimum protocol (hard error, `--force` to override) |
| **B7** | 🟡 | `evaluate_agent.py:264` | Random baseline uses module-global unseeded `random.randint` | Oversight | Per-episode `np.random.default_rng(episode_seed)` | Same seed ⇒ identical Random-policy trajectory (test) |
| **B8** | 🟡 | `train_agent.py:95,106` | Prints "Training PPO on library_v1" and checkpoints as `ppo_library_v1` regardless of `--train-layouts` | Stale strings | Interpolate actual layout list into logs, checkpoint prefix, and run metadata | Grep test on emitted run-config JSON |
| **B9** | 🟡 | repo root | `src/introvertensemble.egg-info/` committed; `racing_ai_plan.md` is an unrelated project doc | Missing ignore rules | `git rm -r --cached`, extend `.gitignore` (`*.egg-info/`, `models/`, `logs/`, `results/`); move racing plan to its own repo or `docs/ideas/` | CI fails on tracked egg-info |
| **B10** | 🟡 | `env.py:50`, `marl_env.py:52` | Asset root via `Path(__file__).resolve().parents[2]` — breaks the moment the package is pip-installed outside the repo | Prototype pathing | Ship layouts as package data; resolve via `importlib.resources`; allow `INTROVERTENSEMBLE_LAYOUT_DIR` env override for custom layouts | Test: `pip install .` into a temp venv, import, `load_layout("library_v1")` succeeds from any CWD |
| **B11** | 🟡 | `env.py:119-134,146-148` | `_reset_layout` has a dead/contradictory fallback branch (`if self.layout_names … else self.layout_names[0]`); layout only re-randomizes when `len(layouts) > 1`; layout choice derives from the sim seed, coupling the two RNG streams | Accretion | Separate RNG streams: `self._layout_rng = np.random.default_rng(seed)` advanced every reset, independent of sim seed; delete dead branch | Test: with 3 layouts and fixed initial seed, 100 resets visit every layout; single-layout path unchanged |
| **B12** | 🟡 | `env.py` overall | Never validated against the Gymnasium API contract; `reset` doesn't call `super().reset(seed=seed)`; obs space is unbounded `(-inf, inf)` while features are mostly [0,1] | Never checked | Run `gymnasium.utils.env_checker.check_env` in CI; call `super().reset(seed=seed)`; tighten Box bounds (helps SB3 normalization too) | `check_env(env)` passes in the test suite |
| **B13** | 🟡 | `env.py:159-197` | Env `step()` duplicates the move-application logic that also exists in `evaluate_agent.py::try_move_focal_to_seat` and in `simulation.py::_move_agent` — three implementations that can drift | Copy-paste | Single `LibrarySimulation.apply_external_move(agent_id, seat_id) -> MoveResult` used by env, eval, and viewer | Deleting either duplicate breaks no tests; move semantics covered once |

---

## 3. Environment & reward redesign (spec)

**Design principle:** the hand-crafted `SeatScorer` is a *behavioral utility model for NPCs*. The *RL reward* must be derived from it deliberately, not inherited wholesale. Introduce `reward_mode: {"legacy", "environment", "shaped"}` on `LibraryEnv` so old results stay reproducible (`legacy`) while new work uses `environment`.

### 3.1 Realized movement cost (fixes half of B4)
On any step where the focal agent relocates:
`reward -= movement_cost = path_cost(old_seat → new_seat) / max_path_cost * MOVE_COST_SCALE`
with `MOVE_COST_SCALE` tuned so one cross-library move costs ≈ 1–2 steps of median seat score. Alternative (richer, optional later): relocation takes `ceil(path_cost / speed)` transit steps during which the agent is unseated and earns 0 — this makes distance a real tradeoff rather than a scalar tax. Start with the scalar tax; it's one line and testable.

### 3.2 Familiarity that can't be farmed
- Increment `seat_history[seat]` only after **K consecutive steps** seated there (K=3 at 15 min/step ≈ 45 min — a defensible "habit" timescale), not on `record_seat` at move time.
- Exponential decay per episode step: `familiarity_value *= λ` (λ ≈ 0.995) so stale visits fade.
- In `reward_mode="environment"`, **exclude** `familiarity` and `dwell_bonus` from the RL reward entirely (they remain in NPC utility). Habit then has to *emerge* from the movement cost + anticipation tradeoff — which is RQ1's whole point — instead of being paid for directly.

### 3.3 Reward decomposition logging
Emit the full `SeatScoreBreakdown` into `info["reward_components"]` every step. This costs nothing (the dataclass already exists) and is the entire measurement instrument for RQ2: plot cumulative `familiarity + dwell_bonus` for the legacy PPO agent vs No-op and the exploit is a picture.

### 3.4 Observation redesign (fixes B5)
- Drop the scalar `score` from candidate features; keep raw features (privacy, noise, crowding ratios, distance, comfort, outlet, preference flags).
- Add forecast-relevant signals so anticipation is *learnable*: current arrival rate, time-of-day encoded as sin/cos (not raw `hour/24`, which puts a discontinuity at midnight), and event pre-cue flags (§3.5).
- Keep `score_in_obs=True` as an ablation flag.

### 3.5 Make anticipation matter (prerequisite for RQ1 being non-trivial)
- **Arrival waves:** replace flat Poisson arrivals with a scheduled rate curve (morning lecture spike, lunch lull, evening exam-season surge) that is *forecastable from time-of-day*.
- **Event pre-cues:** each transient event (Collaboration Burst, Cafe Rush, …) announces itself 2–3 steps early via an observation flag before the noise/crowding actually lands. A myopic greedy cannot use the cue; a learned policy can. The PPO-minus-greedy gap on cue-on vs cue-off runs is the cleanest measurement in the whole project.

### 3.6 Gymnasium hygiene (B12)
`super().reset(seed=seed)`; bounded observation space; `check_env` in CI; `truncated` driven by a config value instead of the magic `1000`.

---

## 4. Evaluation methodology

Follow Agarwal et al. (2021) — this is the difference between "numbers" and "results."

### 4.1 Protocol (minimum publishable unit)
- **Training seeds:** retrain PPO with ≥5 independent seeds. A single trained model is an anecdote; the unit of analysis is the *training run*, not the eval episode.
- **Eval episodes:** ≥50 per (policy × layout), **paired** — every policy sees the identical episode seed list (you already do this; keep it and exploit it with paired statistics).
- **Reporting:** interquartile mean (IQM) with stratified-bootstrap 95% CIs (use the `rliable` package — it's exactly this); per-seed paired differences (PPO − baseline) with Wilcoxon signed-rank + effect size. Never report bare mean ± std again.
- **Success criterion for any "X beats Y" claim in the README:** the 95% CI of the paired difference excludes zero across training seeds.

### 4.2 Baseline suite (post-B3)
| Baseline | Purpose |
|---|---|
| Random | floor |
| No-op | the "do nothing" null — currently your strongest policy, so it stays |
| Greedy (raw-feature heuristic) | myopic policy under the *same observability as PPO* |
| Greedy (true-score) | myopic policy with utility oracle access — separates "knowing the utility" from "anticipating dynamics" |
| Per-step oracle, no cooldown | exact myopic upper bound |
| H-step MPC oracle | exact *anticipatory* upper bound (feasible because utilities and arrival schedule are closed-form — your unique asset) |
| Legacy rule-based focal | continuity with old results |

The interesting science lives in the ordering **greedy(raw) ≤ PPO ≤ MPC**: the left gap is "learning the utility," the right gap is "anticipation still missing."

### 4.3 Ablation grid (each answers one question)
1. `reward_mode legacy vs environment` → quantifies the reward hack (RQ2).
2. `score_in_obs on/off` → how much of PPO's performance was reading the answer key (B5).
3. Event pre-cues on/off → is the PPO-vs-greedy gap anticipation? (RQ1's causal test).
4. Single-layout vs multi-layout training, evaluated on held-out layout → generalization claim (currently unsupported).
5. Arrival waves forecastable vs i.i.d. → RQ1 falsification arm.

### 4.4 Compute budget honesty
80k timesteps is a smoke test, not a training run. Budget ≥1M steps per seed for headline numbers (this env is cheap — CPU-only, minutes-to-hours). Report timesteps, wall-clock, and hardware in the README. Plot sample-efficiency curves (eval return vs timesteps), not just final numbers.

### 4.5 Layout generation
Write `scripts/generate_layout.py` (seeded: rooms → zones → seat grids with jitter → walk graph → feature fields). Then: train on 20 generated layouts, validate on 5, **test once** on 5 never-touched layouts. Four hand-made layouts cannot support a generalization claim; a held-out distribution can.

### 4.6 Results pipeline (kills B1 permanently)
Every eval run writes `results/<run_id>/` containing: resolved config, git SHA, per-episode CSV, summary JSON, and rliable plots. `scripts/make_results_table.py` renders the README results section **from the latest summary JSON between `<!-- results:begin/end -->` markers**. Hand-edited results numbers become impossible by construction.

---

## 5. Software engineering standards

### 5.1 Toolchain
- **ruff** (lint + format, replaces black/isort/flake8), **mypy --strict** on `src/` (the codebase is already typed dataclasses — strict is within reach), **pytest + pytest-cov** with a ratcheting coverage floor (start at current %, only allow it to rise), **pre-commit** running all three.
- **uv** for env + lockfile (`uv.lock` committed) so "works on my machine" is reproducible.

### 5.2 CI (GitHub Actions)
Single workflow, Python 3.11 + 3.12 matrix:
1. `ruff check` + `ruff format --check`
2. `mypy src/`
3. `pytest` (unit + `check_env` + the B-register acceptance tests)
4. **Smoke train:** 2 048 PPO timesteps must complete and save a model (~30 s)
5. **Smoke eval:** 2 episodes × all policies on `library_v1` must produce a well-formed summary JSON
6. Guards: fail on tracked `egg-info`/`models/`/`logs/`; fail if README results markers were hand-edited (checksum vs committed summary JSON)

### 5.3 Packaging
Layouts as package data via `importlib.resources` (B10); proper `[project.scripts]` entry points (`ie-train`, `ie-eval`, `ie-view`) so `run.sh` becomes a thin convenience wrapper instead of the API.

### 5.4 Configuration & experiment tracking
- One frozen `ExperimentConfig` dataclass (env + reward_mode + PPO hyperparams + eval protocol), serialized to YAML; CLI flags override fields. Every run directory contains the resolved config + git SHA + dirty-tree flag.
- Tracking: TensorBoard is fine to start; add Weights & Biases when you begin the ≥5-seed sweeps (free tier is enough, and sweep comparison by hand is where solo projects silently rot).

### 5.5 Testing strategy (beyond the bug-register tests)
- **Property tests** (hypothesis): score monotonicity (↑privacy ⇒ ↑score, ↑crowding ⇒ ↓score, all else fixed); occupancy invariants (never two agents on one seat; release+occupy conserves counts).
- **Golden regression test:** fixed seed + fixed config ⇒ exact expected episode trajectory hash. Any unintentional physics change fails loudly.
- **Statistical smoke test:** trained smoke-model ≥ Random with p < 0.05 over 20 paired episodes (catches "training silently broken").

### 5.6 Repo & process hygiene
- `docs/` for `LIBRARY_MARL_RESEARCH_PLAN.md` + this file; `racing_ai_plan.md` out of the repo (B9).
- **Conventional commits** (`feat:`, `fix:`, `exp:` for experiment runs) — commits named "refact"/"test" make the history unusable as a lab notebook, and the git history *is* your lab notebook.
- `decisions.md` (ADR-lite): every irreversible modeling choice (reward composition, familiarity semantics, oracle definition) gets 3 lines — context, decision, consequence.
- `CHANGELOG.md` keyed to experiment-relevant changes, so any results file can be interpreted against the code that produced it.

---

## 6. Roadmap

| Phase | Scope | Effort | Acceptance criteria (gate to next phase) |
|---|---|---|---|
| **0 — Hygiene** | B9, ruff+mypy+pre-commit, CI skeleton, uv lock, conventional commits from now on | 1–2 days | CI green on main; no tracked artifacts |
| **1 — Correctness** | B2, B3, B7, B8, B10, B11, B12, B13 + their acceptance tests | 3–5 days | `check_env` passes; oracle strictly ≥ all policies per paired seed; oracle ≢ greedy; all register tests green |
| **2 — Reward & obs redesign** | §3 complete: `reward_mode`, realized move cost, dwell-gated decaying familiarity, obs without score, pre-cues, arrival waves, decomposition logging | ~1 week | Ping-pong exploit test passes (scripted farmer < No-op everywhere); legacy mode byte-reproduces old behavior |
| **3 — Eval harness** | §4: paired protocol, rliable stats, baseline suite, results pipeline, auto-README table (retires B1, B6) | ~1 week | README table generated, not typed; harness rejects under-powered runs |
| **4 — RQ2 writeup** | Legacy-env reward-hacking case study: decomposition plots, before/after fixes | 3–4 days | Blog post / workshop draft with figures; old model archived with its config |
| **5 — Retrain & RQ1** | ≥5 seeds × ≥1M steps in `environment` mode; full ablation grid (§4.3) | ~1–2 weeks (mostly compute) | Every README claim backed by a paired CI excluding zero — including negative results, stated plainly |
| **6 — Generalization (RQ4)** | Layout generator, 20/5/5 split, domain-randomization runs | ~1 week | Held-out-distribution numbers replace the current "generalization" claim |
| **7 — MARL (RQ3)** | `LibraryParallelEnv` + parameter-shared PPO; congestion metrics: price of anarchy, zone Gini, territory stability, empirical Nash check | 2–3 weeks | Exact unilateral-deviation check implemented (utilities are closed-form — use the asset); welfare vs. central optimum reported |
| **8 — Publication** | arXiv preprint or serious technical blog + polished repo (env registered on PyPI as a Gymnasium/PettingZoo env is a legitimate artifact contribution by itself) | 1–2 weeks | External-reader review: someone else reproduces one table from a fresh clone using only the README |

### Risks & de-scoping
- **Risk:** after fixes, PPO ties greedy everywhere → **that's RQ1 answered negatively + RQ2 intact.** A clean negative with exact oracles is more credible than the current inflated positive. Publish it.
- **Risk:** scope explosion. **Rule:** nothing from Phase 7 starts before Phase 5's gate passes. The MARL env is scaffolding, not a commitment.
- **Risk:** solo-project entropy. The CI gates in §5.2 are what keep standards from decaying when motivation dips — build them in Phase 0, not "later."

---

## Appendix A — Immediate one-day quick wins (can precede Phase 0)
1. Fix the README: un-bold PPO where it loses, correct the "Mean Steps" header, add "⚠ preliminary — 4 episodes/cell" to the table.
2. `git rm -r --cached src/introvertensemble.egg-info && echo '*.egg-info/' >> .gitignore`.
3. Seed the Random baseline (B7) — 2 lines.
4. Fix the stale training log strings (B8) — 3 lines.
5. Add `info["reward_components"]` logging (§3.3) — ~10 lines, unlocks the RQ2 plots immediately.
