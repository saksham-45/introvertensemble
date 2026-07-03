# introvertensemble

`introvertensemble` is a Python simulation and reinforcement-learning package for library seat selection under multi-agent dynamics. A focal introvert agent learns to choose and re-choose seats while background agents arrive, depart, and crowd the space.

The bundled `library_v1` layout defines **74 seats**, **9 zones**, **4 entrances**, and a walk graph used for entrance-to-seat path cost. Agents score seats against feature layers such as privacy, noise, and crowding, while the simulation updates arrivals, departures, reseating, and optional transient zone events.

## What works today

- **Library simulator** with typed layout assets, seat scoring, events, and rule-based background agents
- **Gymnasium RL environment** (`LibraryEnv`) for training a single focal agent against scripted NPCs
- **PPO training** via Stable-Baselines3 (`./run.sh train`) with support for **multi-layout domain randomization**
- **Evaluation suite** comparing policies against random, no-op, greedy, and a myopic best-seat reference (`./run.sh eval`) with **seed sweeps, cross-layout testing, paired 95% CIs, and result exporting**
- **Pygame viewer** for watching the trained RL agent act in the library (`./run.sh view`)

> **Corrected methodology (2026-07).** Earlier versions of this README reported
> PPO as the top policy on layouts where it actually lost to no-op, using a
> gameable reward (familiarity/dwell farmable via cost-free moves), a leaked
> score in the observation, a degenerate "oracle" identical to greedy, and 4
> episodes/cell. Those defects are fixed: relocation is charged, habit terms are
> excluded from the RL reward, the score is withheld from observations, spawn is
> randomized, and the baselines are honest. The results table below is generated
> directly from an evaluation export (see [docs/RESEARCH_ENGINEERING_PLAN.md](docs/RESEARCH_ENGINEERING_PLAN.md)),
> so it can no longer disagree with the numbers. PPO rows appear once a training
> run on a supported Python (3.11–3.12, for Stable-Baselines3) is committed.

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[rl]"
```

Or use the project launcher (creates `.venv` automatically):

```bash
./run.sh test          # run unit tests
./run.sh sim           # headless text simulation
./run.sh train         # train PPO agent (default: library_v1, use --train-layouts for multiple)
./run.sh eval          # compare PPO vs baselines (default: library_v1, use --eval-layouts for multiple)
./run.sh view          # pygame viewer with trained RL agent
```

Train first before viewing:

```bash
./run.sh train
./run.sh view
```

## Commands

| Command | Description |
|---------|-------------|
| `./run.sh sim` | Headless simulation with focal agent + background NPCs |
| `./run.sh view` | Pygame viewer driven by trained PPO policy |
| `./run.sh view-rule` | Pygame viewer with rule-based focal agent (legacy) |
| `./run.sh train` | Train PPO agent (`--timesteps 80000` by default, use `--train-layouts` for multiple layouts) |
| `./run.sh eval` | Evaluate all policies (`--episodes 50` recommended, use `--eval-layouts` for multiple layouts, `--num-seeds` for seed sweeps) |
| `./run.sh best` | Print highest-scoring seat for empty library |
| `./run.sh test` | Run unit tests |

### Training Options
- `--train-layouts`: List of layout names to train on (e.g., `library_v1 library_v2_riverside`)
- `--val-layouts`: List of layout names to validate on (default: `library_v1`)
- `--timesteps`: Total training timesteps (default: 80000)
- `--session-steps`: Focal agent session length in steps (default: 24)

### Evaluation Options
- `--eval-layouts`: List of layout names to evaluate on (default: `library_v1`)
- `--num-seeds`: Number of different seeds to sweep for each layout (default: 1)
- `--episodes`: Episodes per policy (default: 10)
- `--export-csv`: Export results to CSV file
- `--export-json`: Export results to JSON file
- `--model-path`: Path to trained PPO model (default: `models/ppo_multi_layout.zip`)

## Package surface

- `introvertensemble.load_layout(...)`: load JSON layout spec from `assets/layouts/`
- `introvertensemble.LibraryWorld`: seat occupancy, feature lookup, graph path cost
- `introvertensemble.LibrarySimulation`: stepwise multi-agent simulation
- `introvertensemble.LibraryEnv`: Gymnasium environment for RL training
- `introvertensemble.ObservationBuilder`: focal-agent candidate-seat observations
- `introvertensemble.run_episode(...)`: rollout metrics for scripted simulations

## RL setup

- **Algorithm**: PPO (Stable-Baselines3)
- **Observation**: 141-dim vector (agent state, current seat, top candidate seats, time/events). The true seat score is withheld by default (`score_in_obs=False`) so the task is not a disguised bandit; pass `--score-in-obs` for the ablation.
- **Actions**: 11 discrete (stay, 5 nearby candidates, 5 global candidates)
- **Reward** (`--reward-mode`, default `environment`): the focal seat's environment score each step, **excluding** farmable familiarity/dwell bonuses, **minus** the realized movement cost on any step the agent relocates. `legacy` reproduces the original raw-score reward for reward-hacking studies.
- **Saved model**: `models/ppo_multi_layout.zip` (when using multi-layout training)
- **Logs**: `logs/ppo_multi_layout/` (TensorBoard)

```bash
tensorboard --logdir logs/
```

## Viewer controls

- **Space**: play / pause
- **N**: single step
- **Up / Down**: simulation speed
- **Left / Right**: cycle feature layer overlay
- **G / L / I**: toggle walk graph, labels, seat ids
- **Esc**: quit

The focal agent is highlighted in gold with a **YOU** label.

## Layout Generalization and Evaluation Hardening

This project now supports training PPO agents across multiple layouts to prevent overfitting and promote generalization. The evaluation system has been hardened to provide statistically significant performance metrics across layouts and random seeds.

### Key Features
- **Domain Randomization**: Train on multiple layouts simultaneously to learn invariant policies
- **Seed Sweeps**: Evaluate across multiple random seeds to reduce variance in results
- **Cross-Layout Testing**: Assess generalization by evaluating on layouts not seen during training
- **Result Exporting**: Save detailed evaluation results to CSV/JSON for further analysis

### Example Usage

Train on three layouts:
```bash
./run.sh train --timesteps 100000 --train-layouts library_v1 library_v2_riverside library_v3_courtyard
```

Evaluate on all four layouts with 5 seed sweeps and 10 episodes each:
```bash
./run.sh eval --eval-layouts library_v1 library_v2_riverside library_v3_courtyard library_v4_atrium --num-seeds 5 --episodes 10 --export-csv results.csv
```

Evaluate generalization to an unseen layout (if library_v4_atrium was not used in training):
```bash
./run.sh eval --eval-layouts library_v4_atrium --num-seeds 3 --episodes 5
```

## Evaluation results

The table below is **generated**, not hand-written. Produce it with:

```bash
# 1. run a properly-powered evaluation (>= 30 episodes/cell recommended)
./run.sh eval --episodes 50 --num-seeds 5 \
    --eval-layouts library_v1 library_v2_riverside library_v3_courtyard library_v4_atrium \
    --export-json results.json
# 2. render it into this README between the results markers
python scripts/make_results_table.py --results results.json --readme README.md
```

`make_results_table.py` refuses to write an under-powered table (fewer than
`--min-episodes` per cell) unless `--force` is passed, so a smoke test can never
be mistaken for a result. PPO rows appear automatically once a trained model is
present. Interpretation note: with **random spawn**, No-op is no longer a strong
policy (it is stuck at an arbitrary seat), which is what makes the comparison
discriminative — a useful policy must actually *find* good seats.

<!-- results:begin -->

Reward mode: `environment` · spawn: `random` · 30–30 episodes/cell · total reward, mean ± 95% CI. Higher is better.

| Layout | Policy | Total reward (mean ± 95% CI) | Moves/ep |
|--------|--------|------------------------------|----------|
| library_v1 | Best-seat (myopic, all seats) | 17.27 ± 6.01 | 1.43 |
| library_v1 | Greedy (candidate actions) | 23.18 ± 2.75 | 0.73 |
| library_v1 | No-op (stay put) | -25.78 ± 13.91 | 0.00 |
| library_v1 | Random | -28.24 ± 5.34 | 5.53 |
| library_v2_riverside | Best-seat (myopic, all seats) | 64.99 ± 0.74 | 1.07 |
| library_v2_riverside | Greedy (candidate actions) | 64.52 ± 0.64 | 0.87 |
| library_v2_riverside | No-op (stay put) | 1.37 ± 13.39 | 0.00 |
| library_v2_riverside | Random | -3.56 ± 5.67 | 5.40 |
| library_v3_courtyard | Best-seat (myopic, all seats) | 64.53 ± 0.51 | 1.03 |
| library_v3_courtyard | Greedy (candidate actions) | 63.36 ± 0.66 | 0.77 |
| library_v3_courtyard | No-op (stay put) | -8.03 ± 17.53 | 0.00 |
| library_v3_courtyard | Random | -3.69 ± 7.48 | 5.23 |
| library_v4_atrium | Best-seat (myopic, all seats) | 64.67 ± 0.49 | 1.03 |
| library_v4_atrium | Greedy (candidate actions) | 62.74 ± 1.72 | 0.80 |
| library_v4_atrium | No-op (stay put) | -9.83 ± 17.11 | 0.00 |
| library_v4_atrium | Random | -0.69 ± 6.37 | 5.33 |

<!-- results:end -->

## Generalist agent on procedurally generated layouts

The strongest agent is trained not on the 4 bundled layouts but on a **pool of
procedurally generated libraries** with **domain randomization** over people and
settings, then evaluated on a **held-out set of layouts it never saw** — the
ProcGen generalization protocol (see [docs/TRAINING_BEST_PRACTICES.md](docs/TRAINING_BEST_PRACTICES.md)).

```bash
# 1. generate disjoint train / val / test layout pools (reproducible from seeds)
./run.sh gen-layouts --n-train 128 --n-val 16 --n-test 16

# 2. train PPO with VecNormalize + domain randomization; best model chosen on val
./run.sh train-gen --timesteps 400000 --n-envs 6 --seed 0 \
    --out models/ppo_generalist/seed_0/ppo_generalist

# 3. evaluate on the never-seen TEST pool vs baselines; measure the gen. gap
./run.sh eval-gen --model models/ppo_generalist/seed_0/best_model.zip \
    --vecnormalize models/ppo_generalist/seed_0/vecnormalize.pkl \
    --episodes 60 --export-json results/gen_seed_0.json

# 4. aggregate across seeds with IQM + bootstrap CIs (rliable methodology)
python scripts/aggregate_seeds.py results/gen_seed_*.json --print
```

The **generalization gap** (return on seen train layouts minus unseen test
layouts) is the headline number — a small gap means the policy learned
seat-selection *competence*, not layout memorization.

**Verdict (current run).** Trained on 128 generated layouts with domain
randomization, the PPO agent **generalizes**: on 16 never-seen test layouts it
scores ~48–50 (IQM), a **~2–5 point gap** below its seen-layout score — small,
so it learned transferable competence rather than memorizing maps. It **beats
No-op and Random by ~65 points** (they cannot escape a bad random spawn), and it
lands **statistically on par with, marginally below, a score-informed greedy** —
despite the true seat score being *withheld* from its observations. That is the
expected, honest ceiling: with this reward the task is close to greedy-solvable,
so a sound agent converges toward greedy rather than blowing past it; matching it
from raw features alone is the real result. *Limitation:* the table below
aggregates **2 training seeds** (compute-limited); ≥3–5 is preferable and can be
added by training more seeds and re-running `aggregate_seeds.py`.

<!-- generalist:begin -->

Aggregated over **2 training seed(s)**, environment reward, random spawn. Metric: **IQM** (interquartile mean) with 95% bootstrap CI. Higher is better.

| Split | Policy | IQM total reward [95% CI] | Mean |
|-------|--------|---------------------------|------|
| test_pool (unseen) | Best-seat (myopic, all seats) | 49.35 [47.91, 50.95] | 52.31 |
| test_pool (unseen) | Greedy (candidate actions) | 49.23 [47.75, 50.75] | 51.66 |
| test_pool (unseen) | No-op (stay put) | -18.19 [-33.01, -2.74] | -10.63 |
| test_pool (unseen) | Random | -17.86 [-22.92, -12.75] | -17.26 |
| test_pool (unseen) | Trained PPO | 47.91 [46.84, 48.96] | 50.13 |
| train_pool (seen) | Best-seat (myopic, all seats) | 51.23 [48.93, 55.95] | 55.73 |
| train_pool (seen) | Greedy (candidate actions) | 50.22 [48.57, 54.04] | 54.44 |
| train_pool (seen) | No-op (stay put) | -26.43 [-33.72, -13.40] | -13.20 |
| train_pool (seen) | Random | -32.01 [-35.94, -28.27] | -31.76 |
| train_pool (seen) | Trained PPO | 49.79 [48.33, 52.83] | 54.06 |

<!-- generalist:end -->

## Project docs

- [docs/RESEARCH_ENGINEERING_PLAN.md](docs/RESEARCH_ENGINEERING_PLAN.md) — research framing, bug register, roadmap
- [docs/TRAINING_BEST_PRACTICES.md](docs/TRAINING_BEST_PRACTICES.md) — cited PPO/PCG/eval recipe
- [docs/LIBRARY_MARL_RESEARCH_PLAN.md](docs/LIBRARY_MARL_RESEARCH_PLAN.md) — original MARL roadmap and terminology
- [decisions.md](decisions.md) — ADR-lite log of modeling choices
