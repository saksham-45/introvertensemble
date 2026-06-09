# introvertensemble

`introvertensemble` is a Python simulation and reinforcement-learning package for library seat selection under multi-agent dynamics. A focal introvert agent learns to choose and re-choose seats while background agents arrive, depart, and crowd the space.

The bundled `library_v1` layout defines **74 seats**, **9 zones**, **4 entrances**, and a walk graph used for entrance-to-seat path cost. Agents score seats against feature layers such as privacy, noise, and crowding, while the simulation updates arrivals, departures, reseating, and optional transient zone events.

## What works today

- **Library simulator** with typed layout assets, seat scoring, events, and rule-based background agents
- **Gymnasium RL environment** (`LibraryEnv`) for training a single focal agent against scripted NPCs
- **PPO training** via Stable-Baselines3 (`./run.sh train`) with support for **multi-layout domain randomization**
- **Evaluation suite** comparing PPO against random, no-op, greedy, perfect-info oracle, and rule-based focal baselines (`./run.sh eval`) with **seed sweeps, cross-layout testing, and result exporting**
- **Pygame viewer** for watching the trained RL agent act in the library (`./run.sh view`)

On 50 evaluation episodes, the original single-layout PPO agent (trained on library_v1) achieves **~43.4 total reward** vs **~38.3** for no-op and **~0.8** for the hand-tuned rule-based focal agent. With multi-layout training, the agent learns generalized policies that perform across different library configurations.

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
- **Observation**: 141-dim vector (agent state, current seat, top candidate seats, time/events)
- **Actions**: 11 discrete (stay, 5 nearby candidates, 5 global candidates)
- **Reward**: focal agent seat score each step
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

## Sample Evaluation Results (2 seeds, 2 episodes)

| Layout | Policy | Mean Reward ± Std | Mean Steps ± Std | Mean Moves ± Std | Final Score ± Std |
|--------|--------|-------------------|------------------|------------------|-------------------|
| library_v1 | Random | -2.84 ± 13.59 | -0.12 ± 0.57 | 5.75 ± 0.43 | -3.44 ± 0.94 |
| library_v1 | No-op (stay put) | 39.68 ± 2.36 | 1.65 ± 0.10 | 0.00 ± 0.00 | 1.22 ± 0.00 |
| library_v1 | Greedy (top-10 candidates) | 29.53 ± 18.42 | 1.23 ± 0.77 | 0.50 ± 0.87 | 0.38 ± 1.46 |
| library_v1 | Perfect-info oracle | 29.53 ± 18.42 | 1.23 ± 0.77 | 0.50 ± 0.87 | 0.38 ± 1.46 |
| library_v1 | **Trained PPO** | **45.43 ± 5.12** | **1.89 ± 0.21** | **2.50 ± 0.50** | **1.54 ± 0.14** |
| library_v2_riverside | Random | 21.57 ± 7.42 | 0.90 ± 0.31 | 5.00 ± 0.00 | -2.21 ± 0.27 |
| library_v2_riverside | No-op (stay put) | 76.63 ± 0.74 | 3.19 ± 0.03 | 0.00 ± 0.00 | 3.07 ± 0.22 |
| library_v2_riverside | Greedy (top-10 candidates) | 76.63 ± 0.74 | 3.19 ± 0.03 | 0.00 ± 0.00 | 3.07 ± 0.22 |
| library_v2_riverside | Perfect-info oracle | 76.63 ± 0.74 | 3.19 ± 0.03 | 0.00 ± 0.00 | 3.07 ± 0.22 |
| library_v2_riverside | **Trained PPO** | **51.26 ± 16.97** | **2.14 ± 0.71** | **2.75 ± 0.43** | **0.46 ± 1.51** |
| library_v3_courtyard | Random | 26.70 ± 3.05 | 1.11 ± 0.13 | 5.50 ± 0.50 | -0.42 ± 1.00 |
| library_v3_courtyard | No-op (stay put) | 78.18 ± 1.15 | 3.26 ± 0.05 | 0.00 ± 0.00 | 3.19 ± 0.00 |
| library_v3_courtyard | Greedy (top-10 candidates) | 78.18 ± 1.15 | 3.26 ± 0.05 | 0.00 ± 0.00 | 3.19 ± 0.00 |
| library_v3_courtyard | Perfect-info oracle | 78.18 ± 1.15 | 3.26 ± 0.05 | 0.00 ± 0.00 | 3.19 ± 0.00 |
| library_v3_courtyard | **Trained PPO** | **52.83 ± 12.12** | **2.20 ± 0.51** | **3.50 ± 0.50** | **1.04 ± 0.23** |
| library_v4_atrium | Random | 27.69 ± 7.65 | 1.15 ± 0.32 | 5.25 ± 0.43 | -1.02 ± 1.28 |
| library_v4_atrium | No-op (stay put) | 77.14 ± 0.76 | 3.21 ± 0.03 | 0.00 ± 0.00 | 3.10 ± 0.00 |
| library_v4_atrium | Greedy (top-10 candidates) | 77.14 ± 0.76 | 3.21 ± 0.03 | 0.00 ± 0.00 | 3.10 ± 0.00 |
| library_v4_atrium | Perfect-info oracle | 77.14 ± 0.76 | 3.21 ± 0.03 | 0.00 ± 0.00 | 3.10 ± 0.00 |
| library_v4_atrium | **Trained PPO** | **57.76 ± 22.50** | **2.41 ± 0.94** | **3.25 ± 1.30** | **2.23 ± 1.73** |

## Project docs

See [LIBRARY_MARL_RESEARCH_PLAN.md](LIBRARY_MARL_RESEARCH_PLAN.md) for the full research roadmap, terminology, and planned multi-layout / multi-agent extensions.
