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

## Project docs

See [LIBRARY_MARL_RESEARCH_PLAN.md](LIBRARY_MARL_RESEARCH_PLAN.md) for the full research roadmap, terminology, and planned multi-layout / multi-agent extensions.
