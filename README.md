# introvertensemble

`introvertensemble` is a Python simulation and reinforcement-learning package for library seat selection under multi-agent dynamics. A focal introvert agent learns to choose and re-choose seats while background agents arrive, depart, and crowd the space.

The bundled `library_v1` layout defines **74 seats**, **9 zones**, **4 entrances**, and a walk graph used for entrance-to-seat path cost. Agents score seats against feature layers such as privacy, noise, and crowding, while the simulation updates arrivals, departures, reseating, and optional transient zone events.

## What works today

- **Library simulator** with typed layout assets, seat scoring, events, and rule-based background agents
- **Gymnasium RL environment** (`LibraryEnv`) for training a single focal agent against scripted NPCs
- **PPO training** via Stable-Baselines3 (`./run.sh train`)
- **Evaluation suite** comparing PPO against random, no-op, greedy, perfect-info oracle, and rule-based focal baselines (`./run.sh eval`)
- **Pygame viewer** for watching the trained RL agent act in the library (`./run.sh view`)

On 50 evaluation episodes, the trained PPO agent currently achieves **~43.4 total reward** vs **~38.3** for no-op and **~0.8** for the hand-tuned rule-based focal agent.

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
./run.sh train         # train PPO on library_v1
./run.sh eval          # compare PPO vs baselines
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
| `./run.sh train` | Train PPO agent (`--timesteps 80000` by default) |
| `./run.sh eval` | Evaluate all policies (`--episodes 50` recommended) |
| `./run.sh best` | Print highest-scoring seat for empty library |
| `./run.sh test` | Run unit tests |

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
- **Saved model**: `models/ppo_library_v1.zip`
- **Logs**: `logs/ppo_library_v1/` (TensorBoard)

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

## Project docs

See [LIBRARY_MARL_RESEARCH_PLAN.md](LIBRARY_MARL_RESEARCH_PLAN.md) for the full research roadmap, terminology, and planned multi-layout / multi-agent extensions.
