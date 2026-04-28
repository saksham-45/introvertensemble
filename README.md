# introvertensemble

`introvertensemble` is a Python simulation package for library seat-selection under multi-agent dynamics. The current repo is not a trained RL system yet; it is the environment and behavioral scaffold: typed layout assets, seat scoring, event perturbations, focal-agent observations, and stepwise simulation over a shared seat map.

The bundled `library_v1` layout defines `74` seats, `9` zones, `4` entrances, and a walk graph used for entrance-to-seat path cost. Agents score seats against feature layers such as privacy, noise, and crowding, while the simulation updates arrivals, departures, reseating, and optional transient zone events. The focal introvert agent has stronger penalties for crowding/disruption and supports habit-biased reseating logic.

## Package Surface

- `introvertensemble.load_layout(...)`: loads the JSON layout spec under `assets/layouts/library_v1/`.
- `introvertensemble.LibraryWorld`: exposes seat occupancy, feature lookup, graph path cost, and spawn-window state.
- `introvertensemble.LibrarySimulation`: advances the environment one step at a time and manages agent behavior.
- `introvertensemble.ObservationBuilder`: constructs focal-agent candidate-seat observations.
- `introvertensemble.run_episode(...)`: executes a rollout and returns aggregate episode metrics.

## Local Usage

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/run_headless_simulation.py
python scripts/view_simulation.py
python -m unittest discover -s tests
```

`run_headless_simulation.py` prints per-step occupancy, profile counts, and zone load. `view_simulation.py` launches the `pygame` viewer for the same world model.
