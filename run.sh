#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh  –  introvertensemble project launcher
# Always uses .venv so imports never break.
# Usage:
#   ./run.sh sim          -> headless simulation (text output)
#   ./run.sh view         -> pygame viewer (trained PPO agent)
#   ./run.sh view-rule    -> pygame viewer (rule-based focal agent)
#   ./run.sh test         -> run all unit tests
#   ./run.sh best         -> print best seat in current layout
#   ./run.sh train        -> train PPO agent on library_v1
#   ./run.sh eval         -> compare trained agent vs baselines
# ---------------------------------------------------------------------------

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Virtual environment not found. Setting it up now..."
    python3 -m venv "$ROOT/.venv"
    "$ROOT/.venv/bin/pip" install -e "$ROOT" 2>&1 | tail -5
    echo "Done."
fi

CMD="${1:-sim}"

ensure_rl_deps() {
    if ! "$PYTHON" -c "import stable_baselines3" 2>/dev/null; then
        echo "Installing RL dependencies (stable-baselines3, torch)..."
        "$ROOT/.venv/bin/pip" install -e "$ROOT[rl]" 2>&1 | tail -8
    fi
}

case "$CMD" in
    sim)
        "$PYTHON" "$ROOT/scripts/run_headless_simulation.py"
        ;;
    view)
        ensure_rl_deps
        "$PYTHON" "$ROOT/scripts/view_rl_simulation.py" "${@:2}"
        ;;
    view-rule)
        "$PYTHON" "$ROOT/scripts/view_simulation.py"
        ;;
    test)
        "$PYTHON" -m unittest discover -s "$ROOT/tests"
        ;;
    best)
        "$PYTHON" "$ROOT/scripts/verify_best_seat.py"
        ;;
    train)
        ensure_rl_deps
        "$PYTHON" "$ROOT/scripts/train_agent.py" "${@:2}"
        ;;
    eval)
        ensure_rl_deps
        "$PYTHON" "$ROOT/scripts/evaluate_agent.py" "${@:2}"
        ;;
    gen-layouts)
        "$PYTHON" "$ROOT/scripts/generate_layouts.py" "${@:2}"
        ;;
    train-gen)
        ensure_rl_deps
        "$PYTHON" "$ROOT/scripts/train_generalization.py" "${@:2}"
        ;;
    eval-gen)
        ensure_rl_deps
        "$PYTHON" "$ROOT/scripts/evaluate_generalization.py" "${@:2}"
        ;;
    *)
        echo "Unknown command: $CMD"
        echo "Usage: ./run.sh [sim|view|view-rule|test|best|train|eval|gen-layouts|train-gen|eval-gen]"
        exit 1
        ;;
esac
