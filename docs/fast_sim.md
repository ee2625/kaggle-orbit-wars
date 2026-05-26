# Orbit Fast Simulator

This is the first piece of the "big unlock": a local simulator with a direct
Python state transition and parity tests against the official Kaggle
interpreter.

## What Exists

- `orbit_fast/sim.py`
  - `GameState.initialize(num_agents, seed, ...)`
  - `GameState.from_observation(obs, ...)`
  - `state.step(actions)`
  - `state.observe(player)`
  - `state.scores()`
- `tests/test_fast_sim_parity.py`
  - checks combat, launch/movement, sun removal, planet-first collision order,
    and a seeded empty rollout through comet spawn against the official
    interpreter.
- `scripts/benchmark_fast_sim.py`
  - compares empty-turn transition throughput against the raw official
    interpreter.
- `scripts/fast_run_local.py`
  - runs local agents directly through `GameState`, bypassing the full Kaggle
    environment wrapper.
- `scripts/extract_replay_transitions.py`
  - converts replay JSON files, replay directories, or `episodes.tar.gz`
    bundles into JSONL transitions for diagnostics or imitation learning.

## Commands

```bash
.venv/bin/python -m unittest tests.test_fast_sim_parity -v
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/benchmark_fast_sim.py --episodes 50 --steps 120 --players 4
.venv/bin/python scripts/fast_run_local.py --agent main.py --opponent random --episodes 3 --seed 42
.venv/bin/python scripts/league_backtest.py --backend fast --agent main.py --agent random --games 4 --players 2
.venv/bin/python scripts/extract_replay_transitions.py backtests/top_replays_20260504 --winner-only --out-jsonl backtests/replay_training/transitions.jsonl
```

## Current Throughput Snapshot

On this machine, an empty 4-player 120-step benchmark is around 2.9K raw
state transitions per second. That is not yet a JAX/RL-grade simulator; it is
the exactness layer. The next speed jump comes from using this contract to
remove Kaggle wrapper overhead in local leagues, then porting the same parity
tests to a vectorized/JAX implementation.

## Next ML Track

1. Feed public top replay datasets into a trajectory extractor.
2. Use `GameState` parity to label observations with legal action masks and
   post-action outcomes.
3. Train a small imitation policy first, not full PPO.
4. Only after imitation is alive, port `GameState.step` to a vectorized backend
   and re-run the same parity tests frame by frame.
