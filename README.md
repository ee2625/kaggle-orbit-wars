# Orbit Wars Kaggle Bot

Starter repo for the Kaggle Orbit Wars competition.

## Quick Start

Use Python 3.12 or another Python version that can install
`kaggle-environments>=1.28.0`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/run_local.py --episodes 3
```

Submit the standalone agent:

```bash
kaggle competitions submit orbit-wars -f main.py -m "baseline v1"
```

## Repo Layout

- `main.py` - Kaggle submission entrypoint with `agent(obs)`.
- `scripts/run_local.py` - local match runner against Kaggle's built-in agents.
- `tests/test_agent_smoke.py` - fast checks that the agent returns legal-looking moves.

## Strategy Notes

The first bot is intentionally conservative:

- captures affordable high-production planets first
- skips direct shots through the sun
- avoids paths that immediately collide with another visible planet
- keeps a small ship reserve on each owned planet
- uses simple forward prediction for orbiting planets and comets when possible

Good next upgrades:

- add incoming fleet threat accounting
- simulate capture timing and production payoff
- coordinate multi-planet attacks
- tune separate neutral, enemy, and comet policies from replay data
- build a replay parser for ladder diagnostics
