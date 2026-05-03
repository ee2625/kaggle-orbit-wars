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
python scripts/run_local.py --episodes 3 --opponents random random random
python scripts/backtest.py --episodes 10 --out-dir backtests
python scripts/league_backtest.py --agent main.py --agent random --games 20 --out-dir backtests
python scripts/collect_feedback.py --latest-submissions 2 --download-logs
```

Submit the standalone agent:

```bash
kaggle competitions submit orbit-wars -f main.py -m "baseline v1"
```

## Repo Layout

- `main.py` - Kaggle submission entrypoint with `agent(obs)`.
- `submissions/` - stable snapshots of submitted/candidate agents for rollback and comparisons.
- `scripts/run_local.py` - local match runner against Kaggle's built-in agents.
- `scripts/backtest.py` - repeatable batch backtester with summaries and JSON/CSV output.
- `scripts/league_backtest.py` - local ladder simulator with Kaggle-style Gaussian skill ratings.
- `scripts/analyze_replay.py` - replay diagnostics for expansion timing, launches, and fleet losses.
- `scripts/collect_feedback.py` - Kaggle episode collector that downloads new replays/logs and writes a feedback summary.
- `scripts/evaluate_replay_suite.py` - counterfactual replay suite that tests `main.py` on collected maps against recorded opponent actions.
- `scripts/weighted_replay_score.py` - weights replay-suite results toward newer submissions and recent ladder feedback.
- `tests/test_agent_smoke.py` - fast checks that the agent returns legal-looking moves.
- `docs/rules_checklist.md` - practical compliance notes from the competition rules.
- `references/` - local reference file index; raw uploads are kept ignored.

## Backtesting

Run a 2-player sweep against `random`:

```bash
python scripts/backtest.py --episodes 50 --seed 100 --out-dir backtests
```

Run a 4-player sweep and rotate our bot through all player slots:

```bash
python scripts/backtest.py --episodes 40 --seed 200 --opponents random random random --rotate-seat --out-dir backtests
```

Compare multiple bot files:

```bash
python scripts/backtest.py --agent main.py --agent path/to/other_bot.py --episodes 25 --out-dir backtests
```

Run a local ladder that scores like the Kaggle simulation leaderboard: each bot
starts at `mu=600`, has Gaussian uncertainty `sigma`, and rating updates use only
win/tie/loss ordering, not final ship margin.

```bash
python scripts/league_backtest.py \
  --agent main.py \
  --agent backtests/agents/submitted_133182e.py \
  --agent random \
  --games 30 \
  --schedule ladder \
  --out-dir backtests/league
```

For 4-player local games, provide at least four agents:

```bash
python scripts/league_backtest.py \
  --players 4 \
  --agent main.py \
  --agent backtests/agents/submitted_133182e.py \
  --agent random \
  --agent random \
  --games 20 \
  --out-dir backtests/league_4p
```

Analyze downloaded Kaggle replays to find failure modes:

```bash
python scripts/analyze_replay.py backtests/replays/v2_public/episode-75784648-replay.json
```

Collect new Kaggle feedback for the latest two submissions without submitting a
new bot:

```bash
python scripts/collect_feedback.py \
  --latest-submissions 2 \
  --download-logs \
  --out-dir backtests/kaggle_feedback
```

The collector is incremental. It keeps `manifest.json`, writes
`summary.md`, and skips replay/log files that have already been downloaded.

Run the current bot against collected replay maps with recorded opponent actions:

```bash
.venv/bin/python scripts/evaluate_replay_suite.py \
  backtests/kaggle_feedback/replays \
  --agent main.py \
  --team orf527 \
  --json backtests/kaggle_feedback/replay_suite_main.json
```

Score that replay suite with extra weight on newer submissions:

```bash
.venv/bin/python scripts/weighted_replay_score.py \
  backtests/kaggle_feedback/replay_suite_main.json
```

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
