# Orbit Wars Kaggle Bot

A rule-based agent for the [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)
simulation competition. Iterates on a publicly-shared community baseline with
substantial reworking of constants, decision modes, routing, and 4-player handling.

## License & Lineage

This project is released under the [MIT License](LICENSE).

From **v5.0 onward**, the agent derives from the publicly-shared Kaggle notebook
[Orbit (Star) Wars | LB: MAX 1224](https://www.kaggle.com/code/romantamrazov/orbit-star-wars-lb-max-1224)
by **Roman Tamrazov** (kernel version 6). The original Kaggle notebook does not
set an explicit license; the third-party distribution at
[automatylicza/orbit-wars-lab](https://github.com/automatylicza/orbit-wars-lab)
characterizes it as Apache 2.0 / MIT (both OSI-approved). We rely on that
distribution's license in good faith.

Subsequent versions (v5.x through v6.71) extensively modify the upstream work:
constants are retuned for our opponent pool, the 4-player decision path is
substantially rewritten, routing/aim caching has been overhauled, and a mode
router was added on top. The forward-simulation core and many strategic
primitives originate from Tamrazov's work — those concepts retain their
original credit.

Full attribution detail and third-party references live in [NOTICE](NOTICE).
In-code attribution headers are preserved in `main.py` and the
`submissions/v5_*` family.

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
python scripts/league_backtest.py --agent main.py --agent random --games 20 --jobs 4 --out-dir backtests
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

`--jobs` parallelizes `round-robin` and `random` schedules. It is ignored for
`ladder`, because ladder matchmaking depends on live rating updates.

For 4-player local games, provide at least four agents:

```bash
python scripts/league_backtest.py \
  --players 4 \
  --agent main.py \
  --agent backtests/agents/submitted_133182e.py \
  --agent random \
  --agent random \
  --games 20 \
  --jobs 4 \
  --out-dir backtests/league_4p
```

Run a small Optuna sweep over the main strategy constants:

```bash
python scripts/tune_constants.py \
  --trials 30 \
  --games-2p 8 \
  --games-4p 8 \
  --jobs 4 \
  --out-dir backtests/optuna_constants
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
