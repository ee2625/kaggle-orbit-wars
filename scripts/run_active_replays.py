#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest import final_scores, load_make, quiet_output  # noqa: E402
from scripts.league_backtest import (  # noqa: E402
    Rating,
    adjusted_scores,
    build_competitors,
    build_matchups,
    choose_matchup,
    rank_scores,
    rotate_lineup,
    update_pairwise_ratings,
)


@dataclass(frozen=True)
class SavedEpisode:
    episode: int
    seed: int
    replay: str
    lineup: list[str]
    specs: list[str]
    scores: list[int]
    rewards: list[float | int | None]
    statuses: list[str]
    ranks: list[int]
    duration_s: float
    steps: int


def run_episode(make: Any, lineup: list[Any], episode: int, seed: int, out_dir: Path, debug: bool, verbose_env: bool) -> SavedEpisode:
    started = time.perf_counter()
    with quiet_output(not verbose_env):
        env = make("orbit_wars", configuration={"seed": seed}, debug=debug)
        env.run([competitor.spec for competitor in lineup])
    duration_s = time.perf_counter() - started

    final = env.steps[-1] if env.steps else []
    rewards = [state.reward for state in final]
    statuses = [state.status for state in final]
    scores = adjusted_scores(final_scores(final[0].observation, len(lineup)) if final else [0 for _ in lineup], statuses)

    replay = env.toJSON()
    replay.setdefault("info", {})
    replay["info"]["CodexLineupLabels"] = [competitor.label for competitor in lineup]
    replay["info"]["CodexLineupSpecs"] = [competitor.spec for competitor in lineup]
    replay["info"]["CodexSeed"] = seed
    replay["info"]["CodexEpisode"] = episode

    out_dir.mkdir(parents=True, exist_ok=True)
    replay_path = out_dir / f"episode-{episode:04d}-seed-{seed}.json"
    replay_path.write_text(json.dumps(replay, indent=2), encoding="utf-8")

    return SavedEpisode(
        episode=episode,
        seed=seed,
        replay=str(replay_path),
        lineup=[competitor.label for competitor in lineup],
        specs=[competitor.spec for competitor in lineup],
        scores=scores,
        rewards=rewards,
        statuses=statuses,
        ranks=rank_scores(scores),
        duration_s=duration_s,
        steps=len(env.steps),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run active Orbit Wars matches and save replay JSONs for behavior analysis.")
    parser.add_argument("--players", type=int, choices=(2, 4), required=True, help="Number of players per episode.")
    parser.add_argument("--games", type=int, default=8, help="Number of active games to run.")
    parser.add_argument("--seed", type=int, default=900000, help="First environment seed.")
    parser.add_argument("--schedule", choices=("round-robin", "random"), default="round-robin", help="Lineup schedule.")
    parser.add_argument("--agent", action="append", required=True, help="Agent file/spec. Repeat at least --players times.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for replay JSONs and summary.json.")
    parser.add_argument("--debug", action="store_true", help="Run Kaggle environment in debug mode.")
    parser.add_argument("--verbose-env", action="store_true", help="Do not suppress environment output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    competitors = build_competitors(args.agent)
    if len(competitors) < args.players:
        print(f"need at least {args.players} agents", file=sys.stderr)
        return 2

    try:
        make = load_make(args.verbose_env)
    except ImportError:
        print("Missing dependency. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    matchups = build_matchups(competitors, args.players)
    ratings = {competitor.label: Rating() for competitor in competitors}
    episodes: list[SavedEpisode] = []

    for episode_index in range(args.games):
        matchup = choose_matchup(
            competitors=competitors,
            matchups=matchups,
            ratings=ratings,
            episode=episode_index,
            players=args.players,
            schedule=args.schedule,
            rng=rng,
        )
        lineup = rotate_lineup(matchup, episode_index)
        seed = args.seed + episode_index
        row = run_episode(
            make=make,
            lineup=lineup,
            episode=episode_index + 1,
            seed=seed,
            out_dir=args.out_dir,
            debug=args.debug,
            verbose_env=args.verbose_env,
        )
        update_pairwise_ratings(row.lineup, row.scores, ratings, beta=200.0, tau=0.0)
        episodes.append(row)
        print(
            f"episode={row.episode} seed={row.seed} lineup={','.join(row.lineup)} "
            f"scores={row.scores} ranks={row.ranks} statuses={row.statuses} replay={Path(row.replay).name}"
        )

    summary = {
        "players": args.players,
        "games": args.games,
        "seed": args.seed,
        "schedule": args.schedule,
        "agents": [asdict(competitor) for competitor in competitors],
        "episodes": [asdict(row) for row in episodes],
        "ratings": {label: asdict(rating) for label, rating in ratings.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
