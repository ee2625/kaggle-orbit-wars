#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest import final_scores, load_make, quiet_output  # noqa: E402


BUCKETS = (20, 40, 60, 80, 120, 180, 250, 500)
MILESTONES = (20, 35, 50, 65)


@dataclass
class SeatEconomy:
    label: str
    seed: int
    seat: int
    players: int
    score: int
    reward: float | int | None
    status: str
    rank: int
    max_prod: int = 0
    max_planets: int = 0
    final_prod: int = 0
    final_planets: int = 0
    prod_by_bucket: dict[str, int] = field(default_factory=dict)
    planets_by_bucket: dict[str, int] = field(default_factory=dict)
    milestone_turns: dict[str, int | None] = field(default_factory=dict)
    first_extra_planet_turn: int | None = None
    launches: int = 0
    ships_launched: int = 0
    first_launch_turn: int | None = None


def label_for(spec: str) -> str:
    path = Path(spec)
    if path.suffix:
        return path.stem
    return spec


def rank_scores(scores: list[int]) -> list[int]:
    sorted_scores = sorted(set(scores), reverse=True)
    return [sorted_scores.index(score) + 1 for score in scores]


def owner_count_and_prod(obs: Any, players: int) -> tuple[list[int], list[int]]:
    if isinstance(obs, dict):
        planets = obs.get("planets", []) or []
        comet_ids = set(obs.get("comet_planet_ids", []) or [])
    else:
        planets = getattr(obs, "planets", []) or []
        comet_ids = set(getattr(obs, "comet_planet_ids", []) or [])

    counts = [0 for _ in range(players)]
    production = [0 for _ in range(players)]
    for row in planets:
        owner = int(row[1])
        planet_id = int(row[0])
        if 0 <= owner < players and planet_id not in comet_ids:
            counts[owner] += 1
            production[owner] += int(row[6])
    return counts, production


def actions_for_state(state: Any) -> list:
    if isinstance(state, dict):
        return state.get("action") or []
    return getattr(state, "action", None) or []


def run_episode(make: Any, specs: list[str], labels: list[str], seed: int, verbose_env: bool) -> list[SeatEconomy]:
    started = time.perf_counter()
    with quiet_output(not verbose_env):
        env = make("orbit_wars", configuration={"seed": seed}, debug=True)
        env.run(specs)
    _duration = time.perf_counter() - started

    players = len(specs)
    final = env.steps[-1]
    rewards = [state.reward for state in final]
    statuses = [state.status for state in final]
    scores = final_scores(final[0].observation, players)
    ranks = rank_scores(scores)

    rows = [
        SeatEconomy(
            label=labels[seat],
            seed=seed,
            seat=seat,
            players=players,
            score=scores[seat],
            reward=rewards[seat],
            status=statuses[seat],
            rank=ranks[seat],
            milestone_turns={str(value): None for value in MILESTONES},
        )
        for seat in range(players)
    ]

    for turn, states in enumerate(env.steps):
        obs = states[0].observation
        counts, production = owner_count_and_prod(obs, players)
        bucket = str(next((value for value in BUCKETS if turn <= value), BUCKETS[-1]))

        for seat, row in enumerate(rows):
            row.max_prod = max(row.max_prod, production[seat])
            row.max_planets = max(row.max_planets, counts[seat])
            row.final_prod = production[seat]
            row.final_planets = counts[seat]
            row.prod_by_bucket[bucket] = production[seat]
            row.planets_by_bucket[bucket] = counts[seat]
            if counts[seat] > 1 and row.first_extra_planet_turn is None:
                row.first_extra_planet_turn = turn
            for milestone in MILESTONES:
                key = str(milestone)
                if row.milestone_turns[key] is None and production[seat] >= milestone:
                    row.milestone_turns[key] = turn

        for seat, state in enumerate(states):
            action = actions_for_state(state)
            for move in action:
                if not isinstance(move, list) or len(move) != 3:
                    continue
                try:
                    ships = int(move[2])
                except (TypeError, ValueError):
                    continue
                rows[seat].launches += 1
                rows[seat].ships_launched += max(0, ships)
                if rows[seat].first_launch_turn is None:
                    rows[seat].first_launch_turn = turn

    return rows


def summarize(rows: list[SeatEconomy]) -> list[dict[str, Any]]:
    grouped: dict[str, list[SeatEconomy]] = {}
    for row in rows:
        grouped.setdefault(row.label, []).append(row)

    summary = []
    for label, items in sorted(grouped.items()):
        wins = sum(1 for row in items if row.rank == 1 and row.status == "DONE")
        losses = sum(1 for row in items if row.rank != 1 and row.status == "DONE")
        errors = sum(1 for row in items if row.status != "DONE")

        def mean(values: list[float]) -> float | None:
            return round(statistics.fmean(values), 3) if values else None

        summary.append(
            {
                "agent": label,
                "games": len(items),
                "wins": wins,
                "losses": losses,
                "errors": errors,
                "avg_score": mean([row.score for row in items]),
                "avg_max_prod": mean([row.max_prod for row in items]),
                "avg_final_prod": mean([row.final_prod for row in items]),
                "avg_max_planets": mean([row.max_planets for row in items]),
                "avg_launches": mean([row.launches for row in items]),
                "avg_ships_launched": mean([row.ships_launched for row in items]),
                "low_prod_games": sum(1 for row in items if row.max_prod < 35),
                "avg_first_extra_planet": mean(
                    [row.first_extra_planet_turn for row in items if row.first_extra_planet_turn is not None]
                ),
                "avg_turn_prod35": mean(
                    [row.milestone_turns["35"] for row in items if row.milestone_turns["35"] is not None]
                ),
                "avg_turn_prod50": mean(
                    [row.milestone_turns["50"] for row in items if row.milestone_turns["50"] is not None]
                ),
            }
        )
    return sorted(summary, key=lambda row: (-row["wins"], -(row["avg_max_prod"] or 0), row["agent"]))


def build_lineups(agent_specs: list[str], players: int, games: int) -> list[list[str]]:
    combos = list(itertools.combinations(agent_specs, players))
    if not combos:
        raise ValueError(f"Need at least {players} agents.")

    lineups = []
    for index in range(games):
        combo = list(combos[index % len(combos)])
        offset = index % players
        lineups.append(combo[offset:] + combo[:offset])
    return lineups


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Orbit Wars production curves across agents.")
    parser.add_argument("--agent", action="append", required=True)
    parser.add_argument("--players", type=int, choices=(2, 4), required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=170000)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--verbose-env", action="store_true")
    args = parser.parse_args()

    make = load_make(args.verbose_env)
    all_rows: list[SeatEconomy] = []
    lineups = build_lineups(args.agent, args.players, args.games)

    for episode, specs in enumerate(lineups):
        labels = [label_for(spec) for spec in specs]
        all_rows.extend(run_episode(make, specs, labels, args.seed + episode, args.verbose_env))

    payload = {
        "players": args.players,
        "games": args.games,
        "seed": args.seed,
        "summary": summarize(all_rows),
        "rows": [asdict(row) for row in all_rows],
    }

    for row in payload["summary"]:
        print(
            f"{row['agent']}: {row['wins']}-{row['losses']} "
            f"avg_max_prod={row['avg_max_prod']} low_prod={row['low_prod_games']} "
            f"avg_launches={row['avg_launches']}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
