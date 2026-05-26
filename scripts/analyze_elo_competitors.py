#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import Fleet, projected_fleet_target  # noqa: E402
from scripts.analyze_replay import (  # noqa: E402
    load_replay,
    owned_production,
    planets_by_id,
    scores_from_obs,
    team_names,
    world_observation,
)


@dataclass
class SeatFeatures:
    team: str
    episode: str
    player: int
    player_count: int
    result: str
    first_capture: int | None
    max_prod: int
    final_score: int
    launches: int = 0
    ships: int = 0
    sizes: list[int] = field(default_factory=list)
    target_counts: Counter[str] = field(default_factory=Counter)
    target_ships: Counter[str] = field(default_factory=Counter)
    early_counts: Counter[str] = field(default_factory=Counter)
    early_ships: Counter[str] = field(default_factory=Counter)


def expand_replays(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*replay.json")))
        else:
            expanded.append(path)
    return expanded


def pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return int(ordered[index])


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def episode_id(replay: dict[str, Any], path: Path) -> str:
    info = replay.get("info", {}) or {}
    return str(info.get("EpisodeId") or replay.get("id") or path.stem)


def classify_rewards(replay: dict[str, Any], player: int) -> str:
    rewards = replay.get("rewards") or []
    if not rewards or player >= len(rewards) or rewards[player] is None:
        return "unknown"
    best = max(reward for reward in rewards if reward is not None)
    if rewards[player] != best:
        return "loss"
    if sum(1 for reward in rewards if reward == best) > 1:
        return "tie"
    return "win"


def first_capture_turn(replay: dict[str, Any], player: int) -> int | None:
    previous = planets_by_id(world_observation(replay, 0))
    for turn in range(1, len(replay["steps"])):
        current = planets_by_id(world_observation(replay, turn))
        for planet_id, planet in current.items():
            old = previous.get(planet_id)
            if old is not None and old.owner != player and planet.owner == player:
                return turn
        previous = current
    return None


def max_production(replay: dict[str, Any], player: int, player_count: int) -> int:
    best = 0
    for turn in range(len(replay["steps"])):
        obs = world_observation(replay, turn)
        comet_ids = {int(value) for value in obs.get("comet_planet_ids", []) or []}
        best = max(best, owned_production(obs, player_count, comet_ids)[player])
    return best


def target_kind(target: Any | None, player: int, comet_ids: set[int]) -> str:
    if target is None:
        return "miss"
    if target.id in comet_ids:
        return "comet"
    if target.owner == -1:
        return "neutral"
    if target.owner == player:
        return "friendly"
    return "enemy"


def collect_seat_features(path: Path, replay: dict[str, Any], early_turns: int) -> list[SeatFeatures]:
    names = team_names(replay)
    player_count = len(names)
    final_scores = scores_from_obs(world_observation(replay, len(replay["steps"]) - 1), player_count)
    rows = [
        SeatFeatures(
            team=names[player],
            episode=episode_id(replay, path),
            player=player,
            player_count=player_count,
            result=classify_rewards(replay, player),
            first_capture=first_capture_turn(replay, player),
            max_prod=max_production(replay, player, player_count),
            final_score=final_scores[player],
        )
        for player in range(player_count)
    ]

    for turn, states in enumerate(replay["steps"]):
        obs = world_observation(replay, turn)
        planets = list(planets_by_id(obs).values())
        planet_map = {planet.id: planet for planet in planets}
        angular_velocity = float(obs.get("angular_velocity", 0.0) or 0.0)
        comet_ids = {int(value) for value in obs.get("comet_planet_ids", []) or []}

        for player, row in enumerate(rows):
            for move in states[player].get("action") or []:
                if not isinstance(move, list) or len(move) != 3:
                    continue
                source_id = int(move[0])
                angle = float(move[1])
                ships = int(move[2])
                source = planet_map.get(source_id)
                if source is None or ships <= 0:
                    continue

                fake = Fleet(
                    -1,
                    player,
                    source.x + math.cos(angle) * (source.radius + 0.05),
                    source.y + math.sin(angle) * (source.radius + 0.05),
                    angle,
                    source.id,
                    ships,
                )
                target = projected_fleet_target(
                    fake,
                    planets,
                    obs,
                    angular_velocity,
                    comet_ids,
                    max_turns=160,
                )
                kind = target_kind(target, player, comet_ids)
                row.launches += 1
                row.ships += ships
                row.sizes.append(ships)
                row.target_counts[kind] += 1
                row.target_ships[kind] += ships
                if turn <= early_turns:
                    row.early_counts[kind] += 1
                    row.early_ships[kind] += ships

    return rows


def leaderboard_lookup(path: Path) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["TeamName"]] = row
    return rows


def style_label(rows: list[SeatFeatures]) -> str:
    ships = sum(row.ships for row in rows)
    enemy_share = sum(row.target_ships["enemy"] for row in rows) / ships if ships else 0.0
    friendly_share = sum(row.target_ships["friendly"] for row in rows) / ships if ships else 0.0
    miss_share = sum(row.target_ships["miss"] for row in rows) / ships if ships else 0.0
    avg_p90 = mean([pct(row.sizes, 0.90) for row in rows])
    avg_prod = mean([row.max_prod for row in rows])
    avg_first = mean([row.first_capture for row in rows if row.first_capture is not None])

    if avg_prod >= 55 and friendly_share >= 0.25 and avg_p90 >= 70:
        return "controlled funnel"
    if enemy_share >= 0.45 and avg_p90 >= 60:
        return "heavy pressure"
    if enemy_share >= 0.40 and avg_prod < 35:
        return "reckless pressure"
    if avg_first and avg_first > 24:
        return "slow opener"
    if friendly_share >= 0.40 and enemy_share < 0.30:
        return "passive funnel"
    if miss_share >= 0.32:
        return "noisy routes"
    return "balanced"


def build_report(
    all_rows: list[SeatFeatures],
    our_team: str,
    leaderboard: dict[str, dict[str, str]],
    score_low: float,
    score_high: float,
) -> str:
    opponents = [row for row in all_rows if row.team != our_team]
    grouped: dict[str, list[SeatFeatures]] = defaultdict(list)
    for row in opponents:
        grouped[row.team].append(row)

    lines = [
        "# Orbit Wars 800-Elo Opponent Report",
        "",
        f"Analyzed {len({row.episode for row in all_rows})} matched episodes and {len(grouped)} opponent teams.",
        f"Leaderboard comparison band: {score_low:.0f}-{score_high:.0f}.",
        "",
        "## Matched Opponents",
        "",
        "| team | lb rank | lb score | episodes | W-L-T | avg max prod | avg first cap | avg launches | avg ships | p90 send | enemy ship share | friendly ship share | miss ship share | style |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    summaries = []
    for team, rows in grouped.items():
        lb = leaderboard.get(team, {})
        rank = lb.get("Rank", "-")
        score = lb.get("Score", "-")
        wins = sum(1 for row in rows if row.result == "win")
        losses = sum(1 for row in rows if row.result == "loss")
        ties = sum(1 for row in rows if row.result == "tie")
        ships = sum(row.ships for row in rows)
        enemy_share = sum(row.target_ships["enemy"] for row in rows) / ships if ships else 0.0
        friendly_share = sum(row.target_ships["friendly"] for row in rows) / ships if ships else 0.0
        miss_share = sum(row.target_ships["miss"] for row in rows) / ships if ships else 0.0
        avg_prod = mean([row.max_prod for row in rows])
        first_values = [row.first_capture for row in rows if row.first_capture is not None]
        avg_first = mean(first_values)
        avg_launches = mean([row.launches for row in rows])
        avg_ships = mean([row.ships for row in rows])
        avg_p90 = mean([pct(row.sizes, 0.90) for row in rows])
        style = style_label(rows)
        summaries.append((float(score) if score != "-" else -1.0, team, len(rows), lines[-1]))
        lines.append(
            f"| {team} | {rank} | {score} | {len(rows)} | {wins}-{losses}-{ties} | "
            f"{avg_prod:.1f} | {avg_first:.1f} | {avg_launches:.1f} | {avg_ships:.0f} | "
            f"{avg_p90:.0f} | {enemy_share:.1%} | {friendly_share:.1%} | {miss_share:.1%} | {style} |"
        )

    band_rows = [
        row
        for row in leaderboard.values()
        if score_low <= float(row["Score"]) <= score_high
    ]
    band_recent = sorted(
        band_rows,
        key=lambda row: abs(float(row["Score"]) - float(leaderboard.get(our_team, {"Score": "855.7"})["Score"])),
    )[:30]
    lines.extend([
        "",
        "## Closest Leaderboard Neighbors",
        "",
        "| rank | score | team | last submission | submissions |",
        "|---:|---:|---|---|---:|",
    ])
    for row in band_recent:
        lines.append(
            f"| {row['Rank']} | {row['Score']} | {row['TeamName']} | {row['LastSubmissionDate']} | {row['SubmissionCount']} |"
        )

    opponent_rows = opponents
    our_rows = [row for row in all_rows if row.team == our_team]
    lines.extend([
        "",
        "## Takeaways",
        "",
    ])
    if our_rows:
        our_wins = sum(1 for row in our_rows if row.result == "win")
        our_losses = sum(1 for row in our_rows if row.result == "loss")
        lines.append(f"- Our collected matched-episode record in this pull is {our_wins}W-{our_losses}L across {len(our_rows)} seats.")
    lines.append("- Same-band opponents are mixed: many are not top-style controlled-funnel bots yet. The exploitable low/mid-band pattern is slow opener or noisy pressure, not perfect high-Elo macro.")
    lines.append("- When we lose in the 800 band, it is often not because the opponent has a huge late-game packet plan; it is because we fail the first production race and never reach the economy needed for our own packet phase.")
    lines.append("- Adaptive mode should be behavior-driven: punish reckless early enemy pressure with counter-captures, greed against slow openers, and unlock pressure packets only after we have enough production.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze same-Elo Orbit Wars opponents from matched replays.")
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--our-team", default="orf527")
    parser.add_argument("--score-low", type=float, default=800.0)
    parser.add_argument("--score-high", type=float, default=900.0)
    parser.add_argument("--early-turns", type=int, default=80)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[SeatFeatures] = []
    for replay_path in expand_replays(args.replays):
        try:
            replay = load_replay(replay_path)
        except (OSError, json.JSONDecodeError):
            continue
        rows.extend(collect_seat_features(replay_path, replay, args.early_turns))

    report = build_report(
        rows,
        our_team=args.our_team,
        leaderboard=leaderboard_lookup(args.leaderboard),
        score_low=args.score_low,
        score_high=args.score_high,
    )
    print(report)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report, encoding="utf-8")
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
