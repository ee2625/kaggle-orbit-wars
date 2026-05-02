#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import (  # noqa: E402
    BOARD_SIZE,
    Planet,
    Fleet,
    crosses_sun,
    distance_to_segment,
    fleet_speed,
    projected_fleet_target,
)


@dataclass
class LaunchStats:
    count: int = 0
    ships: int = 0
    projected_hits: int = 0
    projected_misses: int = 0
    neutral_targets: int = 0
    enemy_targets: int = 0
    friendly_targets: int = 0
    comet_targets: int = 0
    first_turn: int | None = None
    early_examples: list[str] = field(default_factory=list)


@dataclass
class FleetDeathStats:
    sun: int = 0
    out_of_bounds: int = 0
    planet: int = 0
    unknown: int = 0


@dataclass
class PlayerStats:
    team: str
    reward: float | int | None
    status: str
    final_score: int
    final_planets: int
    final_production: int
    first_capture_turn: int | None = None
    first_enemy_capture_turn: int | None = None
    max_planets: int = 0
    max_production: int = 0
    launch: LaunchStats = field(default_factory=LaunchStats)
    deaths: FleetDeathStats = field(default_factory=FleetDeathStats)
    captures: list[str] = field(default_factory=list)


def load_replay(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def world_observation(replay: dict[str, Any], turn: int) -> dict[str, Any]:
    return replay["steps"][turn][0]["observation"]


def planets_by_id(obs: dict[str, Any]) -> dict[int, Planet]:
    return {int(raw[0]): Planet(*raw) for raw in obs.get("planets", [])}


def fleets_by_id(obs: dict[str, Any]) -> dict[int, Fleet]:
    return {int(raw[0]): Fleet(*raw) for raw in obs.get("fleets", [])}


def scores_from_obs(obs: dict[str, Any], player_count: int) -> list[int]:
    scores = [0 for _ in range(player_count)]
    for planet in obs.get("planets", []) or []:
        owner = int(planet[1])
        if 0 <= owner < player_count:
            scores[owner] += int(planet[5])

    for fleet in obs.get("fleets", []) or []:
        owner = int(fleet[1])
        if 0 <= owner < player_count:
            scores[owner] += int(fleet[6])

    return scores


def owned_production(obs: dict[str, Any], player_count: int, comet_ids: set[int]) -> list[int]:
    production = [0 for _ in range(player_count)]
    for planet in obs.get("planets", []) or []:
        owner = int(planet[1])
        planet_id = int(planet[0])
        if 0 <= owner < player_count and planet_id not in comet_ids:
            production[owner] += int(planet[6])

    return production


def owned_planet_counts(obs: dict[str, Any], player_count: int, comet_ids: set[int]) -> list[int]:
    counts = [0 for _ in range(player_count)]
    for planet in obs.get("planets", []) or []:
        owner = int(planet[1])
        planet_id = int(planet[0])
        if 0 <= owner < player_count and planet_id not in comet_ids:
            counts[owner] += 1

    return counts


def team_names(replay: dict[str, Any]) -> list[str]:
    info = replay.get("info", {}) or {}
    names = info.get("TeamNames") or []
    if names:
        return [str(name) for name in names]

    agents = info.get("Agents") or []
    if agents:
        return [str(agent.get("Name", f"player{index}")) for index, agent in enumerate(agents)]

    return [f"player{index}" for index in range(len(replay.get("steps", [[]])[0]))]


def classify_target_owner(target: Planet, player: int, launch: LaunchStats, comet_ids: set[int]) -> None:
    if target.id in comet_ids:
        launch.comet_targets += 1
    if target.owner == -1:
        launch.neutral_targets += 1
    elif target.owner == player:
        launch.friendly_targets += 1
    else:
        launch.enemy_targets += 1


def update_launch_stats(replay: dict[str, Any], player_count: int, stats: list[PlayerStats], early_turns: int) -> None:
    for turn, states in enumerate(replay["steps"]):
        obs = world_observation(replay, turn)
        planets = list(planets_by_id(obs).values())
        planet_map = {planet.id: planet for planet in planets}
        angular_velocity = float(obs.get("angular_velocity", 0.0) or 0.0)
        comet_ids = {int(value) for value in obs.get("comet_planet_ids", []) or []}

        for player in range(player_count):
            action = states[player].get("action") or []
            for move in action:
                if not isinstance(move, list) or len(move) != 3:
                    continue
                source_id = int(move[0])
                angle = float(move[1])
                ships = int(move[2])
                source = planet_map.get(source_id)
                if source is None:
                    continue

                launch = stats[player].launch
                launch.count += 1
                launch.ships += ships
                if launch.first_turn is None:
                    launch.first_turn = turn

                fake_fleet = Fleet(
                    -1,
                    player,
                    source.x + math.cos(angle) * (source.radius + 0.05),
                    source.y + math.sin(angle) * (source.radius + 0.05),
                    angle,
                    source.id,
                    ships,
                )
                target = projected_fleet_target(fake_fleet, planets, obs, angular_velocity, comet_ids, max_turns=140)
                if target is None:
                    launch.projected_misses += 1
                    if turn <= early_turns and len(launch.early_examples) < 6:
                        launch.early_examples.append(f"t{turn}: {ships} ships from {source_id} projected to no planet")
                    continue

                launch.projected_hits += 1
                classify_target_owner(target, player, launch, comet_ids)
                if turn <= early_turns and len(launch.early_examples) < 6:
                    launch.early_examples.append(
                        f"t{turn}: {ships} ships from {source_id} -> p{target.id} owner={target.owner} prod={target.production}"
                    )


def classify_fleet_deaths(replay: dict[str, Any], player_count: int, stats: list[PlayerStats]) -> None:
    for turn in range(1, len(replay["steps"])):
        prev_obs = world_observation(replay, turn - 1)
        curr_obs = world_observation(replay, turn)
        prev_fleets = fleets_by_id(prev_obs)
        curr_fleets = fleets_by_id(curr_obs)
        curr_planets = list(planets_by_id(curr_obs).values())

        for fleet_id, fleet in prev_fleets.items():
            if fleet_id in curr_fleets or not (0 <= fleet.owner < player_count):
                continue

            speed = fleet_speed(fleet.ships)
            next_x = fleet.x + math.cos(fleet.angle) * speed
            next_y = fleet.y + math.sin(fleet.angle) * speed
            deaths = stats[fleet.owner].deaths

            if crosses_sun(fleet.x, fleet.y, next_x, next_y):
                deaths.sun += 1
            elif next_x < 0.0 or next_x > BOARD_SIZE or next_y < 0.0 or next_y > BOARD_SIZE:
                deaths.out_of_bounds += 1
            elif fleet_hits_planet(fleet, next_x, next_y, curr_planets):
                deaths.planet += 1
            else:
                deaths.unknown += 1


def fleet_hits_planet(fleet: Fleet, next_x: float, next_y: float, planets: Iterable[Planet]) -> bool:
    for planet in planets:
        if planet.id == fleet.from_planet_id:
            continue
        if distance_to_segment(planet.x, planet.y, fleet.x, fleet.y, next_x, next_y) <= planet.radius + 0.35:
            return True
    return False


def update_capture_and_curve_stats(replay: dict[str, Any], player_count: int, stats: list[PlayerStats]) -> None:
    previous = planets_by_id(world_observation(replay, 0))

    for turn, _states in enumerate(replay["steps"]):
        obs = world_observation(replay, turn)
        comet_ids = {int(value) for value in obs.get("comet_planet_ids", []) or []}
        counts = owned_planet_counts(obs, player_count, comet_ids)
        production = owned_production(obs, player_count, comet_ids)

        for player in range(player_count):
            stats[player].max_planets = max(stats[player].max_planets, counts[player])
            stats[player].max_production = max(stats[player].max_production, production[player])

        if turn == 0:
            continue

        current = planets_by_id(obs)
        for planet_id, planet in current.items():
            old = previous.get(planet_id)
            if old is None or old.owner == planet.owner:
                continue

            if 0 <= planet.owner < player_count:
                player = planet.owner
                if old.owner != player:
                    if stats[player].first_capture_turn is None:
                        stats[player].first_capture_turn = turn
                    stats[player].captures.append(
                        f"t{turn}: p{planet.id} prod={planet.production} ships={planet.ships} from owner {old.owner}"
                    )

            if 0 <= old.owner < player_count and planet.owner != old.owner:
                lost_player = old.owner
                if stats[lost_player].first_enemy_capture_turn is None and planet.owner != -1:
                    stats[lost_player].first_enemy_capture_turn = turn

        previous = current


def build_player_stats(replay: dict[str, Any], early_turns: int) -> list[PlayerStats]:
    names = team_names(replay)
    player_count = len(names)
    final_obs = world_observation(replay, len(replay["steps"]) - 1)
    final_scores = scores_from_obs(final_obs, player_count)
    final_counts = owned_planet_counts(final_obs, player_count, set())
    final_production = owned_production(final_obs, player_count, set())
    rewards = replay.get("rewards") or [None for _ in range(player_count)]
    statuses = replay.get("statuses") or ["UNKNOWN" for _ in range(player_count)]

    stats = [
        PlayerStats(
            team=names[player],
            reward=rewards[player] if player < len(rewards) else None,
            status=statuses[player] if player < len(statuses) else "UNKNOWN",
            final_score=final_scores[player],
            final_planets=final_counts[player],
            final_production=final_production[player],
        )
        for player in range(player_count)
    ]

    update_capture_and_curve_stats(replay, player_count, stats)
    update_launch_stats(replay, player_count, stats, early_turns)
    classify_fleet_deaths(replay, player_count, stats)
    return stats


def score_timeline(replay: dict[str, Any], interval: int) -> list[tuple[int, list[int], list[int], list[int]]]:
    player_count = len(team_names(replay))
    rows: list[tuple[int, list[int], list[int], list[int]]] = []
    for turn in range(0, len(replay["steps"]), interval):
        obs = world_observation(replay, turn)
        comet_ids = {int(value) for value in obs.get("comet_planet_ids", []) or []}
        rows.append(
            (
                turn,
                scores_from_obs(obs, player_count),
                owned_planet_counts(obs, player_count, comet_ids),
                owned_production(obs, player_count, comet_ids),
            )
        )

    final_turn = len(replay["steps"]) - 1
    if not rows or rows[-1][0] != final_turn:
        obs = world_observation(replay, final_turn)
        comet_ids = {int(value) for value in obs.get("comet_planet_ids", []) or []}
        rows.append(
            (
                final_turn,
                scores_from_obs(obs, player_count),
                owned_planet_counts(obs, player_count, comet_ids),
                owned_production(obs, player_count, comet_ids),
            )
        )

    return rows


def print_replay_report(path: Path, replay: dict[str, Any], early_turns: int, timeline_interval: int) -> None:
    info = replay.get("info", {}) or {}
    episode_id = info.get("EpisodeId", path.stem)
    names = team_names(replay)
    stats = build_player_stats(replay, early_turns)

    print(f"\n{path}")
    print(f"episode={episode_id} players={len(names)} steps={len(replay['steps'])} seed={info.get('seed')}")
    print("teams:", " | ".join(f"{index}:{name}" for index, name in enumerate(names)))

    print("\nTimeline")
    for turn, scores, counts, production in score_timeline(replay, timeline_interval):
        print(f"  t{turn:>3}: scores={scores} planets={counts} production={production}")

    print("\nPlayer Diagnostics")
    for player, row in enumerate(stats):
        launch = row.launch
        deaths = row.deaths
        miss_rate = launch.projected_misses / launch.count if launch.count else 0.0
        hit_rate = launch.projected_hits / launch.count if launch.count else 0.0
        print(
            f"  p{player} {row.team}: reward={row.reward} status={row.status} "
            f"final_score={row.final_score} max_planets={row.max_planets} max_prod={row.max_production}"
        )
        print(
            f"    first_capture={row.first_capture_turn} first_planet_lost_to_enemy={row.first_enemy_capture_turn} "
            f"launches={launch.count} ships_launched={launch.ships} first_launch={launch.first_turn}"
        )
        print(
            f"    projected_hit_rate={hit_rate:.1%} miss_rate={miss_rate:.1%} "
            f"targets neutral/enemy/friendly/comet="
            f"{launch.neutral_targets}/{launch.enemy_targets}/{launch.friendly_targets}/{launch.comet_targets}"
        )
        print(
            f"    fleet_deaths sun/oob/planet/unknown="
            f"{deaths.sun}/{deaths.out_of_bounds}/{deaths.planet}/{deaths.unknown}"
        )
        if row.captures:
            print("    captures:", "; ".join(row.captures[:6]))
        if launch.early_examples:
            print("    early launches:", "; ".join(launch.early_examples))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Orbit Wars replay failure modes.")
    parser.add_argument("replays", nargs="+", type=Path, help="Replay JSON files.")
    parser.add_argument("--early-turns", type=int, default=80, help="Store sample launch diagnostics through this turn.")
    parser.add_argument("--timeline-interval", type=int, default=50, help="Turns between timeline snapshots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for replay_path in args.replays:
        print_replay_report(replay_path, load_replay(replay_path), args.early_turns, args.timeline_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
