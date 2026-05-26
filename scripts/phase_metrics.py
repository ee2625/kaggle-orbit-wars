#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict, namedtuple
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


BOARD_SIZE = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
ROTATION_LIMIT = 50.0
LAUNCH_CLEARANCE = 0.1
TARGET_HORIZON = 140
BUCKETS = ((0, 49), (50, 99), (100, 199), (200, 349), (350, 499))
CHECKPOINTS = (25, 50, 75, 100, 150, 200, 300)

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])


@dataclass
class BucketStats:
    launches: int = 0
    ships: int = 0
    sends: list[int] = field(default_factory=list)


@dataclass
class PlayerMetrics:
    replay: str
    episode_id: str
    player_count: int
    player: int
    team: str
    result: str
    reward: float | int | None
    status: str
    final_score: int
    final_planets: int
    final_prod: int
    max_score: int = 0
    max_planets: int = 0
    max_prod: int = 0
    first_capture: int | None = None
    first_high_prod_neutral: int | None = None
    first_enemy_capture: int | None = None
    first_planet_lost: int | None = None
    launches: int = 0
    ships_launched: int = 0
    avg_send: float = 0.0
    p50_send: float = 0.0
    p90_send: float = 0.0
    p99_send: float = 0.0
    projected_hits: int = 0
    projected_misses: int = 0
    neutral_targets: int = 0
    enemy_targets: int = 0
    friendly_targets: int = 0
    comet_targets: int = 0
    sun_or_oob_routes: int = 0
    first_launch: int | None = None
    score_curve: dict[str, int] = field(default_factory=dict)
    planet_curve: dict[str, int] = field(default_factory=dict)
    prod_curve: dict[str, int] = field(default_factory=dict)
    bucket_launches: dict[str, int] = field(default_factory=dict)
    bucket_ships: dict[str, int] = field(default_factory=dict)
    bucket_p90: dict[str, float] = field(default_factory=dict)


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return float(ordered[low])
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "steps" not in payload:
        return None
    return payload


def expand_inputs(paths: Iterable[Path], limit: int | None) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.json")))
        else:
            expanded.append(path)
    filtered = [path for path in expanded if path.name != "dataset-metadata.json"]
    return filtered[:limit] if limit is not None else filtered


def observation(replay: dict[str, Any], turn: int) -> dict[str, Any]:
    return replay["steps"][turn][0]["observation"]


def episode_id(replay: dict[str, Any], path: Path) -> str:
    info = replay.get("info", {}) or {}
    return str(info.get("EpisodeId") or info.get("CodexEpisode") or replay.get("id") or path.stem)


def team_names(replay: dict[str, Any]) -> list[str]:
    info = replay.get("info", {}) or {}
    codex_labels = info.get("CodexLineupLabels") or []
    if codex_labels:
        return [str(label) for label in codex_labels]
    names = info.get("TeamNames") or []
    if names:
        return [str(name) for name in names]
    agents = info.get("Agents") or []
    if agents:
        return [str(agent.get("Name", f"player{index}")) for index, agent in enumerate(agents)]
    return [f"player{index}" for index in range(len(replay.get("steps", [[]])[0]))]


def planets_by_id(obs: dict[str, Any]) -> dict[int, Planet]:
    planets: dict[int, Planet] = {}
    for row in obs.get("planets", []) or []:
        try:
            planet = Planet(*row)
        except TypeError:
            continue
        planets[int(planet.id)] = planet
    return planets


def comet_ids(obs: dict[str, Any]) -> set[int]:
    return {int(value) for value in obs.get("comet_planet_ids", []) or []}


def scores_counts_prod(obs: dict[str, Any], player_count: int) -> tuple[list[int], list[int], list[int]]:
    scores = [0 for _ in range(player_count)]
    counts = [0 for _ in range(player_count)]
    prod = [0 for _ in range(player_count)]
    comets = comet_ids(obs)

    for row in obs.get("planets", []) or []:
        try:
            owner = int(row[1])
            planet_id = int(row[0])
            ships = int(row[5])
            production = int(row[6])
        except (IndexError, TypeError, ValueError):
            continue
        if 0 <= owner < player_count:
            scores[owner] += ships
            if planet_id not in comets:
                counts[owner] += 1
                prod[owner] += production

    for row in obs.get("fleets", []) or []:
        try:
            owner = int(row[1])
            ships = int(row[6])
        except (IndexError, TypeError, ValueError):
            continue
        if 0 <= owner < player_count:
            scores[owner] += ships

    return scores, counts, prod


def fleet_speed(ships: int) -> float:
    ships = max(1, int(ships))
    if ships <= 1:
        return 1.0
    ratio = max(0.0, min(1.0, math.log(ships) / math.log(1000.0)))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio**1.5)


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def distance_to_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return distance(px, py, x1, y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
    return distance(px, py, x1 + dx * t, y1 + dy * t)


def crosses_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    return distance_to_segment(CENTER_X, CENTER_Y, x1, y1, x2, y2) <= SUN_R


def initial_planets_by_id(obs: dict[str, Any]) -> dict[int, Planet]:
    return {int(row[0]): Planet(*row) for row in obs.get("initial_planets", []) or []}


def orbital_radius(planet: Planet) -> float:
    return distance(float(planet.x), float(planet.y), CENTER_X, CENTER_Y)


def predict_planet_position(
    planet: Planet,
    turns: int,
    initial_by_id: dict[int, Planet],
    angular_velocity: float,
    comets: list[dict[str, Any]],
    comets_set: set[int],
) -> tuple[float, float] | None:
    if int(planet.id) in comets_set:
        for group in comets:
            pids = [int(value) for value in group.get("planet_ids", []) or []]
            if int(planet.id) not in pids:
                continue
            idx = pids.index(int(planet.id))
            paths = group.get("paths", []) or []
            path_index = int(group.get("path_index", 0) or 0)
            if idx >= len(paths):
                return None
            future_index = path_index + turns
            if 0 <= future_index < len(paths[idx]):
                point = paths[idx][future_index]
                return float(point[0]), float(point[1])
            return None

    initial = initial_by_id.get(int(planet.id))
    if initial is None or orbital_radius(initial) + float(initial.radius) >= ROTATION_LIMIT:
        return float(planet.x), float(planet.y)

    radius = orbital_radius(initial)
    current_angle = math.atan2(float(planet.y) - CENTER_Y, float(planet.x) - CENTER_X)
    next_angle = current_angle + angular_velocity * turns
    return CENTER_X + radius * math.cos(next_angle), CENTER_Y + radius * math.sin(next_angle)


def projected_target(
    source: Planet,
    angle: float,
    ships: int,
    planets: list[Planet],
    obs: dict[str, Any],
) -> tuple[str, Planet | None]:
    speed = fleet_speed(ships)
    x = float(source.x) + math.cos(angle) * (float(source.radius) + LAUNCH_CLEARANCE)
    y = float(source.y) + math.sin(angle) * (float(source.radius) + LAUNCH_CLEARANCE)
    initial_by_id = initial_planets_by_id(obs)
    angular_velocity = float(obs.get("angular_velocity", 0.0) or 0.0)
    comets = obs.get("comets", []) or []
    comets_set = comet_ids(obs)

    for turn in range(1, TARGET_HORIZON + 1):
        nx = x + math.cos(angle) * speed
        ny = y + math.sin(angle) * speed
        if nx < 0.0 or nx > BOARD_SIZE or ny < 0.0 or ny > BOARD_SIZE:
            return "sun_oob", None
        if crosses_sun(x, y, nx, ny):
            return "sun_oob", None

        best: tuple[float, Planet] | None = None
        for planet in planets:
            if int(planet.id) == int(source.id):
                continue
            pos = predict_planet_position(planet, turn, initial_by_id, angular_velocity, comets, comets_set)
            if pos is None:
                continue
            threshold = float(planet.radius) + 0.35
            hit_distance = distance_to_segment(pos[0], pos[1], x, y, nx, ny)
            if hit_distance <= threshold:
                entry_score = distance(float(source.x), float(source.y), pos[0], pos[1])
                if best is None or entry_score < best[0]:
                    best = (entry_score, planet)
        if best is not None:
            return "hit", best[1]
        x, y = nx, ny

    return "miss", None


def result_for_player(replay: dict[str, Any], player: int, final_scores: list[int]) -> str:
    statuses = replay.get("statuses") or []
    if player < len(statuses) and statuses[player] != "DONE":
        return "error"
    best = max(final_scores) if final_scores else 0
    if final_scores[player] < best:
        return "loss"
    return "win" if final_scores.count(best) == 1 else "tie"


def bucket_label(turn: int) -> str:
    for start, end in BUCKETS:
        if start <= turn <= end:
            return f"t{start}_{end}"
    return "t500_plus"


def analyze_replay(path: Path, replay: dict[str, Any]) -> list[PlayerMetrics]:
    names = team_names(replay)
    player_count = len(names)
    final_obs = observation(replay, len(replay["steps"]) - 1)
    final_scores, final_counts, final_prod = scores_counts_prod(final_obs, player_count)
    rewards = replay.get("rewards") or [None for _ in range(player_count)]
    statuses = replay.get("statuses") or ["UNKNOWN" for _ in range(player_count)]
    eid = episode_id(replay, path)

    rows = [
        PlayerMetrics(
            replay=str(path),
            episode_id=eid,
            player_count=player_count,
            player=player,
            team=names[player],
            result=result_for_player(replay, player, final_scores),
            reward=rewards[player] if player < len(rewards) else None,
            status=statuses[player] if player < len(statuses) else "UNKNOWN",
            final_score=final_scores[player],
            final_planets=final_counts[player],
            final_prod=final_prod[player],
        )
        for player in range(player_count)
    ]
    sends: list[list[int]] = [[] for _ in range(player_count)]
    bucket_stats: list[dict[str, BucketStats]] = [defaultdict(BucketStats) for _ in range(player_count)]

    previous = planets_by_id(observation(replay, 0))
    for turn, states in enumerate(replay["steps"]):
        obs = observation(replay, turn)
        planets = planets_by_id(obs)
        planet_list = list(planets.values())
        scores, counts, prod = scores_counts_prod(obs, player_count)

        for player in range(player_count):
            row = rows[player]
            row.max_score = max(row.max_score, scores[player])
            row.max_planets = max(row.max_planets, counts[player])
            row.max_prod = max(row.max_prod, prod[player])
            if turn in CHECKPOINTS:
                key = f"t{turn}"
                row.score_curve[key] = scores[player]
                row.planet_curve[key] = counts[player]
                row.prod_curve[key] = prod[player]

        if turn > 0:
            for planet_id, planet in planets.items():
                old = previous.get(planet_id)
                if old is None or int(old.owner) == int(planet.owner):
                    continue
                new_owner = int(planet.owner)
                old_owner = int(old.owner)
                if 0 <= new_owner < player_count:
                    attacker = rows[new_owner]
                    if attacker.first_capture is None:
                        attacker.first_capture = turn
                    if old_owner == -1 and int(planet.production) >= 4 and attacker.first_high_prod_neutral is None:
                        attacker.first_high_prod_neutral = turn
                    if old_owner >= 0 and old_owner != new_owner and attacker.first_enemy_capture is None:
                        attacker.first_enemy_capture = turn
                if 0 <= old_owner < player_count and old_owner != new_owner and new_owner != -1:
                    loser = rows[old_owner]
                    if loser.first_planet_lost is None:
                        loser.first_planet_lost = turn
            previous = planets

        # Kaggle replay rows store each action on the row produced after that
        # action is applied. The legal source state for states[turn].action is
        # therefore observation(turn - 1), not the already-updated observation
        # at turn.
        if turn == 0:
            continue
        action_obs = observation(replay, turn - 1)
        action_planets = planets_by_id(action_obs)
        action_planet_list = list(action_planets.values())
        action_turn = int(action_obs.get("step", turn - 1))

        for player in range(min(player_count, len(states))):
            action = states[player].get("action") or []
            remaining = {
                planet_id: int(planet.ships)
                for planet_id, planet in action_planets.items()
                if int(planet.owner) == player
            }
            for move in action:
                if not isinstance(move, list) or len(move) != 3:
                    continue
                try:
                    source_id = int(move[0])
                    angle = float(move[1])
                    ships = int(move[2])
                except (TypeError, ValueError):
                    continue
                source = action_planets.get(source_id)
                available = remaining.get(source_id, 0)
                if source is None or int(source.owner) != player or ships <= 0 or ships > available:
                    continue
                remaining[source_id] = available - ships

                row = rows[player]
                row.launches += 1
                row.ships_launched += ships
                sends[player].append(ships)
                if row.first_launch is None:
                    row.first_launch = action_turn
                label = bucket_label(action_turn)
                bucket = bucket_stats[player][label]
                bucket.launches += 1
                bucket.ships += ships
                bucket.sends.append(ships)

                route_status, target = projected_target(source, angle, ships, action_planet_list, action_obs)
                if route_status == "hit" and target is not None:
                    row.projected_hits += 1
                    if int(target.id) in comet_ids(action_obs):
                        row.comet_targets += 1
                    if int(target.owner) == -1:
                        row.neutral_targets += 1
                    elif int(target.owner) == player:
                        row.friendly_targets += 1
                    else:
                        row.enemy_targets += 1
                elif route_status == "sun_oob":
                    row.projected_misses += 1
                    row.sun_or_oob_routes += 1
                else:
                    row.projected_misses += 1

    final_turn = len(replay["steps"]) - 1
    final_key = f"t{final_turn}"
    for player, row in enumerate(rows):
        row.score_curve[final_key] = final_scores[player]
        row.planet_curve[final_key] = final_counts[player]
        row.prod_curve[final_key] = final_prod[player]
        row.avg_send = statistics.fmean(sends[player]) if sends[player] else 0.0
        row.p50_send = percentile(sends[player], 0.50)
        row.p90_send = percentile(sends[player], 0.90)
        row.p99_send = percentile(sends[player], 0.99)
        for start, end in BUCKETS:
            label = f"t{start}_{end}"
            bucket = bucket_stats[player].get(label, BucketStats())
            row.bucket_launches[label] = bucket.launches
            row.bucket_ships[label] = bucket.ships
            row.bucket_p90[label] = percentile(bucket.sends, 0.90)

    return rows


def selected(rows: list[PlayerMetrics], selector: str, team: str | None) -> list[PlayerMetrics]:
    if team is not None:
        return [row for row in rows if team.lower() in row.team.lower()]
    if selector == "winners":
        return [row for row in rows if row.result == "win"]
    if selector == "losers":
        return [row for row in rows if row.result == "loss"]
    return rows


def avg(rows: list[PlayerMetrics], attr: str) -> float:
    values = [getattr(row, attr) for row in rows if getattr(row, attr) is not None]
    return float(statistics.fmean(values)) if values else 0.0


def med(rows: list[PlayerMetrics], attr: str) -> float:
    values = [getattr(row, attr) for row in rows if getattr(row, attr) is not None]
    return float(statistics.median(values)) if values else 0.0


def summarize(rows: list[PlayerMetrics]) -> dict[str, Any]:
    if not rows:
        return {"players": 0}
    hit_count = sum(row.projected_hits for row in rows)
    launch_count = sum(row.launches for row in rows)
    ships = sum(row.ships_launched for row in rows)
    target_counts = {
        "neutral": sum(row.neutral_targets for row in rows),
        "enemy": sum(row.enemy_targets for row in rows),
        "friendly": sum(row.friendly_targets for row in rows),
        "comet": sum(row.comet_targets for row in rows),
    }
    buckets: dict[str, dict[str, float]] = {}
    for start, end in BUCKETS:
        label = f"t{start}_{end}"
        buckets[label] = {
            "launches_avg": statistics.fmean(row.bucket_launches.get(label, 0) for row in rows),
            "ships_avg": statistics.fmean(row.bucket_ships.get(label, 0) for row in rows),
            "p90_send_median": statistics.median(row.bucket_p90.get(label, 0.0) for row in rows),
        }

    return {
        "players": len(rows),
        "episodes": len({row.episode_id for row in rows}),
        "win_rate": sum(1 for row in rows if row.result == "win") / len(rows),
        "avg_final_score": avg(rows, "final_score"),
        "avg_max_score": avg(rows, "max_score"),
        "avg_max_planets": avg(rows, "max_planets"),
        "avg_max_prod": avg(rows, "max_prod"),
        "median_first_capture": med(rows, "first_capture"),
        "median_first_high_prod_neutral": med(rows, "first_high_prod_neutral"),
        "median_first_enemy_capture": med(rows, "first_enemy_capture"),
        "avg_launches": avg(rows, "launches"),
        "avg_ships_launched": avg(rows, "ships_launched"),
        "avg_send": avg(rows, "avg_send"),
        "median_p90_send": med(rows, "p90_send"),
        "median_p99_send": med(rows, "p99_send"),
        "projected_hit_rate": hit_count / launch_count if launch_count else 0.0,
        "target_share": {key: value / launch_count if launch_count else 0.0 for key, value in target_counts.items()},
        "buckets": buckets,
        "avg_prod_curve": {
            checkpoint: statistics.fmean(row.prod_curve.get(checkpoint, 0) for row in rows)
            for checkpoint in sorted({key for row in rows for key in row.prod_curve})
        },
    }


def write_csv(path: Path, rows: list[PlayerMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for row in rows:
        payload = asdict(row)
        for key in ("score_curve", "planet_curve", "prod_curve", "bucket_launches", "bucket_ships", "bucket_p90"):
            payload[key] = json.dumps(payload[key], sort_keys=True)
        flat_rows.append(payload)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0].keys()) if flat_rows else [])
        if flat_rows:
            writer.writeheader()
            writer.writerows(flat_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute phase/economy/launch metrics from Orbit Wars replay JSONs.")
    parser.add_argument("replays", nargs="+", type=Path, help="Replay files or directories.")
    parser.add_argument("--limit", type=int, help="Analyze at most this many replay files after sorting.")
    parser.add_argument("--selector", choices=("winners", "losers", "all"), default="winners", help="Rows to summarize.")
    parser.add_argument("--team", help="Filter rows whose team name contains this string.")
    parser.add_argument("--json", type=Path, help="Write metrics and summary JSON.")
    parser.add_argument("--csv", type=Path, help="Write flat per-player CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows: list[PlayerMetrics] = []
    replay_paths = expand_inputs(args.replays, args.limit)
    skipped = 0
    for path in replay_paths:
        replay = read_json(path)
        if replay is None:
            skipped += 1
            continue
        try:
            all_rows.extend(analyze_replay(path, replay))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            skipped += 1
            print(f"skipped {path}: {exc}", file=sys.stderr)

    focus_rows = selected(all_rows, args.selector, args.team)
    summary = summarize(focus_rows)
    print(
        f"analyzed_replays={len(replay_paths) - skipped} skipped={skipped} "
        f"selected_players={summary.get('players', 0)} selector={args.selector}"
    )
    if focus_rows:
        print(
            "summary "
            f"win_rate={summary['win_rate']:.1%} "
            f"max_prod={summary['avg_max_prod']:.1f} "
            f"max_planets={summary['avg_max_planets']:.1f} "
            f"launches={summary['avg_launches']:.1f} "
            f"ships={summary['avg_ships_launched']:.0f} "
            f"avg_send={summary['avg_send']:.1f} "
            f"p90_send={summary['median_p90_send']:.1f} "
            f"hit_rate={summary['projected_hit_rate']:.1%}"
        )
        for label, bucket in summary["buckets"].items():
            print(
                f"  {label}: launches={bucket['launches_avg']:.1f} "
                f"ships={bucket['ships_avg']:.0f} p90={bucket['p90_send_median']:.1f}"
            )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "rows": [asdict(row) for row in focus_rows],
                    "all_row_count": len(all_rows),
                    "selected_row_count": len(focus_rows),
                    "skipped": skipped,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.csv is not None:
        write_csv(args.csv, focus_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
