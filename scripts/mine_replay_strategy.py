#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import Fleet  # noqa: E402
from scripts.analyze_replay import (  # noqa: E402
    build_player_stats,
    load_replay,
    planets_by_id,
    project_fleet_target_for_obs,
    team_names,
    world_observation,
)


BUCKETS = ((0, 49), (50, 99), (100, 199), (200, 349), (350, 10_000))
SIZE_BUCKETS = ((1, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 10_000))


def bucket_label(turn: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= turn <= hi:
            return f"t{lo}-{hi if hi < 10_000 else 'end'}"
    return "unknown"


def size_label(ships: int) -> str:
    for lo, hi in SIZE_BUCKETS:
        if lo <= ships <= hi:
            return f"{lo}-{hi if hi < 10_000 else 'plus'}"
    return "0"


def pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return int(values[index])


def replay_id(replay: dict[str, Any], path: Path) -> str:
    info = replay.get("info", {}) or {}
    return str(info.get("EpisodeId") or replay.get("id") or path.stem)


def mine(path: Path, replay: dict[str, Any]) -> None:
    names = team_names(replay)
    stats = build_player_stats(replay, early_turns=80)
    player_count = len(names)

    per_player = [
        {
            "sizes": [],
            "size_hist": Counter(),
            "target_counts": Counter(),
            "target_ships": Counter(),
            "bucket_counts": defaultdict(Counter),
            "bucket_ships": defaultdict(Counter),
            "first_actions": [],
        }
        for _ in range(player_count)
    ]

    for turn, states in enumerate(replay["steps"]):
        obs = world_observation(replay, turn - 1 if turn > 0 else turn)
        planets = list(planets_by_id(obs).values())
        planet_map = {planet.id: planet for planet in planets}
        angular_velocity = float(obs.get("angular_velocity", 0.0) or 0.0)
        comet_ids = {int(value) for value in obs.get("comet_planet_ids", []) or []}

        for player in range(player_count):
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
                target = project_fleet_target_for_obs(
                    fake,
                    planets,
                    obs,
                    angular_velocity,
                    comet_ids,
                    max_turns=160,
                )
                if target is None:
                    kind = "miss"
                    target_text = "miss"
                elif target.id in comet_ids:
                    kind = "comet"
                    target_text = f"p{target.id}:comet"
                elif target.owner == -1:
                    kind = "neutral"
                    target_text = f"p{target.id}:N{int(target.production)}"
                elif target.owner == player:
                    kind = "friendly"
                    target_text = f"p{target.id}:F{int(target.production)}"
                else:
                    kind = "enemy"
                    target_text = f"p{target.id}:E{int(target.production)}"

                row = per_player[player]
                row["sizes"].append(ships)
                row["size_hist"][size_label(ships)] += 1
                row["target_counts"][kind] += 1
                row["target_ships"][kind] += ships
                row["bucket_counts"][bucket_label(turn)][kind] += 1
                row["bucket_ships"][bucket_label(turn)][kind] += ships
                if len(row["first_actions"]) < 10:
                    row["first_actions"].append(f"t{turn}:{ships}->{target_text}")

    print(f"\n{path}")
    print(f"episode={replay_id(replay, path)} players={player_count} steps={len(replay['steps'])}")
    print("teams:", " | ".join(f"{idx}:{name}" for idx, name in enumerate(names)))

    for player, name in enumerate(names):
        base = stats[player]
        mined = per_player[player]
        sizes = mined["sizes"]
        avg_size = statistics.fmean(sizes) if sizes else 0.0
        print(
            f"\nP{player} {name} reward={base.reward} final={base.final_score} "
            f"max_prod={base.max_production} max_planets={base.max_planets}"
        )
        print(
            f"  launches={len(sizes)} ships={sum(sizes)} avg={avg_size:.1f} "
            f"p50/p90/p99={pct(sizes, .50)}/{pct(sizes, .90)}/{pct(sizes, .99)} "
            f"sizes={dict(mined['size_hist'])}"
        )
        print(
            f"  targets count={dict(mined['target_counts'])} "
            f"ships={dict(mined['target_ships'])}"
        )
        for label in [bucket_label(lo) for lo, _ in BUCKETS]:
            if mined["bucket_counts"].get(label):
                print(
                    f"  {label}: count={dict(mined['bucket_counts'][label])} "
                    f"ships={dict(mined['bucket_ships'][label])}"
                )
        print("  first actions:", "; ".join(mined["first_actions"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine observable strategy features from Orbit Wars replays.")
    parser.add_argument("replays", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.replays:
        mine(path, load_replay(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
