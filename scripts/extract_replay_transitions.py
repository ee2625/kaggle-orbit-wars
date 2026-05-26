#!/usr/bin/env python3
"""Extract Orbit Wars replay transitions for diagnostics or imitation learning."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any, Iterator


def iter_replays(paths: list[Path], limit: int | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
    seen = 0
    for path in paths:
        if limit is not None and seen >= limit:
            return
        if path.is_dir():
            for child in sorted(path.rglob("*.json")):
                if limit is not None and seen >= limit:
                    return
                yield child.name, json.loads(child.read_text(encoding="utf-8"))
                seen += 1
            continue

        if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
            with tarfile.open(path, "r:gz") as archive:
                members = [
                    member
                    for member in archive.getmembers()
                    if member.isfile() and member.name.endswith(".json")
                ]
                for member in sorted(members, key=lambda item: item.name):
                    if limit is not None and seen >= limit:
                        return
                    fileobj = archive.extractfile(member)
                    if fileobj is None:
                        continue
                    yield member.name, json.load(fileobj)
                    seen += 1
            continue

        if path.suffix == ".json":
            yield path.name, json.loads(path.read_text(encoding="utf-8"))
            seen += 1


def owner_stats(obs: dict[str, Any], player: int, player_count: int) -> dict[str, Any]:
    planets = obs.get("planets", []) or []
    fleets = obs.get("fleets", []) or []

    my_planets = enemy_planets = neutral_planets = 0
    my_ships_planets = enemy_ships_planets = neutral_ships = 0
    my_production = enemy_production = neutral_production = 0
    my_fleet_ships = enemy_fleet_ships = 0
    my_fleets = enemy_fleets = 0

    for planet in planets:
        owner = int(planet[1])
        ships = int(planet[5])
        production = int(planet[6])
        if owner == player:
            my_planets += 1
            my_ships_planets += ships
            my_production += production
        elif owner == -1:
            neutral_planets += 1
            neutral_ships += ships
            neutral_production += production
        else:
            enemy_planets += 1
            enemy_ships_planets += ships
            enemy_production += production

    for fleet in fleets:
        owner = int(fleet[1])
        ships = int(fleet[6])
        if owner == player:
            my_fleets += 1
            my_fleet_ships += ships
        elif 0 <= owner < player_count:
            enemy_fleets += 1
            enemy_fleet_ships += ships

    return {
        "planet_count": len(planets),
        "fleet_count": len(fleets),
        "my_planets": my_planets,
        "enemy_planets": enemy_planets,
        "neutral_planets": neutral_planets,
        "my_ships_planets": my_ships_planets,
        "enemy_ships_planets": enemy_ships_planets,
        "neutral_ships": neutral_ships,
        "my_fleet_ships": my_fleet_ships,
        "enemy_fleet_ships": enemy_fleet_ships,
        "my_fleets": my_fleets,
        "enemy_fleets": enemy_fleets,
        "my_total_ships": my_ships_planets + my_fleet_ships,
        "enemy_total_ships": enemy_ships_planets + enemy_fleet_ships,
        "my_production": my_production,
        "enemy_production": enemy_production,
        "neutral_production": neutral_production,
    }


def action_stats(action: Any) -> dict[str, int]:
    if not isinstance(action, list):
        return {"launches": 0, "ships_launched": 0}
    launches = 0
    ships = 0
    for move in action:
        if not isinstance(move, list) or len(move) != 3:
            continue
        launches += 1
        try:
            ships += int(move[2])
        except (TypeError, ValueError):
            pass
    return {"launches": launches, "ships_launched": ships}


def final_rewards(replay: dict[str, Any], player_count: int) -> list[float | int | None]:
    rewards = replay.get("rewards")
    if isinstance(rewards, list) and len(rewards) == player_count:
        return rewards
    steps = replay.get("steps", []) or []
    if not steps:
        return [None] * player_count
    return [state.get("reward") for state in steps[-1]]


def extract_replay(
    name: str,
    replay: dict[str, Any],
    winner_only: bool,
    include_observation: bool,
) -> Iterator[dict[str, Any]]:
    steps = replay.get("steps", []) or []
    if not steps:
        return

    episode_id = replay.get("id") or Path(name).stem
    player_count = len(steps[0])
    rewards = final_rewards(replay, player_count)
    numeric_rewards = [reward for reward in rewards if reward is not None]
    best_reward = max(numeric_rewards) if numeric_rewards else None
    winners = {
        player
        for player, reward in enumerate(rewards)
        if best_reward is not None and reward == best_reward
    }

    for step_index, step in enumerate(steps):
        for player, state in enumerate(step):
            if winner_only and player not in winners:
                continue
            obs = state.get("observation") or {}
            if not isinstance(obs, dict):
                continue
            action = state.get("action") or []
            record = {
                "episode_id": episode_id,
                "source": name,
                "step": int(obs.get("step", step_index)),
                "player": player,
                "player_count": player_count,
                "winner": player in winners,
                "final_reward": rewards[player] if player < len(rewards) else None,
                "status": state.get("status"),
                "action": action,
                "features": {
                    **owner_stats(obs, player, player_count),
                    **action_stats(action),
                },
            }
            if include_observation:
                record["observation"] = obs
            yield record


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract replay transitions into JSONL.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Replay JSON files, directories, or episodes.tar.gz files.")
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--winner-only", action="store_true")
    parser.add_argument("--include-observation", action="store_true")
    args = parser.parse_args()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "episodes": 0,
        "transitions": 0,
        "winner_only": args.winner_only,
        "include_observation": args.include_observation,
        "player_counts": {},
        "launches": 0,
        "ships_launched": 0,
    }

    with args.out_jsonl.open("w", encoding="utf-8") as out:
        for name, replay in iter_replays(args.inputs, args.limit_episodes):
            summary["episodes"] += 1
            seen_player_counts = set()
            for record in extract_replay(
                name,
                replay,
                winner_only=args.winner_only,
                include_observation=args.include_observation,
            ):
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
                summary["transitions"] += 1
                features = record["features"]
                summary["launches"] += features["launches"]
                summary["ships_launched"] += features["ships_launched"]
                seen_player_counts.add(str(record["player_count"]))
            for player_count in seen_player_counts:
                summary["player_counts"][player_count] = summary["player_counts"].get(player_count, 0) + 1

    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
