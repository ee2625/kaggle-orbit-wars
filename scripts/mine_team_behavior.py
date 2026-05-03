#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import Fleet, fleet_speed, projected_fleet_target  # noqa: E402
from scripts.analyze_replay import (  # noqa: E402
    load_replay,
    owned_planet_counts,
    owned_production,
    planets_by_id,
    scores_from_obs,
    team_names,
    world_observation,
)


@dataclass
class TeamBehavior:
    team: str
    episodes: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    player_counts: Counter[int] = field(default_factory=Counter)
    first_captures: list[int] = field(default_factory=list)
    max_productions: list[int] = field(default_factory=list)
    launches: int = 0
    ships_launched: int = 0
    early_launches: int = 0
    early_ships_launched: int = 0
    target_types: Counter[str] = field(default_factory=Counter)
    early_targets: Counter[str] = field(default_factory=Counter)
    opening_targets: Counter[str] = field(default_factory=Counter)
    launch_by_turn_bucket: Counter[str] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)


def expand_replays(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*replay.json")))
        else:
            expanded.append(path)
    return expanded


def classify_result(rewards: list[Any], player: int) -> str:
    if not rewards or player >= len(rewards) or rewards[player] is None:
        return "unknown"
    best = max(reward for reward in rewards if reward is not None)
    if rewards[player] != best:
        return "loss"
    if sum(1 for reward in rewards if reward == best) > 1:
        return "tie"
    return "win"


def turn_bucket(turn: int) -> str:
    if turn < 25:
        return "000-024"
    if turn < 50:
        return "025-049"
    if turn < 100:
        return "050-099"
    if turn < 200:
        return "100-199"
    return "200+"


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


def target_label(target_owner: int, player: int, planet_id: int, comet_ids: set[int]) -> str:
    suffix = "_comet" if planet_id in comet_ids else ""
    if target_owner == -1:
        return f"neutral{suffix}"
    if target_owner == player:
        return f"friendly{suffix}"
    return f"enemy{suffix}"


def update_launches(replay: dict[str, Any], player: int, behavior: TeamBehavior, early_turns: int) -> None:
    opening_seen = 0
    for turn, states in enumerate(replay["steps"]):
        obs = world_observation(replay, turn)
        planets = list(planets_by_id(obs).values())
        planet_map = {planet.id: planet for planet in planets}
        angular_velocity = float(obs.get("angular_velocity", 0.0) or 0.0)
        comet_ids = {int(value) for value in obs.get("comet_planet_ids", []) or []}
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

            behavior.launches += 1
            behavior.ships_launched += ships
            if turn <= early_turns:
                behavior.early_launches += 1
                behavior.early_ships_launched += ships
            behavior.launch_by_turn_bucket[turn_bucket(turn)] += 1

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
                label = "miss"
                detail = f"MISS ships={ships}"
            else:
                label = target_label(target.owner, player, target.id, comet_ids)
                detail = f"p{target.id}:{label}:prod{target.production}:g{target.ships}:ships{ships}"

            behavior.target_types[label] += 1
            if turn <= early_turns:
                behavior.early_targets[label] += 1
            if opening_seen < 8:
                behavior.opening_targets[f"t{turn}:{detail}"] += 1
                opening_seen += 1


def update_behavior(replay_path: Path, replay: dict[str, Any], team: str, behavior: TeamBehavior, early_turns: int) -> None:
    names = team_names(replay)
    matching = [index for index, name in enumerate(names) if name == team]
    if not matching:
        return

    player_count = len(names)
    rewards = replay.get("rewards") or [None for _ in range(player_count)]
    episode_id = (replay.get("info", {}) or {}).get("EpisodeId", replay_path.stem)

    for player in matching:
        behavior.episodes += 1
        behavior.player_counts[player_count] += 1
        result = classify_result(rewards, player)
        if result == "win":
            behavior.wins += 1
        elif result == "loss":
            behavior.losses += 1
        elif result == "tie":
            behavior.ties += 1

        capture_turn = first_capture_turn(replay, player)
        if capture_turn is not None:
            behavior.first_captures.append(capture_turn)
        behavior.max_productions.append(max_production(replay, player, player_count))
        update_launches(replay, player, behavior, early_turns)

        final_scores = scores_from_obs(world_observation(replay, len(replay["steps"]) - 1), player_count)
        final_counts = owned_planet_counts(world_observation(replay, len(replay["steps"]) - 1), player_count, set())
        if len(behavior.examples) < 12:
            behavior.examples.append(
                f"episode={episode_id} seat={player} players={player_count} result={result} "
                f"score={final_scores[player]} planets={final_counts[player]} replay={replay_path}"
            )


def mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_behavior(behavior: TeamBehavior) -> None:
    print(f"Team: {behavior.team}")
    print(f"Episodes: {behavior.episodes}  W-T-L: {behavior.wins}-{behavior.ties}-{behavior.losses}")
    if behavior.episodes == 0:
        print("No matching local replays found.")
        return

    print(f"Player counts: {dict(sorted(behavior.player_counts.items()))}")
    print(f"Avg first capture: {mean(behavior.first_captures):.1f}")
    print(f"Avg max production: {mean(behavior.max_productions):.1f}")
    print(f"Launches: {behavior.launches}  ships launched: {behavior.ships_launched}")
    print(f"Early launches: {behavior.early_launches}  early ships: {behavior.early_ships_launched}")
    print(f"Target mix: {dict(behavior.target_types.most_common())}")
    print(f"Early target mix: {dict(behavior.early_targets.most_common())}")
    print(f"Launch timing: {dict(sorted(behavior.launch_by_turn_bucket.items()))}")

    print("\nOpening signatures")
    for signature, count in behavior.opening_targets.most_common(16):
        print(f"  {count:>3}x {signature}")

    print("\nExamples")
    for example in behavior.examples:
        print(f"  {example}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine public replay behavior for a specific Orbit Wars team.")
    parser.add_argument("team", help="Exact team name to analyze.")
    parser.add_argument("replays", nargs="+", type=Path, help="Replay files or directories.")
    parser.add_argument("--early-turns", type=int, default=80, help="Turns counted as opening/early game.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    behavior = TeamBehavior(team=args.team)
    for replay_path in expand_replays(args.replays):
        try:
            replay = load_replay(replay_path)
        except (OSError, json.JSONDecodeError):
            continue
        update_behavior(replay_path, replay, args.team, behavior, args.early_turns)

    print_behavior(behavior)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
