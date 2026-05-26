#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    build_player_stats,
    load_replay,
    planets_by_id,
    team_names,
    world_observation,
)


TURN_BUCKETS = ((0, 49), (50, 99), (100, 199), (200, 349), (350, 10_000))


@dataclass
class PlayerFeatures:
    episode: str
    player: int
    team: str
    players: int
    steps: int
    is_winner: bool
    reward: float | int | None
    final_score: int
    max_prod: int
    max_planets: int
    first_capture: int | None
    first_lost: int | None
    first_high_prod_neutral: int | None = None
    first_enemy_capture: int | None = None
    first_high_prod_enemy_capture: int | None = None
    launches: int = 0
    ships: int = 0
    sizes: list[int] = field(default_factory=list)
    target_counts: Counter[str] = field(default_factory=Counter)
    target_ships: Counter[str] = field(default_factory=Counter)
    bucket_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    bucket_ships: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    first_actions: list[str] = field(default_factory=list)

    @property
    def avg_send(self) -> float:
        return self.ships / self.launches if self.launches else 0.0

    @property
    def enemy_ship_share(self) -> float:
        return self.target_ships["enemy"] / self.ships if self.ships else 0.0

    @property
    def friendly_ship_share(self) -> float:
        return self.target_ships["friendly"] / self.ships if self.ships else 0.0

    @property
    def miss_ship_share(self) -> float:
        return self.target_ships["miss"] / self.ships if self.ships else 0.0


def pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return int(ordered[index])


def bucket_label(turn: int) -> str:
    for low, high in TURN_BUCKETS:
        if low <= turn <= high:
            return f"t{low}-{high if high < 10_000 else 'end'}"
    return "unknown"


def episode_id(replay: dict[str, Any], path: Path) -> str:
    info = replay.get("info", {}) or {}
    return str(info.get("EpisodeId") or replay.get("id") or path.stem)


def winner_flags(replay: dict[str, Any], player_count: int) -> list[bool]:
    rewards = replay.get("rewards") or []
    if len(rewards) == player_count and any(reward is not None for reward in rewards):
        best = max(reward for reward in rewards if reward is not None)
        return [reward == best for reward in rewards]

    stats = build_player_stats(replay, early_turns=0)
    best_score = max(row.final_score for row in stats)
    return [row.final_score == best_score for row in stats]


def target_kind(target: Any | None, player: int, comet_ids: set[int]) -> tuple[str, str]:
    if target is None:
        return "miss", "miss"
    if target.id in comet_ids:
        return "comet", f"p{target.id}:C{int(target.production)}"
    if target.owner == -1:
        return "neutral", f"p{target.id}:N{int(target.production)}"
    if target.owner == player:
        return "friendly", f"p{target.id}:F{int(target.production)}"
    return "enemy", f"p{target.id}:E{int(target.production)}"


def collect_features(path: Path) -> list[PlayerFeatures]:
    replay = load_replay(path)
    names = team_names(replay)
    player_count = len(names)
    stats = build_player_stats(replay, early_turns=80)
    winners = winner_flags(replay, player_count)
    eid = episode_id(replay, path)
    rows = [
        PlayerFeatures(
            episode=eid,
            player=player,
            team=names[player],
            players=player_count,
            steps=len(replay["steps"]),
            is_winner=winners[player],
            reward=stats[player].reward,
            final_score=stats[player].final_score,
            max_prod=stats[player].max_production,
            max_planets=stats[player].max_planets,
            first_capture=stats[player].first_capture_turn,
            first_lost=stats[player].first_enemy_capture_turn,
        )
        for player in range(player_count)
    ]

    previous = planets_by_id(world_observation(replay, 0))
    for turn in range(1, len(replay["steps"])):
        current = planets_by_id(world_observation(replay, turn))
        for planet_id, planet in current.items():
            old = previous.get(planet_id)
            if old is None or old.owner == planet.owner:
                continue
            if 0 <= planet.owner < player_count:
                row = rows[int(planet.owner)]
                if old.owner == -1 and planet.production >= 4 and row.first_high_prod_neutral is None:
                    row.first_high_prod_neutral = turn
                if old.owner not in (-1, planet.owner):
                    if row.first_enemy_capture is None:
                        row.first_enemy_capture = turn
                    if planet.production >= 3 and row.first_high_prod_enemy_capture is None:
                        row.first_high_prod_enemy_capture = turn
        previous = current

    for turn, states in enumerate(replay["steps"]):
        obs = world_observation(replay, turn)
        planets = list(planets_by_id(obs).values())
        planet_map = {planet.id: planet for planet in planets}
        angular_velocity = float(obs.get("angular_velocity", 0.0) or 0.0)
        comet_ids = {int(value) for value in obs.get("comet_planet_ids", []) or []}

        for player in range(player_count):
            row = rows[player]
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
                kind, text = target_kind(target, player, comet_ids)
                label = bucket_label(turn)
                row.launches += 1
                row.ships += ships
                row.sizes.append(ships)
                row.target_counts[kind] += 1
                row.target_ships[kind] += ships
                row.bucket_counts[label][kind] += 1
                row.bucket_ships[label][kind] += ships
                if len(row.first_actions) < 8:
                    row.first_actions.append(f"t{turn}:{ships}->{text}")

    return rows


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize_group(name: str, rows: list[PlayerFeatures]) -> list[str]:
    if not rows:
        return [f"{name}: no rows"]
    return [
        (
            f"{name}: n={len(rows)} avg_max_prod={mean([r.max_prod for r in rows]):.1f} "
            f"avg_launches={mean([r.launches for r in rows]):.1f} "
            f"avg_ships={mean([r.ships for r in rows]):.0f} "
            f"avg_send={mean([r.avg_send for r in rows]):.1f} "
            f"enemy_ship_share={mean([r.enemy_ship_share for r in rows]):.1%} "
            f"friendly_ship_share={mean([r.friendly_ship_share for r in rows]):.1%} "
            f"miss_ship_share={mean([r.miss_ship_share for r in rows]):.1%}"
        )
    ]


def format_optional(value: int | None) -> str:
    return "-" if value is None else str(value)


def build_report(rows: list[PlayerFeatures]) -> str:
    winners = [row for row in rows if row.is_winner]
    non_winners = [row for row in rows if not row.is_winner]
    lines: list[str] = []
    lines.append("# Orbit Wars Replay Alpha Report")
    lines.append("")
    lines.append(f"Analyzed {len({row.episode for row in rows})} episodes and {len(rows)} player seats.")
    lines.append("")
    lines.append("## Winner Rows")
    lines.append("")
    lines.append(
        "| episode | winner | players | steps | final | max_prod | first_cap | first_p4N | first_enemy | launches | ships | avg | p50 | p90 | p99 | enemy ships | friendly ships | miss ships |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in sorted(winners, key=lambda item: (item.players, item.episode, item.player)):
        lines.append(
            f"| {row.episode} | {row.team} p{row.player} | {row.players} | {row.steps} | "
            f"{row.final_score} | {row.max_prod} | {format_optional(row.first_capture)} | "
            f"{format_optional(row.first_high_prod_neutral)} | "
            f"{format_optional(row.first_high_prod_enemy_capture or row.first_enemy_capture)} | "
            f"{row.launches} | {row.ships} | {row.avg_send:.1f} | {pct(row.sizes, .50)} | "
            f"{pct(row.sizes, .90)} | {pct(row.sizes, .99)} | "
            f"{row.target_ships['enemy']} | {row.target_ships['friendly']} | {row.target_ships['miss']} |"
        )
    lines.append("")
    lines.append("## Group Averages")
    lines.append("")
    lines.extend(summarize_group("Winners", winners))
    lines.extend(summarize_group("Non-winners", non_winners))
    lines.extend(summarize_group("4P winners", [row for row in winners if row.players >= 4]))
    lines.extend(summarize_group("2P winners", [row for row in winners if row.players == 2]))
    lines.append("")
    lines.append("## Winner Turn Buckets")
    lines.append("")
    for row in sorted(winners, key=lambda item: (item.players, item.episode, item.player)):
        lines.append(f"- {row.episode} {row.team}:")
        for low, _high in TURN_BUCKETS:
            label = bucket_label(low)
            counts = dict(row.bucket_counts.get(label, {}))
            ships = dict(row.bucket_ships.get(label, {}))
            if counts:
                lines.append(f"  - {label}: count={counts} ships={ships}")
        lines.append(f"  - first actions: {'; '.join(row.first_actions)}")
    lines.append("")
    lines.append("## Candidate Signals")
    lines.append("")
    lines.append("- Top wins usually secure a production-4/5 neutral early, but the timing varies by map. The important marker is not just first capture; it is reaching 35-80 production before the field can stabilize.")
    lines.append("- Strong 4P wins convert surplus into enemy pressure. Across these winner seats, enemy-targeted ships are comparable to or higher than friendly-transfer ships; passive rear accumulation is death.")
    lines.append("- The best seats use larger mature packets. The scary winners show p90 send sizes from roughly 68 up to 223, while weak seats often stay at low p90 or waste volume into misses.")
    lines.append("- Friendly funneling is still alpha. Winners are not pure attackers: they route thousands of ships through owned planets before the kill phase.")
    lines.append("- High miss share is not automatically bad because moving planets make projection fuzzy, but repeated huge misses before t50 are a warning sign. We should validate routes for big packets more carefully than for cheap probes.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate alpha signals from public Orbit Wars replays.")
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--markdown", type=Path, help="Write a markdown report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[PlayerFeatures] = []
    for replay_path in args.replays:
        rows.extend(collect_features(replay_path))
    report = build_report(rows)
    print(report)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report, encoding="utf-8")
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
