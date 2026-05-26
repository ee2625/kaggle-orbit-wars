#!/usr/bin/env python3
"""Compare an agent's decisions to actions found in replay files.

This is a counterfactual policy diagnostic: for each selected replay state we
ask "what would our agent do here?" and compare launch volume, ships launched,
and coarse target mix to what the replay player actually did.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import runpy
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_metrics import (  # noqa: E402
    BUCKETS,
    Planet,
    comet_ids,
    expand_inputs,
    observation,
    planets_by_id,
    projected_target,
    read_json,
    scores_counts_prod,
    team_names,
)


AgentFn = Callable[..., list[list[Any]]]


@dataclass
class MoveSummary:
    launches: int = 0
    ships: int = 0
    sends: list[int] = field(default_factory=list)
    target_count: Counter[str] = field(default_factory=Counter)
    target_ships: Counter[str] = field(default_factory=Counter)
    invalid: int = 0


@dataclass
class BucketDiff:
    states: int = 0
    actual: MoveSummary = field(default_factory=MoveSummary)
    policy: MoveSummary = field(default_factory=MoveSummary)
    actual_only_states: int = 0
    policy_only_states: int = 0
    both_launch_states: int = 0
    both_empty_states: int = 0


@dataclass
class PolicyReplayRow:
    episode_id: str
    replay: str
    team: str
    player_count: int
    player: int
    step: int
    bucket: str
    winner: bool
    my_planets: int
    my_prod: int
    my_total_ships: int
    actual_launches: int
    actual_ships: int
    policy_launches: int
    policy_ships: int
    actual_sends: list[int]
    policy_sends: list[int]
    actual_targets: dict[str, int]
    policy_targets: dict[str, int]
    actual_target_ships: dict[str, int]
    policy_target_ships: dict[str, int]


def bucket_label(step: int) -> str:
    for start, end in BUCKETS:
        if start <= step <= end:
            return f"t{start}_{end}"
    return "t500_plus"


def load_agent(path: Path) -> AgentFn:
    namespace = runpy.run_path(str(path), run_name=f"_policy_compare_{path.stem}")
    agent = namespace.get("agent")
    if not callable(agent):
        raise ValueError(f"{path} does not define callable agent(obs)")
    return agent


def call_agent(agent: AgentFn, obs: dict[str, Any]) -> list[list[Any]]:
    signature = inspect.signature(agent)
    accepts_config = any(
        param.kind == inspect.Parameter.VAR_POSITIONAL
        for param in signature.parameters.values()
    ) or len(signature.parameters) >= 2
    config = {
        "actTimeout": 1,
        "episodeSteps": 500,
        "shipSpeed": 6.0,
        "sunRadius": 10.0,
        "boardSize": 100.0,
        "cometSpeed": 4.0,
    }
    if accepts_config:
        action = agent(obs, config)
    else:
        action = agent(obs)
    return action if isinstance(action, list) else []


def final_winners(replay: dict[str, Any], player_count: int) -> set[int]:
    rewards = replay.get("rewards")
    if isinstance(rewards, list) and len(rewards) == player_count:
        numeric = [reward for reward in rewards if reward is not None]
        if numeric:
            best = max(numeric)
            return {idx for idx, reward in enumerate(rewards) if reward == best}

    final_obs = observation(replay, len(replay["steps"]) - 1)
    scores, _, _ = scores_counts_prod(final_obs, player_count)
    best_score = max(scores) if scores else 0
    return {idx for idx, score in enumerate(scores) if score == best_score}


def player_economy(obs: dict[str, Any], player: int, player_count: int) -> tuple[int, int, int]:
    scores, counts, production = scores_counts_prod(obs, player_count)
    return counts[player], production[player], scores[player]


def summarize_moves(action: Any, obs: dict[str, Any], player: int) -> MoveSummary:
    result = MoveSummary()
    if not isinstance(action, list):
        return result

    planets = planets_by_id(obs)
    planet_list = list(planets.values())
    remaining = {
        planet_id: int(planet.ships)
        for planet_id, planet in planets.items()
        if int(planet.owner) == player
    }
    for move in action:
        if not isinstance(move, list) or len(move) != 3:
            result.invalid += 1
            continue
        try:
            source_id = int(move[0])
            angle = float(move[1])
            ships = int(move[2])
        except (TypeError, ValueError):
            result.invalid += 1
            continue
        source = planets.get(source_id)
        available = remaining.get(source_id, 0)
        if source is None or int(source.owner) != player or ships <= 0 or ships > available:
            result.invalid += 1
            continue
        remaining[source_id] = available - ships

        result.launches += 1
        result.ships += ships
        result.sends.append(ships)

        route_status, target = projected_target(source, angle, ships, planet_list, obs)
        if route_status != "hit" or target is None:
            kind = "miss"
        elif int(target.id) in comet_ids(obs):
            kind = "comet"
        elif int(target.owner) == -1:
            kind = "neutral"
        elif int(target.owner) == player:
            kind = "friendly"
        else:
            kind = "enemy"
        result.target_count[kind] += 1
        result.target_ships[kind] += ships
    return result


def add_summary(dst: MoveSummary, src: MoveSummary) -> None:
    dst.launches += src.launches
    dst.ships += src.ships
    dst.sends.extend(src.sends)
    dst.target_count.update(src.target_count)
    dst.target_ships.update(src.target_ships)
    dst.invalid += src.invalid


def replay_id(replay: dict[str, Any], path: Path) -> str:
    info = replay.get("info", {}) or {}
    return str(info.get("EpisodeId") or replay.get("id") or path.stem)


def compare_replay(
    path: Path,
    replay: dict[str, Any],
    agent: AgentFn,
    selector: str,
    max_step: int | None,
) -> list[PolicyReplayRow]:
    names = team_names(replay)
    player_count = len(replay.get("steps", [[]])[0])
    winners = final_winners(replay, player_count)
    rows: list[PolicyReplayRow] = []
    eid = replay_id(replay, path)

    # Kaggle replay rows store the action with the resulting observation. The
    # decision that appears on row t was made from the observation on row t-1.
    # Step 0 has no previous observation, so skip it.
    steps = replay.get("steps", []) or []
    for step_index in range(1, len(steps)):
        states = steps[step_index]
        obs0 = observation(replay, step_index - 1)
        step = int(obs0.get("step", step_index))
        if max_step is not None and step > max_step:
            continue

        for player in range(min(player_count, len(states))):
            is_winner = player in winners
            if selector == "winners" and not is_winner:
                continue
            if selector == "losers" and is_winner:
                continue

            obs = copy.deepcopy(obs0)
            obs["player"] = player
            actual = summarize_moves(states[player].get("action") or [], obs, player)
            try:
                policy_action = call_agent(agent, obs)
            except Exception:
                policy_action = []
            policy = summarize_moves(policy_action, obs, player)
            my_planets, my_prod, my_total_ships = player_economy(obs, player, player_count)

            rows.append(
                PolicyReplayRow(
                    episode_id=eid,
                    replay=str(path),
                    team=names[player] if player < len(names) else f"player{player}",
                    player_count=player_count,
                    player=player,
                    step=step,
                    bucket=bucket_label(step),
                    winner=is_winner,
                    my_planets=my_planets,
                    my_prod=my_prod,
                    my_total_ships=my_total_ships,
                    actual_launches=actual.launches,
                    actual_ships=actual.ships,
                    policy_launches=policy.launches,
                    policy_ships=policy.ships,
                    actual_sends=list(actual.sends),
                    policy_sends=list(policy.sends),
                    actual_targets=dict(actual.target_count),
                    policy_targets=dict(policy.target_count),
                    actual_target_ships=dict(actual.target_ships),
                    policy_target_ships=dict(policy.target_ships),
                )
            )
    return rows


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    low = int(index)
    high = min(len(ordered) - 1, low + 1)
    if low == high:
        return float(ordered[low])
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def summarize_rows(rows: list[PolicyReplayRow]) -> dict[str, Any]:
    buckets: dict[str, BucketDiff] = defaultdict(BucketDiff)
    overall = BucketDiff()

    for row in rows:
        for bucket in (overall, buckets[row.bucket]):
            bucket.states += 1
            actual = MoveSummary(
                launches=row.actual_launches,
                ships=row.actual_ships,
                sends=list(row.actual_sends),
                target_count=Counter(row.actual_targets),
                target_ships=Counter(row.actual_target_ships),
            )
            policy = MoveSummary(
                launches=row.policy_launches,
                ships=row.policy_ships,
                sends=list(row.policy_sends),
                target_count=Counter(row.policy_targets),
                target_ships=Counter(row.policy_target_ships),
            )
            add_summary(bucket.actual, actual)
            add_summary(bucket.policy, policy)
            if row.actual_launches and not row.policy_launches:
                bucket.actual_only_states += 1
            elif row.policy_launches and not row.actual_launches:
                bucket.policy_only_states += 1
            elif row.actual_launches and row.policy_launches:
                bucket.both_launch_states += 1
            else:
                bucket.both_empty_states += 1

    def render(diff: BucketDiff) -> dict[str, Any]:
        states = max(1, diff.states)
        actual_launches = diff.actual.launches
        policy_launches = diff.policy.launches
        actual_ships = diff.actual.ships
        policy_ships = diff.policy.ships
        return {
            "states": diff.states,
            "actual_launches_per_state": actual_launches / states,
            "policy_launches_per_state": policy_launches / states,
            "actual_ships_per_state": actual_ships / states,
            "policy_ships_per_state": policy_ships / states,
            "actual_avg_send": (actual_ships / actual_launches) if actual_launches else 0.0,
            "policy_avg_send": (policy_ships / policy_launches) if policy_launches else 0.0,
            "actual_p90_send": percentile(diff.actual.sends, 0.90),
            "policy_p90_send": percentile(diff.policy.sends, 0.90),
            "launch_ratio_policy_over_actual": policy_launches / actual_launches if actual_launches else None,
            "ship_ratio_policy_over_actual": policy_ships / actual_ships if actual_ships else None,
            "actual_only_state_share": diff.actual_only_states / states,
            "policy_only_state_share": diff.policy_only_states / states,
            "both_launch_state_share": diff.both_launch_states / states,
            "both_empty_state_share": diff.both_empty_states / states,
            "actual_target_share": {
                key: value / actual_launches if actual_launches else 0.0
                for key, value in sorted(diff.actual.target_count.items())
            },
            "policy_target_share": {
                key: value / policy_launches if policy_launches else 0.0
                for key, value in sorted(diff.policy.target_count.items())
            },
            "actual_target_ship_share": {
                key: value / actual_ships if actual_ships else 0.0
                for key, value in sorted(diff.actual.target_ships.items())
            },
            "policy_target_ship_share": {
                key: value / policy_ships if policy_ships else 0.0
                for key, value in sorted(diff.policy.target_ships.items())
            },
        }

    launch_deltas = [row.policy_launches - row.actual_launches for row in rows]
    ship_deltas = [row.policy_ships - row.actual_ships for row in rows]
    return {
        "rows": len(rows),
        "episodes": len({row.episode_id for row in rows}),
        "players": len({(row.episode_id, row.player) for row in rows}),
        "overall": render(overall),
        "buckets": {label: render(buckets[label]) for label in sorted(buckets)},
        "delta_distribution": {
            "launch_delta_avg": statistics.fmean(launch_deltas) if launch_deltas else 0.0,
            "ship_delta_avg": statistics.fmean(ship_deltas) if ship_deltas else 0.0,
            "ship_delta_p10": percentile(ship_deltas, 0.10),
            "ship_delta_p50": percentile(ship_deltas, 0.50),
            "ship_delta_p90": percentile(ship_deltas, 0.90),
        },
    }


def write_csv(path: Path, rows: list[PolicyReplayRow]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(asdict(rows[0]).keys()) if rows else []
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            for row in rows:
                payload = asdict(row)
                payload["actual_targets"] = json.dumps(payload["actual_targets"], sort_keys=True)
                payload["policy_targets"] = json.dumps(payload["policy_targets"], sort_keys=True)
                payload["actual_sends"] = json.dumps(payload["actual_sends"])
                payload["policy_sends"] = json.dumps(payload["policy_sends"])
                writer.writerow(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Counterfactually compare an agent to replay actions.")
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--agent", type=Path, default=Path("main.py"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-step", type=int, default=200)
    parser.add_argument("--selector", choices=("winners", "losers", "all"), default="winners")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    agent = load_agent(args.agent)
    rows: list[PolicyReplayRow] = []
    skipped = 0
    for path in expand_inputs(args.replays, args.limit):
        replay = read_json(path)
        if replay is None:
            skipped += 1
            continue
        try:
            rows.extend(compare_replay(path, replay, agent, args.selector, args.max_step))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            skipped += 1
            print(f"skipped {path}: {exc}", file=sys.stderr)

    summary = summarize_rows(rows)
    print(
        f"compared_rows={summary['rows']} episodes={summary['episodes']} "
        f"players={summary['players']} skipped={skipped} selector={args.selector}"
    )
    overall = summary["overall"]
    print(
        "overall "
        f"actual_launches/state={overall['actual_launches_per_state']:.3f} "
        f"policy_launches/state={overall['policy_launches_per_state']:.3f} "
        f"actual_ships/state={overall['actual_ships_per_state']:.1f} "
        f"policy_ships/state={overall['policy_ships_per_state']:.1f} "
        f"actual_p90={overall['actual_p90_send']:.1f} "
        f"policy_p90={overall['policy_p90_send']:.1f} "
        f"ship_ratio={overall['ship_ratio_policy_over_actual']}"
    )
    for label, bucket in summary["buckets"].items():
        print(
            f"  {label}: actual ships/state={bucket['actual_ships_per_state']:.1f} "
            f"policy={bucket['policy_ships_per_state']:.1f} "
            f"actual launches/state={bucket['actual_launches_per_state']:.3f} "
            f"policy={bucket['policy_launches_per_state']:.3f} "
            f"actual_p90={bucket['actual_p90_send']:.1f} "
            f"policy_p90={bucket['policy_p90_send']:.1f}"
        )

    payload = {
        "summary": summary,
        "rows": [asdict(row) for row in rows],
        "skipped": skipped,
        "agent": str(args.agent),
        "selector": args.selector,
        "max_step": args.max_step,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.csv is not None:
        write_csv(args.csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
