#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.league_backtest import (  # noqa: E402
    Competitor,
    CompetitorStats,
    Rating,
    adjusted_scores,
    build_competitors,
    build_matchups,
    classify_outcomes,
    leaderboard_rows,
    print_leaderboard,
    rank_scores,
    rating_snapshot,
    run_league_episode_worker,
    update_pairwise_ratings,
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def seed_from_replay(path: Path) -> int | None:
    payload = read_json(path)
    if payload is None:
        return None
    info = payload.get("info") or {}
    for key in ("seed", "Seed", "CodexSeed"):
        value = info.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def seeds_from_replays(paths: list[Path], limit: int | None) -> list[int]:
    seeds: list[int] = []
    seen: set[int] = set()
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        else:
            files.append(path)

    for path in files:
        if path.name == "dataset-metadata.json":
            continue
        seed = seed_from_replay(path)
        if seed is None or seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
        if limit is not None and len(seeds) >= limit:
            break
    return seeds


def explicit_seeds(values: list[str]) -> list[int]:
    seeds: list[int] = []
    seen: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            seed = int(part)
            if seed not in seen:
                seen.add(seed)
                seeds.append(seed)
    return seeds


def rotated(lineup: tuple[Competitor, ...], offset: int) -> list[Competitor]:
    offset %= len(lineup)
    return list(lineup[offset:] + lineup[:offset])


def planned_lineups(competitors: list[Competitor], players: int, rotations: int) -> list[tuple[Competitor, ...]]:
    matchups = build_matchups(competitors, players)
    plans: list[tuple[Competitor, ...]] = []
    for matchup in matchups:
        count = players if rotations <= 0 else min(rotations, players)
        for offset in range(count):
            plans.append(tuple(rotated(matchup, offset)))
    return plans


def run_plan(
    plans: list[tuple[Competitor, ...]],
    seeds: list[int],
    episode_steps: int | None,
    debug: bool,
    verbose_env: bool,
    jobs: int,
    backend: str,
    fast_act_timeout: float,
) -> list[dict[str, Any]]:
    payloads = []
    episode = 1
    for seed, lineup in itertools.product(seeds, plans):
        payloads.append(
            (
                list(lineup),
                episode,
                seed,
                episode_steps,
                debug,
                verbose_env,
                backend,
                fast_act_timeout,
            )
        )
        episode += 1

    results: dict[int, tuple[int, list[int], list[float | int | None], list[str], float, int]] = {}
    if jobs <= 1:
        for payload in payloads:
            results[payload[1]] = run_league_episode_worker(payload)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_league_episode_worker, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                results[row[0]] = row

    rows: list[dict[str, Any]] = []
    for payload in payloads:
        lineup, episode, seed = payload[0], payload[1], payload[2]
        _, scores, rewards, statuses, duration_s, steps = results[episode]
        scores = adjusted_scores(scores, statuses)
        rows.append(
            {
                "episode": episode,
                "seed": seed,
                "lineup": [competitor.label for competitor in lineup],
                "specs": [competitor.spec for competitor in lineup],
                "scores": scores,
                "ranks": rank_scores(scores),
                "rewards": rewards,
                "statuses": statuses,
                "duration_s": duration_s,
                "steps": steps,
            }
        )
    return rows


def summarize(competitors: list[Competitor], episodes: list[dict[str, Any]], beta: float) -> list[dict[str, Any]]:
    ratings = {competitor.label: Rating() for competitor in competitors}
    stats = {competitor.label: CompetitorStats() for competitor in competitors}
    for row in episodes:
        labels = row["lineup"]
        scores = row["scores"]
        statuses = row["statuses"]
        row["ratings_before"] = rating_snapshot(labels, ratings)
        update_pairwise_ratings(labels, scores, ratings, beta=beta, tau=0.0)
        classify_outcomes(labels, scores, statuses, stats)
        row["ratings_after"] = rating_snapshot(labels, ratings)
    return leaderboard_rows(ratings, stats)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "seed",
                "lineup",
                "specs",
                "scores",
                "ranks",
                "rewards",
                "statuses",
                "duration_s",
                "steps",
                "ratings_before",
                "ratings_after",
            ],
        )
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["lineup"] = "|".join(payload["lineup"])
            payload["specs"] = "|".join(payload["specs"])
            for key in ("scores", "ranks", "rewards", "statuses", "ratings_before", "ratings_after"):
                payload[key] = json.dumps(payload[key])
            writer.writerow(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local leagues on exact seeds mined from Kaggle replays.")
    parser.add_argument("--agent", action="append", required=True, help="Agent file/spec. Repeat for the pool.")
    parser.add_argument("--players", type=int, choices=(2, 4), required=True)
    parser.add_argument("--seed", action="append", default=[], help="Seed or comma-separated seeds.")
    parser.add_argument("--replays", action="append", type=Path, default=[], help="Replay file/dir to mine seeds from.")
    parser.add_argument("--limit", type=int, help="Maximum mined replay seeds.")
    parser.add_argument("--rotations", type=int, default=0, help="Rotations per matchup. 0 means all player slots.")
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose-env", action="store_true")
    parser.add_argument("--backend", choices=("kaggle", "fast"), default="kaggle")
    parser.add_argument("--fast-act-timeout", type=float, default=0.08)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    competitors = build_competitors(args.agent)
    try:
        plans = planned_lineups(competitors, args.players, args.rotations)
    except ValueError as exc:
        print(f"exact_seed_league: {exc}", file=sys.stderr)
        return 2

    seeds = explicit_seeds(args.seed)
    seeds.extend(seed for seed in seeds_from_replays(args.replays, args.limit) if seed not in set(seeds))
    if not seeds:
        print("exact_seed_league: no seeds provided or mined", file=sys.stderr)
        return 2

    jobs = max(1, min(args.jobs, os.cpu_count() or 1))
    episodes = run_plan(
        plans=plans,
        seeds=seeds,
        episode_steps=args.episode_steps,
        debug=args.debug,
        verbose_env=args.verbose_env,
        jobs=jobs,
        backend=args.backend,
        fast_act_timeout=args.fast_act_timeout,
    )
    leaderboard = summarize(competitors, episodes, beta=args.beta)
    print(f"seeds={len(seeds)} plans_per_seed={len(plans)} episodes={len(episodes)} backend={args.backend}")
    print_leaderboard(leaderboard)

    payload = {
        "seeds": seeds,
        "players": args.players,
        "agents": [asdict(competitor) for competitor in competitors],
        "plans_per_seed": len(plans),
        "backend": args.backend,
        "leaderboard": leaderboard,
        "episodes": episodes,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    if args.csv is not None:
        write_csv(args.csv, episodes)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
