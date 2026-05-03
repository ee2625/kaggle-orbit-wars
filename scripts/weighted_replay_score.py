#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.min
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def submission_order(manifest: dict[str, Any]) -> list[int]:
    submissions = manifest.get("submissions", {}) or {}
    return [
        int(submission_id)
        for submission_id, _submission in sorted(
            submissions.items(),
            key=lambda item: parse_time(item[1].get("date")),
        )
    ]


def episode_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(key): value for key, value in (manifest.get("episodes", {}) or {}).items()}


def episode_weight(
    episode: dict[str, Any] | None,
    submission_rank: dict[int, int],
    latest_multiplier: float,
    previous_multiplier: float,
    validation_multiplier: float,
    recency_decay: float,
    newest_index: int,
    episode_index: int,
) -> float:
    if episode is None:
        return 1.0

    weight = validation_multiplier if episode.get("type") == "EPISODE_TYPE_VALIDATION" else 1.0
    submission_id = int(episode.get("submission_id", -1))
    rank = submission_rank.get(submission_id, -1)
    latest_rank = max(submission_rank.values(), default=-1)
    if rank == latest_rank:
        weight *= latest_multiplier
    elif rank == latest_rank - 1:
        weight *= previous_multiplier

    if recency_decay < 1.0 and newest_index >= 0:
        age = max(0, newest_index - episode_index)
        weight *= recency_decay**age

    return weight


def result_score(result: str) -> float:
    if result == "win":
        return 1.0
    if result == "tie":
        return 0.5
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Score replay-suite results with recency/submission weighting.")
    parser.add_argument("suite", type=Path, help="Replay suite JSON produced by evaluate_replay_suite.py.")
    parser.add_argument("--manifest", type=Path, default=Path("backtests/kaggle_feedback/manifest.json"))
    parser.add_argument("--latest-multiplier", type=float, default=5.0)
    parser.add_argument("--previous-multiplier", type=float, default=2.0)
    parser.add_argument("--validation-multiplier", type=float, default=0.2)
    parser.add_argument("--recency-decay", type=float, default=1.0, help="Optional per-episode age multiplier.")
    args = parser.parse_args()

    suite = load_json(args.suite)
    manifest = load_json(args.manifest)
    episodes = episode_lookup(manifest)
    submission_ids = submission_order(manifest)
    submission_rank = {submission_id: index for index, submission_id in enumerate(submission_ids)}
    episode_times = {
        episode_id: parse_time(episode.get("create_time"))
        for episode_id, episode in episodes.items()
    }
    ordered_episode_ids = sorted(episode_times, key=lambda episode_id: episode_times[episode_id])
    episode_index = {episode_id: index for index, episode_id in enumerate(ordered_episode_ids)}
    newest_index = len(ordered_episode_ids) - 1

    weighted_total = 0.0
    weighted_score = 0.0
    by_submission: dict[str, dict[str, float]] = {}
    losses: list[tuple[float, dict[str, Any], dict[str, Any] | None]] = []

    for row in suite:
        episode_id = str(row.get("episode_id"))
        episode = episodes.get(episode_id)
        index = episode_index.get(episode_id, newest_index)
        weight = episode_weight(
            episode,
            submission_rank,
            args.latest_multiplier,
            args.previous_multiplier,
            args.validation_multiplier,
            args.recency_decay,
            newest_index,
            index,
        )
        score = result_score(row.get("result", "unknown"))
        weighted_total += weight
        weighted_score += weight * score

        submission_name = "unknown"
        if episode is not None:
            submission_name = str(episode.get("submission_description") or episode.get("submission_id") or "unknown")
        bucket = by_submission.setdefault(submission_name, {"weight": 0.0, "score": 0.0, "wins": 0.0, "losses": 0.0})
        bucket["weight"] += weight
        bucket["score"] += weight * score
        if row.get("result") == "win":
            bucket["wins"] += 1
        elif row.get("result") == "loss":
            bucket["losses"] += 1
            losses.append((weight, row, episode))

    weighted_rate = weighted_score / weighted_total if weighted_total else 0.0
    print(f"weighted_score={weighted_score:.2f}/{weighted_total:.2f} weighted_win_rate={weighted_rate:.1%}")
    print()
    print("By submission")
    for name, bucket in sorted(by_submission.items(), key=lambda item: item[1]["weight"], reverse=True):
        rate = bucket["score"] / bucket["weight"] if bucket["weight"] else 0.0
        print(f"{name}: weighted_win_rate={rate:.1%} raw={int(bucket['wins'])}W-{int(bucket['losses'])}L weight={bucket['weight']:.2f}")

    if losses:
        print()
        print("Weighted losses")
        for weight, row, episode in sorted(losses, key=lambda item: item[0], reverse=True):
            description = "unknown" if episode is None else episode.get("submission_description", "unknown")
            print(
                f"episode={row.get('episode_id')} weight={weight:.2f} submission={description} "
                f"scores={row.get('simulated_scores')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
