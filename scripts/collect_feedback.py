#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_replay import build_player_stats, load_replay, team_names  # noqa: E402


DEFAULT_OUT_DIR = Path("backtests/kaggle_feedback")


def load_kaggle_api() -> Any:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Missing dependency. Run: pip install -r requirements.txt") from exc

    api = KaggleApi()
    api.authenticate()
    return api


def submission_dict(submission: Any) -> dict[str, Any]:
    if hasattr(submission, "to_dict"):
        raw = submission.to_dict()
    else:
        raw = json.loads(str(submission))

    return {
        "id": int(raw.get("ref") or raw.get("id")),
        "date": raw.get("date"),
        "description": raw.get("description"),
        "status": str(raw.get("status")),
        "file_name": raw.get("fileName") or raw.get("file_name"),
        "public_score": raw.get("publicScore") or raw.get("public_score"),
        "private_score": raw.get("privateScore") or raw.get("private_score"),
    }


def episode_dict(episode: Any) -> dict[str, Any]:
    raw = episode.to_dict() if hasattr(episode, "to_dict") else json.loads(str(episode))
    agents = raw.get("agents") or []
    return {
        "id": int(raw["id"]),
        "create_time": raw.get("createTime") or raw.get("create_time"),
        "end_time": raw.get("endTime") or raw.get("end_time"),
        "state": str(raw.get("state")),
        "type": str(raw.get("type")),
        "agents": agents,
    }


def selected_submissions(api: Any, competition: str, explicit_ids: list[int], latest: int) -> list[dict[str, Any]]:
    if explicit_ids:
        known = {
            submission["id"]: submission
            for submission in (submission_dict(item) for item in api.competition_submissions(competition))
        }
        return [
            known.get(submission_id, {"id": submission_id, "description": None, "public_score": None})
            for submission_id in explicit_ids
        ]

    submissions = [submission_dict(item) for item in api.competition_submissions(competition)]
    return submissions[:latest]


def matching_agent_indices(episode: dict[str, Any], submission_id: int, team: str | None) -> list[int]:
    matches: list[int] = []
    for agent in episode["agents"]:
        agent_submission = int(agent.get("submissionId") or agent.get("submission_id") or -1)
        agent_team = agent.get("teamName") or agent.get("team_name")
        if agent_submission == submission_id or (team is not None and agent_team == team):
            matches.append(int(agent.get("index", len(matches))))
    return sorted(set(matches))


def download_replay(api: Any, episode_id: int, path: Path) -> bool:
    if path.exists():
        return False

    from kagglesdk.competitions.types.competition_api_service import ApiGetEpisodeReplayRequest

    path.parent.mkdir(parents=True, exist_ok=True)
    with api.build_kaggle_client() as kaggle:
        request = ApiGetEpisodeReplayRequest()
        request.episode_id = episode_id
        response = kaggle.competitions.competition_api_client.get_episode_replay(request)
        response.raise_for_status()
        path.write_bytes(response.content)
    return True


def download_logs(api: Any, episode_id: int, agent_index: int, path: Path) -> bool:
    if path.exists():
        return False

    from kagglesdk.competitions.types.competition_api_service import ApiGetEpisodeAgentLogsRequest

    path.parent.mkdir(parents=True, exist_ok=True)
    with api.build_kaggle_client() as kaggle:
        request = ApiGetEpisodeAgentLogsRequest()
        request.episode_id = episode_id
        request.agent_index = agent_index
        response = kaggle.competitions.competition_api_client.get_episode_agent_logs(request)
        response.raise_for_status()
        path.write_bytes(response.content)
    return True


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"episodes": {}, "submissions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def summarize_replay(
    replay_path: Path,
    episode: dict[str, Any],
    submission: dict[str, Any],
    our_indices: list[int],
) -> dict[str, Any]:
    replay = load_replay(replay_path)
    stats = build_player_stats(replay, early_turns=80)
    names = team_names(replay)
    winners = [index for index, row in enumerate(stats) if row.reward == max(player.reward for player in stats)]
    our_rows = [stats[index] for index in our_indices if index < len(stats)]
    opponents = [name for index, name in enumerate(names) if index not in our_indices]

    return {
        "episode_id": episode["id"],
        "submission_id": submission["id"],
        "submission_description": submission.get("description"),
        "submission_public_score": submission.get("public_score"),
        "type": episode["type"],
        "state": episode["state"],
        "create_time": episode.get("create_time"),
        "players": len(names),
        "teams": names,
        "opponents": opponents,
        "our_indices": our_indices,
        "won": any(index in winners for index in our_indices),
        "winner_indices": winners,
        "winner_teams": [names[index] for index in winners],
        "our_rewards": [row.reward for row in our_rows],
        "our_final_scores": [row.final_score for row in our_rows],
        "our_first_capture_turns": [row.first_capture_turn for row in our_rows],
        "our_first_launch_turns": [row.launch.first_turn for row in our_rows],
        "our_max_planets": [row.max_planets for row in our_rows],
        "our_max_production": [row.max_production for row in our_rows],
        "our_launch_miss_rates": [
            row.launch.projected_misses / row.launch.count if row.launch.count else 0.0
            for row in our_rows
        ],
        "our_fleet_deaths": [asdict(row.deaths) for row in our_rows],
        "replay_path": str(replay_path),
    }


def write_markdown_summary(path: Path, episodes: list[dict[str, Any]], submissions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Orbit Wars Feedback",
        "",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Submissions",
        "",
        "| id | description | status | score |",
        "| --- | --- | --- | --- |",
    ]
    for submission in submissions:
        lines.append(
            f"| {submission['id']} | {submission.get('description') or ''} | "
            f"{submission.get('status') or ''} | {submission.get('public_score') or ''} |"
        )

    public = [episode for episode in episodes if "PUBLIC" in episode["type"]]
    wins = sum(1 for episode in public if episode["won"])
    lines.extend(
        [
            "",
            "## Public Episode Summary",
            "",
            f"Public episodes tracked: {len(public)}",
            f"Wins: {wins}",
            f"Losses: {len(public) - wins}",
            "",
            "| episode | submission | result | players | first capture | max prod | miss rate | replay |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for episode in sorted(public, key=lambda item: item["episode_id"], reverse=True):
        result = "win" if episode["won"] else "loss"
        first_capture = ",".join("n/a" if value is None else str(value) for value in episode["our_first_capture_turns"])
        max_production = ",".join(str(value) for value in episode["our_max_production"])
        miss_rate = ",".join(f"{value:.1%}" for value in episode["our_launch_miss_rates"])
        replay_path = episode["replay_path"]
        lines.append(
            f"| {episode['episode_id']} | {episode['submission_description'] or episode['submission_id']} | "
            f"{result} | {episode['players']} | {first_capture} | {max_production} | {miss_rate} | `{replay_path}` |"
        )

    lines.extend(
        [
            "",
            "## Latest Notes",
            "",
            "- Use `python scripts/analyze_replay.py <replay>` for detailed per-episode diagnostics.",
            "- Kaggle score updates are live rating estimates; score margin is diagnostic only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_feedback(
    competition: str,
    out_dir: Path,
    submission_ids: list[int],
    latest_submissions: int,
    team: str | None,
    include_validation: bool,
    download_agent_logs: bool,
    max_episodes_per_submission: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    api = load_kaggle_api()
    submissions = selected_submissions(api, competition, submission_ids, latest_submissions)
    manifest_path = out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    episode_summaries: list[dict[str, Any]] = []
    new_summaries: list[dict[str, Any]] = []

    for submission in submissions:
        submission_id = int(submission["id"])
        manifest["submissions"][str(submission_id)] = submission
        episodes = [episode_dict(item) for item in api.competition_list_episodes(submission_id)]
        if max_episodes_per_submission is not None:
            episodes = episodes[:max_episodes_per_submission]

        for episode in episodes:
            if episode["state"] != "COMPLETED":
                continue
            if not include_validation and "VALIDATION" in episode["type"]:
                continue

            our_indices = matching_agent_indices(episode, submission_id, team)
            if not our_indices:
                continue

            replay_path = out_dir / "replays" / f"submission_{submission_id}" / f"episode-{episode['id']}-replay.json"
            downloaded = download_replay(api, episode["id"], replay_path)
            if download_agent_logs:
                for index in our_indices:
                    log_path = out_dir / "logs" / f"submission_{submission_id}" / f"episode-{episode['id']}-agent-{index}-logs.json"
                    download_logs(api, episode["id"], index, log_path)

            summary = summarize_replay(replay_path, episode, submission, our_indices)
            manifest["episodes"][str(episode["id"])] = summary
            episode_summaries.append(summary)
            if downloaded:
                new_summaries.append(summary)

    all_episodes = list(manifest["episodes"].values())
    all_submissions = list(manifest["submissions"].values())
    write_json(manifest_path, manifest)
    write_markdown_summary(out_dir / "summary.md", all_episodes, all_submissions)
    write_json(out_dir / "latest_run.json", {"new_episodes": new_summaries, "tracked_episodes": all_episodes})
    return new_summaries, all_episodes, submissions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Kaggle Orbit Wars feedback and summarize replay diagnostics.")
    parser.add_argument("--competition", default="orbit-wars", help="Kaggle competition slug.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for manifest, replays, logs, and summary.")
    parser.add_argument("--submission-id", action="append", type=int, default=[], help="Submission ID to collect. Repeatable.")
    parser.add_argument("--latest-submissions", type=int, default=2, help="Use this many latest submissions when no ID is given.")
    parser.add_argument("--team", help="Optional team name fallback when matching replay players.")
    parser.add_argument("--include-validation", action="store_true", help="Include validation episodes.")
    parser.add_argument("--download-logs", action="store_true", help="Download our agent logs for matched episodes.")
    parser.add_argument("--max-episodes-per-submission", type=int, help="Cap episodes fetched per submission.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        new_summaries, all_episodes, submissions = collect_feedback(
            competition=args.competition,
            out_dir=args.out_dir,
            submission_ids=args.submission_id,
            latest_submissions=args.latest_submissions,
            team=args.team,
            include_validation=args.include_validation,
            download_agent_logs=args.download_logs,
            max_episodes_per_submission=args.max_episodes_per_submission,
        )
    except RuntimeError as exc:
        print(f"collect_feedback: {exc}", file=sys.stderr)
        return 1

    print(f"submissions checked: {len(submissions)}")
    print(f"episodes tracked: {len(all_episodes)}")
    print(f"new replays downloaded: {len(new_summaries)}")
    print(f"summary: {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
