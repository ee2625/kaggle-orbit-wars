#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAYS = [
    ROOT / "backtests/v613_peak_feedback/replays/submission_52335247/episode-75877726-replay.json",
    ROOT / "backtests/v613_peak_feedback/replays/submission_52335247/episode-75877525-replay.json",
    ROOT / "backtests/v613_peak_feedback/replays/submission_52335247/episode-75879990-replay.json",
    ROOT / "backtests/v613_peak_feedback/replays/submission_52335247/episode-75879489-replay.json",
    ROOT / "backtests/v613_peak_feedback/replays/submission_52335247/episode-75877107-replay.json",
    ROOT / "backtests/v613_peak_feedback/replays/submission_52335247/episode-75876988-replay.json",
    ROOT / "backtests/v613_peak_feedback/replays/submission_52335247/episode-75879458-replay.json",
    ROOT / "backtests/v613_peak_feedback/replays/submission_52335247/episode-75878926-replay.json",
]


def suggest_params(trial: Any) -> dict[str, Any]:
    return {
        "HOSTILE_TARGET_VALUE_MULT": trial.suggest_float("HOSTILE_TARGET_VALUE_MULT", 1.75, 2.45),
        "OPENING_HOSTILE_TARGET_VALUE_MULT": trial.suggest_float("OPENING_HOSTILE_TARGET_VALUE_MULT", 1.25, 1.95),
        "SAFE_NEUTRAL_VALUE_MULT": trial.suggest_float("SAFE_NEUTRAL_VALUE_MULT", 1.00, 1.40),
        "CONTESTED_NEUTRAL_VALUE_MULT": trial.suggest_float("CONTESTED_NEUTRAL_VALUE_MULT", 0.45, 0.95),
        "PROACTIVE_DEFENSE_RATIO": trial.suggest_float("PROACTIVE_DEFENSE_RATIO", 0.18, 0.40),
        "MULTI_ENEMY_PROACTIVE_RATIO": trial.suggest_float("MULTI_ENEMY_PROACTIVE_RATIO", 0.25, 0.46),
        "PRESSURE_PACKET_MIN_LEFT": trial.suggest_int("PRESSURE_PACKET_MIN_LEFT", 42, 72),
        "COORDINATED_WAVE_MIN_PROD_GAP": trial.suggest_int("COORDINATED_WAVE_MIN_PROD_GAP", 3, 8),
        "TWO_PLAYER_STRATEGY.opening_high_prod_min_packet": trial.suggest_int(
            "TWO_PLAYER_STRATEGY.opening_high_prod_min_packet", 18, 32
        ),
        "TWO_PLAYER_STRATEGY.opening_high_prod_future_turns": trial.suggest_int(
            "TWO_PLAYER_STRATEGY.opening_high_prod_future_turns", 10, 24
        ),
        "TWO_PLAYER_STRATEGY.pressure_packet_base_ratio": trial.suggest_float(
            "TWO_PLAYER_STRATEGY.pressure_packet_base_ratio", 0.45, 0.75
        ),
        "TWO_PLAYER_STRATEGY.rear_send_ratio": trial.suggest_float(
            "TWO_PLAYER_STRATEGY.rear_send_ratio", 0.50, 0.76
        ),
        "FOUR_PLAYER_STRATEGY.partial_source_min": trial.suggest_int(
            "FOUR_PLAYER_STRATEGY.partial_source_min", 3, 6
        ),
        "FOUR_PLAYER_STRATEGY.opening_high_prod_min_packet": trial.suggest_int(
            "FOUR_PLAYER_STRATEGY.opening_high_prod_min_packet", 10, 24
        ),
        "FOUR_PLAYER_STRATEGY.rear_send_ratio": trial.suggest_float(
            "FOUR_PLAYER_STRATEGY.rear_send_ratio", 0.56, 0.80
        ),
    }


def _literal(value: Any) -> str:
    if isinstance(value, float):
        return repr(round(value, 6))
    return repr(value)


def _replace_assignment(text: str, name: str, value: Any) -> str:
    pattern = re.compile(rf"^({re.escape(name)}\s*=\s*).*$", re.MULTILINE)
    replacement = rf"\g<1>{_literal(value)}"
    text, _count = pattern.subn(replacement, text, count=1)
    return text


def _replace_dict_entry(text: str, dict_name: str, key: str, value: Any) -> str:
    block_pattern = re.compile(
        rf"({re.escape(dict_name)}\s*=\s*\{{)(.*?)(\n\}})",
        re.DOTALL,
    )
    match = block_pattern.search(text)
    if match is None:
        return text

    body = match.group(2)
    entry_pattern = re.compile(rf"^(\s*['\"]{re.escape(key)}['\"]\s*:\s*).*(,?)$", re.MULTILINE)
    new_body, count = entry_pattern.subn(rf"\g<1>{_literal(value)},", body, count=1)
    if count == 0:
        return text
    return text[: match.start(2)] + new_body + text[match.end(2) :]


def write_candidate(path: Path, base_agent: Path, params: dict[str, Any]) -> None:
    text = base_agent.read_text(encoding="utf-8")
    for name, value in sorted(params.items()):
        if "." in name:
            dict_name, key = name.split(".", 1)
            text = _replace_dict_entry(text, dict_name, key, value)
        else:
            text = _replace_assignment(text, name, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def replay_score(candidate: Path, replays: list[Path], out_dir: Path) -> tuple[float, int, int]:
    existing = [path for path in replays if path.exists()]
    if not existing:
        return 0.0, 0, 0

    suite_path = out_dir / f"{candidate.stem}_replay_suite.json"
    command = [
        sys.executable,
        "scripts/evaluate_replay_suite.py",
        *[str(path) for path in existing],
        "--agent",
        str(candidate),
        "--team",
        "orf527",
        "--json",
        str(suite_path),
    ]
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode not in (0, 1):
        completed.check_returncode()

    rows = json.loads(suite_path.read_text(encoding="utf-8"))
    if not rows:
        return 0.0, 0, 0
    points = 0.0
    losses = 0
    for row in rows:
        result = row.get("result")
        if result == "win":
            points += 1.0
        elif result == "tie":
            points += 0.5
        elif result == "loss":
            losses += 1
    return points / len(rows), len(rows), losses


def league_mu(
    candidate: Path,
    players: int,
    games: int,
    seed: int,
    jobs: int,
    out_dir: Path,
    baseline_agent: Path | None = None,
) -> float:
    if games <= 0:
        return 600.0

    json_path = out_dir / f"{candidate.stem}_{players}p_league.json"
    agents = [str(candidate)]
    if baseline_agent is not None and baseline_agent.exists() and baseline_agent.resolve() != candidate.resolve():
        agents.append(str(baseline_agent))
    if players == 2:
        agents.extend(
            [
                "submissions/v5_1_submitted.py",
                "submissions/v6_20_2p_4p_profile_split_candidate.py",
                "submissions/v6_13_starved_economy_unlock_320_candidate.py",
                "submissions/v5_9_partial_volume_candidate.py",
            ]
        )
    else:
        agents.extend(
            [
                "submissions/v5_1_submitted.py",
                "submissions/v6_20_2p_4p_profile_split_candidate.py",
                "submissions/v6_13_starved_economy_unlock_320_candidate.py",
                "submissions/v5_9_partial_volume_candidate.py",
            ]
        )

    agents = [agent for agent in agents if agent == "random" or (ROOT / agent).exists() or Path(agent).exists()]
    while len(agents) < players:
        agents.append("random")

    command = [
        sys.executable,
        "scripts/league_backtest.py",
        "--players",
        str(players),
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--schedule",
        "random",
        "--jobs",
        str(jobs),
        "--quiet",
        "--json",
        str(json_path),
    ]
    for agent in agents:
        command.extend(["--agent", agent])
    run_command(command)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    candidate_label = candidate.stem
    for row in payload["leaderboard"]:
        if row["agent"] == candidate_label:
            return float(row["mu"])
    return 600.0


def objective_factory(args: argparse.Namespace):
    out_dir = args.out_dir
    replays = args.replay or [path for path in DEFAULT_REPLAYS if path.exists()]

    def objective(trial: Any) -> float:
        params = suggest_params(trial)
        candidate = out_dir / "candidates" / f"trial_{trial.number:04d}.py"
        write_candidate(candidate, args.base_agent, params)

        replay_rate, replay_count, replay_losses = replay_score(candidate, replays, out_dir)
        baseline_agent = args.baseline_agent or args.base_agent
        mu_2p = league_mu(
            candidate,
            2,
            args.games_2p,
            args.seed + trial.number * 1000,
            args.jobs,
            out_dir,
            baseline_agent,
        )
        mu_4p = league_mu(
            candidate,
            4,
            args.games_4p,
            args.seed + 500 + trial.number * 1000,
            args.jobs,
            out_dir,
            baseline_agent,
        )

        mixed_score = replay_rate * args.replay_weight + (mu_2p + mu_4p) * 0.5 - replay_losses * args.loss_penalty
        trial.set_user_attr("candidate", str(candidate))
        trial.set_user_attr("replay_rate", replay_rate)
        trial.set_user_attr("replay_count", replay_count)
        trial.set_user_attr("replay_losses", replay_losses)
        trial.set_user_attr("mu_2p", mu_2p)
        trial.set_user_attr("mu_4p", mu_4p)
        return mixed_score

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuner for Orbit Wars constants.")
    parser.add_argument("--base-agent", type=Path, default=ROOT / "main.py")
    parser.add_argument("--baseline-agent", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "backtests/optuna_constants")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--games-2p", type=int, default=8)
    parser.add_argument("--games-4p", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=92000)
    parser.add_argument("--replay", action="append", type=Path, default=[])
    parser.add_argument("--replay-weight", type=float, default=350.0)
    parser.add_argument("--loss-penalty", type=float, default=80.0)
    parser.add_argument("--study-name", default="orbit-wars-constants")
    parser.add_argument("--storage", help="Optional Optuna storage URL, e.g. sqlite:///backtests/optuna.db")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import optuna
    except ImportError:
        print("Missing optional dependency. Install with: pip install optuna", file=sys.stderr)
        return 1

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )
    study.optimize(objective_factory(args), n_trials=args.trials)

    best_path = args.out_dir / "best_params.json"
    best_path.write_text(
        json.dumps(
            {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_attrs": study.best_trial.user_attrs,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"best_value={study.best_value:.3f}")
    print(f"best_params={study.best_params}")
    print(f"wrote {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
