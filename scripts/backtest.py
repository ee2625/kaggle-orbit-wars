#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


VALID_PLAYER_COUNTS = {2, 4}


@dataclass(frozen=True)
class EpisodeResult:
    agent: str
    episode: int
    seed: int
    agent_index: int
    opponents: list[str]
    lineup: list[str]
    result: str
    rewards: list[float | int | None]
    statuses: list[str]
    margin: float | None
    duration_s: float
    steps: int


@contextmanager
def quiet_output(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def build_lineup(agent: str, opponents: list[str], seat: int) -> tuple[list[str], int]:
    player_count = len(opponents) + 1
    if player_count not in VALID_PLAYER_COUNTS:
        raise ValueError("Orbit Wars local matches must have 2 or 4 total agents.")
    if seat < 0 or seat >= player_count:
        raise ValueError(f"seat must be between 0 and {player_count - 1}.")

    lineup = list(opponents)
    lineup.insert(seat, agent)
    return lineup, seat


def classify_result(rewards: list[float | int | None], statuses: list[str], agent_index: int) -> str:
    if agent_index >= len(statuses) or statuses[agent_index] != "DONE":
        return "error"
    if not rewards or any(reward is None for reward in rewards):
        return "unknown"

    agent_reward = rewards[agent_index]
    best_reward = max(reward for reward in rewards if reward is not None)
    best_count = sum(1 for reward in rewards if reward == best_reward)

    if agent_reward == best_reward and best_count == 1:
        return "win"
    if agent_reward == best_reward:
        return "tie"
    return "loss"


def reward_margin(rewards: list[float | int | None], agent_index: int) -> float | None:
    if not rewards or any(reward is None for reward in rewards):
        return None

    opponent_rewards = [float(reward) for index, reward in enumerate(rewards) if index != agent_index]
    if not opponent_rewards:
        return None

    return float(rewards[agent_index]) - max(opponent_rewards)


def summarize_results(results: list[EpisodeResult]) -> dict[str, Any]:
    episodes = len(results)
    wins = sum(1 for result in results if result.result == "win")
    ties = sum(1 for result in results if result.result == "tie")
    losses = sum(1 for result in results if result.result == "loss")
    errors = sum(1 for result in results if result.result == "error")
    unknown = sum(1 for result in results if result.result == "unknown")
    margins = [result.margin for result in results if result.margin is not None]
    durations = [result.duration_s for result in results]

    return {
        "episodes": episodes,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "errors": errors,
        "unknown": unknown,
        "win_rate": wins / episodes if episodes else 0.0,
        "non_loss_rate": (wins + ties) / episodes if episodes else 0.0,
        "avg_margin": statistics.fmean(margins) if margins else None,
        "avg_duration_s": statistics.fmean(durations) if durations else None,
    }


def load_make(verbose_env: bool) -> Callable[..., Any]:
    with quiet_output(not verbose_env):
        from kaggle_environments import make

    return make


def run_episode(
    make: Callable[..., Any],
    agent: str,
    opponents: list[str],
    episode: int,
    seed: int,
    seat: int,
    episode_steps: int | None,
    debug: bool,
    verbose_env: bool,
) -> EpisodeResult:
    lineup, agent_index = build_lineup(agent, opponents, seat)
    configuration: dict[str, Any] = {"seed": seed}
    if episode_steps is not None:
        configuration["episodeSteps"] = episode_steps

    started = time.perf_counter()
    with quiet_output(not verbose_env):
        env = make("orbit_wars", configuration=configuration, debug=debug)
        env.run(lineup)
    duration_s = time.perf_counter() - started

    final = env.steps[-1] if env.steps else []
    rewards = [state.reward for state in final]
    statuses = [state.status for state in final]
    result = classify_result(rewards, statuses, agent_index)

    return EpisodeResult(
        agent=agent,
        episode=episode,
        seed=seed,
        agent_index=agent_index,
        opponents=opponents,
        lineup=lineup,
        result=result,
        rewards=rewards,
        statuses=statuses,
        margin=reward_margin(rewards, agent_index),
        duration_s=duration_s,
        steps=len(env.steps),
    )


def run_backtest(
    make: Callable[..., Any],
    agents: list[str],
    opponents: list[str],
    episodes: int,
    seed_start: int,
    seat: int,
    rotate_seat: bool,
    episode_steps: int | None,
    debug: bool,
    verbose_env: bool,
    quiet: bool,
) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    player_count = len(opponents) + 1

    for agent in agents:
        for episode in range(episodes):
            seed = seed_start + episode
            agent_seat = episode % player_count if rotate_seat else seat
            result = run_episode(
                make=make,
                agent=agent,
                opponents=opponents,
                episode=episode + 1,
                seed=seed,
                seat=agent_seat,
                episode_steps=episode_steps,
                debug=debug,
                verbose_env=verbose_env,
            )
            results.append(result)

            if not quiet:
                print(
                    f"agent={Path(agent).name} episode={result.episode} seed={seed} "
                    f"seat={agent_seat} result={result.result} rewards={result.rewards} "
                    f"statuses={result.statuses} duration={result.duration_s:.2f}s"
                )

    return results


def grouped_summaries(results: list[EpisodeResult]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for agent in sorted({result.agent for result in results}):
        summaries[agent] = summarize_results([result for result in results if result.agent == agent])
    return summaries


def print_summaries(summaries: dict[str, dict[str, Any]]) -> None:
    print()
    print("Summary")
    for agent, summary in summaries.items():
        avg_margin = summary["avg_margin"]
        avg_duration = summary["avg_duration_s"]
        margin_text = "n/a" if avg_margin is None else f"{avg_margin:.3f}"
        duration_text = "n/a" if avg_duration is None else f"{avg_duration:.2f}s"
        print(
            f"{Path(agent).name}: {summary['wins']}W-{summary['ties']}T-{summary['losses']}L "
            f"errors={summary['errors']} unknown={summary['unknown']} "
            f"win_rate={summary['win_rate']:.1%} avg_margin={margin_text} "
            f"avg_duration={duration_text}"
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, results: list[EpisodeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "agent",
                "episode",
                "seed",
                "agent_index",
                "opponents",
                "lineup",
                "result",
                "rewards",
                "statuses",
                "margin",
                "duration_s",
                "steps",
            ],
        )
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["opponents"] = "|".join(result.opponents)
            row["lineup"] = "|".join(result.lineup)
            row["rewards"] = json.dumps(result.rewards)
            row["statuses"] = json.dumps(result.statuses)
            writer.writerow(row)


def resolve_output_paths(out_dir: Path | None, json_path: Path | None, csv_path: Path | None) -> tuple[Path | None, Path | None]:
    if out_dir is None:
        return json_path, csv_path

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        json_path or out_dir / f"backtest_{stamp}.json",
        csv_path or out_dir / f"backtest_{stamp}.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeatable Orbit Wars backtests.")
    parser.add_argument("--agent", action="append", default=[], help="Agent file to evaluate. Repeat for comparisons.")
    parser.add_argument("--opponent", default="random", help="Single opponent for compatibility with run_local.py.")
    parser.add_argument("--opponents", nargs="*", help="Opponent list. Use three entries for 4-player games.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of games per agent.")
    parser.add_argument("--seed", type=int, default=42, help="First seed to use.")
    parser.add_argument("--seat", type=int, default=0, help="Player slot for the tested agent.")
    parser.add_argument("--rotate-seat", action="store_true", help="Rotate the tested agent through all player slots.")
    parser.add_argument("--episode-steps", type=int, help="Override episodeSteps for faster smoke runs.")
    parser.add_argument("--debug", action="store_true", help="Run the Kaggle environment in debug mode.")
    parser.add_argument("--verbose-env", action="store_true", help="Do not suppress Kaggle environment stdout/stderr.")
    parser.add_argument("--quiet", action="store_true", help="Print only the final summary.")
    parser.add_argument("--out-dir", type=Path, help="Write timestamped JSON and CSV results into this directory.")
    parser.add_argument("--json", type=Path, help="Write JSON results to this path.")
    parser.add_argument("--csv", type=Path, help="Write CSV episode rows to this path.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero if the tested agent errors.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    agents = args.agent or ["main.py"]
    opponents = args.opponents if args.opponents is not None else [args.opponent]

    try:
        for agent in agents:
            build_lineup(agent, opponents, args.seat)
    except ValueError as exc:
        print(f"backtest: {exc}", file=sys.stderr)
        return 2

    try:
        make = load_make(args.verbose_env)
    except ImportError:
        print("Missing dependency. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    results = run_backtest(
        make=make,
        agents=agents,
        opponents=opponents,
        episodes=args.episodes,
        seed_start=args.seed,
        seat=args.seat,
        rotate_seat=args.rotate_seat,
        episode_steps=args.episode_steps,
        debug=args.debug,
        verbose_env=args.verbose_env,
        quiet=args.quiet,
    )
    summaries = grouped_summaries(results)

    if not args.quiet:
        print_summaries(summaries)
    else:
        print_summaries(summaries)

    json_path, csv_path = resolve_output_paths(args.out_dir, args.json, args.csv)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": "orbit_wars",
        "agents": agents,
        "opponents": opponents,
        "episodes_per_agent": args.episodes,
        "seed_start": args.seed,
        "seat": args.seat,
        "rotate_seat": args.rotate_seat,
        "episode_steps": args.episode_steps,
        "summaries": summaries,
        "episodes": [asdict(result) for result in results],
    }

    if json_path is not None:
        write_json(json_path, payload)
        print(f"wrote {json_path}")
    if csv_path is not None:
        write_csv(csv_path, results)
        print(f"wrote {csv_path}")

    if args.fail_on_error and any(result.result == "error" for result in results):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
