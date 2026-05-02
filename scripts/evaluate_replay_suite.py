#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_replay import load_replay, scores_from_obs, team_names, world_observation  # noqa: E402
from scripts.backtest import load_make  # noqa: E402


@dataclass(frozen=True)
class ReplaySuiteResult:
    replay: str
    episode_id: int | str
    seed: int
    player_index: int
    original_teams: list[str]
    original_scores: list[int]
    original_rewards: list[float | int | None]
    simulated_scores: list[int]
    simulated_rewards: list[float | int | None]
    simulated_statuses: list[str]
    result: str
    duration_s: float
    steps: int


def load_agent(path: Path) -> Callable[..., Any]:
    module_name = f"candidate_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load agent from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise RuntimeError(f"{path} does not define callable agent(obs)")
    return agent


def trace_agent(actions: list[Any], offset: int) -> Callable[..., Any]:
    counter = {"turn": 0}

    def agent(_obs: Any, _config: Any = None) -> Any:
        index = counter["turn"] + offset
        counter["turn"] += 1
        if 0 <= index < len(actions):
            return copy.deepcopy(actions[index])
        return []

    return agent


def actions_for_player(replay: dict[str, Any], player_index: int) -> list[Any]:
    return [states[player_index].get("action") or [] for states in replay["steps"]]


def replay_seed(replay: dict[str, Any]) -> int:
    info = replay.get("info", {}) or {}
    if info.get("seed") is not None:
        return int(info["seed"])

    config = replay.get("configuration", {}) or {}
    if config.get("seed") is not None:
        return int(config["seed"])

    raise RuntimeError("replay has no usable seed")


def episode_id(replay: dict[str, Any], path: Path) -> int | str:
    info = replay.get("info", {}) or {}
    return info.get("EpisodeId") or replay.get("id") or path.stem


def original_scores(replay: dict[str, Any], player_count: int) -> list[int]:
    return scores_from_obs(world_observation(replay, len(replay["steps"]) - 1), player_count)


def find_player_indices(replay: dict[str, Any], team: str | None, explicit_index: int | None) -> list[int]:
    if explicit_index is not None:
        return [explicit_index]

    names = team_names(replay)
    if team is None:
        return []

    return [index for index, name in enumerate(names) if name == team]


def classify_result(rewards: list[float | int | None], player_index: int) -> str:
    if not rewards or player_index >= len(rewards) or rewards[player_index] is None:
        return "unknown"

    best = max(reward for reward in rewards if reward is not None)
    best_count = sum(1 for reward in rewards if reward == best)
    if rewards[player_index] == best and best_count == 1:
        return "win"
    if rewards[player_index] == best:
        return "tie"
    return "loss"


def evaluate_replay(
    make: Callable[..., Any],
    replay_path: Path,
    replay: dict[str, Any],
    candidate_agent: Callable[..., Any],
    team: str | None,
    player_index: int | None,
    trace_offset: int,
    save_simulated_dir: Path | None,
    verbose_env: bool,
) -> list[ReplaySuiteResult]:
    names = team_names(replay)
    player_count = len(names)
    indices = find_player_indices(replay, team, player_index)
    if not indices:
        return []

    seed = replay_seed(replay)
    original = original_scores(replay, player_count)
    original_rewards = replay.get("rewards") or [None for _ in range(player_count)]
    results: list[ReplaySuiteResult] = []

    for index in indices:
        lineup: list[Any] = []
        for seat in range(player_count):
            if seat == index:
                lineup.append(candidate_agent)
            else:
                lineup.append(trace_agent(actions_for_player(replay, seat), trace_offset))

        started = time.perf_counter()
        with maybe_quiet(not verbose_env):
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run(lineup)
        duration_s = time.perf_counter() - started

        final = env.steps[-1] if env.steps else []
        rewards = [state.reward for state in final]
        statuses = [state.status for state in final]
        scores = scores_from_obs(final[0].observation, player_count) if final else [0 for _ in range(player_count)]
        if save_simulated_dir is not None:
            save_simulated_dir.mkdir(parents=True, exist_ok=True)
            sim_path = save_simulated_dir / f"episode-{episode_id(replay, replay_path)}-seat-{index}-simulated.json"
            sim_path.write_text(json.dumps(env.toJSON(), indent=2), encoding="utf-8")
        results.append(
            ReplaySuiteResult(
                replay=str(replay_path),
                episode_id=episode_id(replay, replay_path),
                seed=seed,
                player_index=index,
                original_teams=names,
                original_scores=original,
                original_rewards=original_rewards,
                simulated_scores=scores,
                simulated_rewards=rewards,
                simulated_statuses=statuses,
                result=classify_result(rewards, index),
                duration_s=duration_s,
                steps=len(env.steps),
            )
        )

    return results


@contextlib.contextmanager
def maybe_quiet(enabled: bool):
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def expand_replay_inputs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*replay.json")))
        else:
            expanded.append(path)
    return expanded


def print_summary(results: list[ReplaySuiteResult]) -> None:
    wins = sum(1 for result in results if result.result == "win")
    ties = sum(1 for result in results if result.result == "tie")
    losses = sum(1 for result in results if result.result == "loss")
    print()
    print(f"Replay Suite: {wins}W-{ties}T-{losses}L over {len(results)} evaluated seats")
    for result in results:
        print(
            f"episode={result.episode_id} seat={result.player_index} result={result.result} "
            f"original_scores={result.original_scores} simulated_scores={result.simulated_scores} "
            f"rewards={result.simulated_rewards} steps={result.steps}"
        )


def write_json(path: Path, results: list[ReplaySuiteResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an agent against recorded replay opponent actions.")
    parser.add_argument("replays", nargs="+", type=Path, help="Replay JSON files or directories.")
    parser.add_argument("--agent", type=Path, default=Path("main.py"), help="Candidate agent file.")
    parser.add_argument("--team", default="orf527", help="Team name to replace with the candidate agent.")
    parser.add_argument("--player-index", type=int, help="Explicit player index to replace.")
    parser.add_argument("--trace-offset", type=int, default=1, help="Replay action offset. 1 reproduces Kaggle replays.")
    parser.add_argument("--json", type=Path, help="Write detailed results JSON.")
    parser.add_argument("--save-simulated-dir", type=Path, help="Write simulated replay JSON files into this directory.")
    parser.add_argument("--verbose-env", action="store_true", help="Do not suppress Kaggle environment output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_agent = load_agent(args.agent)
    try:
        make = load_make(args.verbose_env)
    except ImportError:
        print("Missing dependency. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    results: list[ReplaySuiteResult] = []
    for replay_path in expand_replay_inputs(args.replays):
        replay = load_replay(replay_path)
        results.extend(
            evaluate_replay(
                make=make,
                replay_path=replay_path,
                replay=replay,
                candidate_agent=candidate_agent,
                team=args.team,
                player_index=args.player_index,
                trace_offset=args.trace_offset,
                save_simulated_dir=args.save_simulated_dir,
                verbose_env=args.verbose_env,
            )
        )

    print_summary(results)
    if args.json is not None:
        write_json(args.json, results)
        print(f"wrote {args.json}")

    return 1 if any(result.result == "loss" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
