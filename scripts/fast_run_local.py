#!/usr/bin/env python3
"""Run Orbit Wars agents through the local fast simulator."""

from __future__ import annotations

import argparse
import inspect
import math
import random
import runpy
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbit_fast.sim import GameState


AgentFn = Callable[..., list[list[Any]]]


def _agent_accepts_config(agent: AgentFn) -> bool:
    signature = inspect.signature(agent)
    return any(
        param.kind == inspect.Parameter.VAR_POSITIONAL
        for param in signature.parameters.values()
    ) or len(signature.parameters) >= 2


def _tag_agent_signature(agent: AgentFn) -> AgentFn:
    try:
        setattr(agent, "_fast_accepts_config", _agent_accepts_config(agent))
    except Exception:
        pass
    return agent


def _random_agent(seed: int) -> AgentFn:
    rng = random.Random(seed)

    def agent(obs, config=None):
        moves = []
        player = obs.get("player", 0)
        for planet in obs.get("planets", []):
            if planet[1] == player and planet[5] > 0:
                ships = int(planet[5]) // 2
                if ships >= 20:
                    moves.append([planet[0], rng.uniform(0, 2 * math.pi), ships])
        return moves

    return agent


def _do_nothing_agent(obs, config=None):
    return []


def load_agent(spec: str, seed: int = 0) -> AgentFn:
    if spec == "random":
        return _tag_agent_signature(_random_agent(seed))
    if spec in {"none", "noop", "do_nothing"}:
        return _tag_agent_signature(_do_nothing_agent)

    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"agent not found: {spec}")
    namespace = runpy.run_path(str(path), run_name=f"_fast_agent_{path.stem}_{seed}")
    agent = namespace.get("agent")
    if not callable(agent):
        raise ValueError(f"{spec} does not define callable agent(obs)")
    return _tag_agent_signature(agent)


def _call_agent(agent: AgentFn, obs: dict[str, Any], config: dict[str, Any]):
    accepts_config = getattr(agent, "_fast_accepts_config", None)
    if accepts_config is None:
        accepts_config = _agent_accepts_config(agent)
    if accepts_config:
        return agent(obs, config)
    return agent(obs)


def final_result(rewards: list[int], agent_index: int) -> str:
    if not rewards:
        return "unknown"
    best = max(rewards)
    best_count = sum(1 for reward in rewards if reward == best)
    if rewards[agent_index] == best and best_count == 1:
        return "win"
    if rewards[agent_index] == best:
        return "tie"
    return "loss"


def run_episode(
    specs: list[str],
    seed: int,
    episode_steps: int,
    tracked_index: int = 0,
    copy_observations: bool = False,
    act_timeout: float = 0.08,
) -> tuple[str, list[int], list[int], list[str], float, int]:
    agents = [load_agent(spec, seed * 100 + index) for index, spec in enumerate(specs)]
    state = GameState.initialize(
        num_agents=len(specs),
        seed=seed,
        episode_steps=episode_steps,
    )
    config = {
        "actTimeout": act_timeout,
        "episodeSteps": episode_steps,
        "shipSpeed": 6.0,
        "sunRadius": 10.0,
        "boardSize": 100.0,
        "cometSpeed": 4.0,
    }

    started = time.perf_counter()
    steps = 0
    while not state.done and steps < episode_steps:
        actions = []
        for player, agent in enumerate(agents):
            try:
                obs = state.observe(player, copy_rows=copy_observations)
                actions.append(_call_agent(agent, obs, config))
            except Exception:
                state.status[player] = "ERROR"
                actions.append([])
        state.step(actions)
        steps += 1
    duration = time.perf_counter() - started
    rewards = state.rewards
    scores = state.scores()
    return final_result(rewards, tracked_index), rewards, scores, state.status, duration, steps


def build_lineup(agent: str, opponents: list[str], seat: int) -> tuple[list[str], int]:
    player_count = len(opponents) + 1
    if player_count not in (2, 4):
        raise ValueError("Orbit Wars matches must have 2 or 4 total agents.")
    lineup = list(opponents)
    lineup.insert(seat, agent)
    return lineup, seat


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Orbit Wars matches through orbit_fast.")
    parser.add_argument("--agent", default="main.py")
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--opponents", nargs="*")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument("--rotate-seat", action="store_true")
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--act-timeout", type=float, default=0.08)
    parser.add_argument(
        "--copy-observations",
        action="store_true",
        help="Deep-copy observations before agent calls. Safer, but slower.",
    )
    args = parser.parse_args()

    opponents = args.opponents if args.opponents is not None else [args.opponent]
    player_count = len(opponents) + 1
    wins = ties = losses = 0

    for episode in range(args.episodes):
        seed = args.seed + episode
        seat = episode % player_count if args.rotate_seat else args.seat
        lineup, tracked_index = build_lineup(args.agent, opponents, seat)
        result, rewards, scores, statuses, duration, steps = run_episode(
            lineup,
            seed=seed,
            episode_steps=args.episode_steps,
            tracked_index=tracked_index,
            copy_observations=args.copy_observations,
            act_timeout=args.act_timeout,
        )
        wins += result == "win"
        ties += result == "tie"
        losses += result == "loss"
        print(
            f"seed={seed} seat={seat} result={result} rewards={rewards} "
            f"scores={scores} statuses={statuses} steps={steps} duration={duration:.3f}s"
        )

    print(f"summary: {wins}W-{ties}T-{losses}L over {args.episodes} episode(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
