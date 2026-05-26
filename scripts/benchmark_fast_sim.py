#!/usr/bin/env python3
"""Benchmark the local Orbit Wars simulator against the raw Kaggle interpreter."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments.envs.orbit_wars.orbit_wars import interpreter

from orbit_fast.sim import GameState


def _advance_step_counter(state):
    next_step = getattr(state[0].observation, "step", 0) + 1
    for agent_state in state:
        agent_state.observation.step = next_step


def _make_official_state(players: int, seed: int, episode_steps: int):
    state = [
        SimpleNamespace(
            observation=SimpleNamespace(step=0),
            action=[],
            status="ACTIVE",
            reward=0,
        )
    ]
    for player in range(1, players):
        state.append(
            SimpleNamespace(
                observation=SimpleNamespace(player=player),
                action=[],
                status="ACTIVE",
                reward=0,
            )
        )
    env = SimpleNamespace(
        configuration=SimpleNamespace(
            seed=seed,
            shipSpeed=6,
            episodeSteps=episode_steps,
            cometSpeed=4,
        ),
        done=False,
        info={"seed": seed},
    )
    state = interpreter(state, env)
    return state, env


def bench_fast(episodes: int, steps: int, players: int) -> int:
    transitions = 0
    actions = [[] for _ in range(players)]
    for seed in range(episodes):
        state = GameState.initialize(
            num_agents=players,
            seed=10_000 + seed,
            episode_steps=steps + 2,
        )
        for _ in range(steps):
            if state.done:
                break
            state.step(actions)
            transitions += 1
    return transitions


def bench_official(episodes: int, steps: int, players: int) -> int:
    transitions = 0
    for seed in range(episodes):
        state, env = _make_official_state(players, 10_000 + seed, steps + 2)
        actions = [[] for _ in range(players)]
        for _ in range(steps):
            if state[0].status == "DONE":
                break
            for i, agent_state in enumerate(state):
                agent_state.action = actions[i]
            state = interpreter(state, env)
            _advance_step_counter(state)
            transitions += 1
    return transitions


def run_one(label, fn, episodes, steps, players):
    start = time.perf_counter()
    transitions = fn(episodes, steps, players)
    elapsed = time.perf_counter() - start
    rate = transitions / elapsed if elapsed else 0.0
    print(f"{label}: {transitions} transitions in {elapsed:.3f}s = {rate:,.0f} steps/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--players", type=int, default=4, choices=(2, 4))
    args = parser.parse_args()

    run_one("fast-sim", bench_fast, args.episodes, args.steps, args.players)
    run_one("official-interpreter", bench_official, args.episodes, args.steps, args.players)


if __name__ == "__main__":
    main()
