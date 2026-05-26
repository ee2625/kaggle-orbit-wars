#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_replay import load_replay, team_names  # noqa: E402
from scripts.backtest import load_make  # noqa: E402
from scripts.evaluate_replay_suite import actions_for_player, replay_seed, trace_agent  # noqa: E402


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("inspect_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["inspect_candidate"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--agent", type=Path, default=Path("main.py"))
    parser.add_argument("--team", default="orf527")
    parser.add_argument("--turns", default="0,25,35,50,75,100,125,150,200")
    args = parser.parse_args()

    module = load_module(args.agent)
    replay = load_replay(args.replay)
    names = team_names(replay)
    indices = [index for index, name in enumerate(names) if name == args.team]
    if not indices:
        raise RuntimeError(f"team {args.team!r} not found in replay")
    seat = indices[0]
    interesting_turns = {int(item) for item in args.turns.split(",") if item.strip()}

    logged = {"turn": 0}
    lines: list[str] = []

    def wrapped_agent(obs, config=None):
        turn = logged["turn"]
        logged["turn"] += 1
        world = module.build_world(obs)
        policy = module.build_policy_state(world)
        if turn in interesting_turns or (
            world.is_four_player
            and 28 <= world.step < 220
            and world.max_enemy_prod >= world.my_prod + 3
            and world.max_enemy_prod >= world.my_prod * 1.35
        ):
            capped = []
            for planet in world.my_planets:
                exact_keep = world.keep_needed_map.get(planet.id, 0)
                first_enemy = world.first_enemy_map.get(planet.id)
                reserve = policy["reserve"].get(planet.id, 0)
                budget = policy["attack_budget"].get(planet.id, 0)
                capped.append(
                    f"p{planet.id}:ships={int(planet.ships)} prod={int(planet.production)} "
                    f"exact={exact_keep} first={first_enemy} reserve={reserve} budget={budget}"
                )
            lines.append(
                f"t={turn:03d} step={world.step:03d} my_total={world.my_total} "
                f"max_enemy_total={world.max_enemy_strength} my_prod={world.my_prod} "
                f"max_enemy_prod={world.max_enemy_prod} planets={len(world.my_planets)}"
            )
            for item in capped:
                lines.append(f"  {item}")

        action = module.agent(copy.deepcopy(obs), config)
        if action and (turn in interesting_turns or len(action) >= 2):
            lines.append(f"  actions={action}")
        return action

    lineup = []
    for index in range(len(names)):
        if index == seat:
            lineup.append(wrapped_agent)
        else:
            lineup.append(trace_agent(actions_for_player(replay, index), 1))

    make = load_make(False)
    env = make("orbit_wars", configuration={"seed": replay_seed(replay)}, debug=False)
    env.run(lineup)
    final = env.steps[-1]
    for line in lines:
        print(line)
    print("final_rewards", [state.reward for state in final])
    print("final_statuses", [state.status for state in final])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
