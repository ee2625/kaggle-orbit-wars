#!/usr/bin/env python3
import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Orbit Wars matches.")
    parser.add_argument("--agent", default="main.py", help="Agent file to evaluate.")
    parser.add_argument("--opponent", default="random", help="Opponent agent name or file.")
    parser.add_argument("--opponents", nargs="*", help="Optional list of opponents for 2p or 4p games.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of games to run.")
    parser.add_argument("--seed", type=int, default=42, help="First seed to use.")
    args = parser.parse_args()

    try:
        from kaggle_environments import make
    except ImportError:
        print('Missing dependency. Run: pip install -r requirements.txt')
        return 1

    agent_path = str(Path(args.agent))
    opponents = args.opponents if args.opponents is not None else [args.opponent]
    agents = [agent_path, *opponents]
    wins = 0
    ties = 0
    losses = 0

    for offset in range(args.episodes):
        seed = args.seed + offset
        env = make("orbit_wars", configuration={"seed": seed}, debug=True)
        env.run(agents)
        final = env.steps[-1]
        rewards = [state.reward for state in final]
        statuses = [state.status for state in final]

        if rewards and all(reward is not None for reward in rewards):
            best_reward = max(rewards)
            best_count = sum(1 for reward in rewards if reward == best_reward)
            if rewards[0] == best_reward and best_count == 1:
                wins += 1
                result = "win"
            elif rewards[0] == best_reward:
                ties += 1
                result = "tie"
            else:
                losses += 1
                result = "loss"
        else:
            result = "unknown"

        print(f"seed={seed} result={result} rewards={rewards} statuses={statuses}")

    print(f"summary: {wins}W-{ties}T-{losses}L over {args.episodes} episode(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
