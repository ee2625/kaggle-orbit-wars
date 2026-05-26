#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.backtest import final_scores, load_make, quiet_output
from scripts.fast_run_local import run_episode as run_fast_episode


VALID_PLAYER_COUNTS = {2, 4}
SIGMA_FLOOR = 1.0


@dataclass(frozen=True)
class Competitor:
    label: str
    spec: str


@dataclass
class Rating:
    mu: float = 600.0
    sigma: float = 200.0


@dataclass
class CompetitorStats:
    games: int = 0
    wins: int = 0
    ties: int = 0
    losses: int = 0
    errors: int = 0
    score_margins: list[int] | None = None

    def __post_init__(self) -> None:
        if self.score_margins is None:
            self.score_margins = []


@dataclass(frozen=True)
class LeagueEpisode:
    episode: int
    seed: int
    lineup: list[str]
    specs: list[str]
    scores: list[int]
    ranks: list[int]
    rewards: list[float | int | None]
    statuses: list[str]
    ratings_before: dict[str, dict[str, float]]
    ratings_after: dict[str, dict[str, float]]
    duration_s: float
    steps: int


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def safe_cdf(x: float) -> float:
    return min(1.0 - 1e-12, max(1e-12, normal_cdf(x)))


def rate_win(winner: Rating, loser: Rating, beta: float, tau: float) -> tuple[Rating, Rating]:
    winner = apply_drift(winner, tau)
    loser = apply_drift(loser, tau)
    variance_sum = winner.sigma * winner.sigma + loser.sigma * loser.sigma + 2.0 * beta * beta
    c = math.sqrt(variance_sum)
    t = (winner.mu - loser.mu) / c
    v = normal_pdf(t) / safe_cdf(t)
    w = v * (v + t)

    winner_variance = winner.sigma * winner.sigma
    loser_variance = loser.sigma * loser.sigma
    winner_mu = winner.mu + (winner_variance / c) * v
    loser_mu = loser.mu - (loser_variance / c) * v
    winner_sigma = math.sqrt(max(SIGMA_FLOOR * SIGMA_FLOOR, winner_variance * (1.0 - (winner_variance / variance_sum) * w)))
    loser_sigma = math.sqrt(max(SIGMA_FLOOR * SIGMA_FLOOR, loser_variance * (1.0 - (loser_variance / variance_sum) * w)))

    return Rating(winner_mu, winner_sigma), Rating(loser_mu, loser_sigma)


def rate_draw(first: Rating, second: Rating, beta: float, tau: float) -> tuple[Rating, Rating]:
    first = apply_drift(first, tau)
    second = apply_drift(second, tau)
    variance_sum = first.sigma * first.sigma + second.sigma * second.sigma + 2.0 * beta * beta
    c = math.sqrt(variance_sum)
    expected_first = safe_cdf((first.mu - second.mu) / c)
    surprise = 0.5 - expected_first

    first_variance = first.sigma * first.sigma
    second_variance = second.sigma * second.sigma
    first_mu = first.mu + (first_variance / c) * surprise
    second_mu = second.mu - (second_variance / c) * surprise
    information = max(0.02, expected_first * (1.0 - expected_first))
    first_sigma = math.sqrt(max(SIGMA_FLOOR * SIGMA_FLOOR, first_variance * (1.0 - (first_variance / variance_sum) * information)))
    second_sigma = math.sqrt(max(SIGMA_FLOOR * SIGMA_FLOOR, second_variance * (1.0 - (second_variance / variance_sum) * information)))

    return Rating(first_mu, first_sigma), Rating(second_mu, second_sigma)


def apply_drift(rating: Rating, tau: float) -> Rating:
    if tau <= 0.0:
        return Rating(rating.mu, rating.sigma)

    return Rating(rating.mu, math.sqrt(rating.sigma * rating.sigma + tau * tau))


def update_pairwise_ratings(
    labels: list[str],
    scores: list[int],
    ratings: dict[str, Rating],
    beta: float,
    tau: float,
) -> None:
    for left_index, right_index in itertools.combinations(range(len(labels)), 2):
        left = labels[left_index]
        right = labels[right_index]
        left_score = scores[left_index]
        right_score = scores[right_index]

        if left_score > right_score:
            ratings[left], ratings[right] = rate_win(ratings[left], ratings[right], beta, tau)
        elif left_score < right_score:
            ratings[right], ratings[left] = rate_win(ratings[right], ratings[left], beta, tau)
        else:
            ratings[left], ratings[right] = rate_draw(ratings[left], ratings[right], beta, tau)


def rank_scores(scores: list[int]) -> list[int]:
    sorted_scores = sorted(set(scores), reverse=True)
    return [sorted_scores.index(score) + 1 for score in scores]


def classify_outcomes(labels: list[str], scores: list[int], statuses: list[str], stats: dict[str, CompetitorStats]) -> None:
    best_score = max(scores) if scores else 0
    winner_count = sum(1 for score in scores if score == best_score)

    for index, label in enumerate(labels):
        row = stats[label]
        row.games += 1
        if index < len(statuses) and statuses[index] != "DONE":
            row.errors += 1

        opponent_best = max((score for pos, score in enumerate(scores) if pos != index), default=scores[index])
        row.score_margins.append(scores[index] - opponent_best)

        if scores[index] == best_score and winner_count == 1:
            row.wins += 1
        elif scores[index] == best_score:
            row.ties += 1
        else:
            row.losses += 1


def rating_snapshot(labels: Iterable[str], ratings: dict[str, Rating]) -> dict[str, dict[str, float]]:
    return {label: asdict(ratings[label]) for label in labels}


def build_competitors(agent_specs: list[str]) -> list[Competitor]:
    seen: dict[str, int] = {}
    competitors: list[Competitor] = []

    for spec in agent_specs:
        path = Path(spec)
        if path.suffix and path.name in {"main.py", "agent.py"} and path.parent.name:
            base = path.parent.name
        elif path.suffix:
            base = path.stem
        else:
            base = spec
        count = seen.get(base, 0) + 1
        seen[base] = count
        label = base if count == 1 else f"{base}#{count}"
        competitors.append(Competitor(label=label, spec=spec))

    return competitors


def build_matchups(competitors: list[Competitor], players: int) -> list[tuple[Competitor, ...]]:
    if players not in VALID_PLAYER_COUNTS:
        raise ValueError("Orbit Wars local matches must have 2 or 4 players.")
    if len(competitors) < players:
        raise ValueError(f"need at least {players} agents for a {players}-player league.")

    return list(itertools.combinations(competitors, players))


def choose_matchup(
    competitors: list[Competitor],
    matchups: list[tuple[Competitor, ...]],
    ratings: dict[str, Rating],
    episode: int,
    players: int,
    schedule: str,
    rng: random.Random,
) -> tuple[Competitor, ...]:
    if schedule == "random":
        return tuple(rng.sample(competitors, players))

    if schedule == "ladder":
        anchor = competitors[episode % len(competitors)]
        others = [competitor for competitor in competitors if competitor.label != anchor.label]
        closest = sorted(
            others,
            key=lambda competitor: (
                abs(ratings[competitor.label].mu - ratings[anchor.label].mu),
                competitor.label,
            ),
        )[: players - 1]
        return tuple([anchor, *closest])

    return matchups[episode % len(matchups)]


def rotate_lineup(matchup: tuple[Competitor, ...], episode: int) -> list[Competitor]:
    offset = episode % len(matchup)
    return list(matchup[offset:] + matchup[:offset])


def adjusted_scores(scores: list[int], statuses: list[str]) -> list[int]:
    adjusted = list(scores)
    for index, status in enumerate(statuses):
        if status != "DONE" and index < len(adjusted):
            adjusted[index] = min(adjusted, default=0) - 1
    return adjusted


def run_league_episode(
    make: Callable[..., Any] | None,
    lineup: list[Competitor],
    episode: int,
    seed: int,
    episode_steps: int | None,
    debug: bool,
    verbose_env: bool,
    backend: str = "kaggle",
    fast_act_timeout: float = 0.08,
) -> tuple[list[int], list[float | int | None], list[str], float, int]:
    if backend == "fast":
        _, rewards, scores, statuses, duration_s, steps = run_fast_episode(
            [competitor.spec for competitor in lineup],
            seed=seed,
            episode_steps=episode_steps or 500,
            act_timeout=fast_act_timeout,
        )
        return adjusted_scores(scores, statuses), rewards, statuses, duration_s, steps

    if make is None:
        raise ValueError("kaggle backend requires a make callable")

    configuration: dict[str, Any] = {"seed": seed}
    if episode_steps is not None:
        configuration["episodeSteps"] = episode_steps

    started = time.perf_counter()
    with quiet_output(not verbose_env):
        env = make("orbit_wars", configuration=configuration, debug=debug)
        env.run([competitor.spec for competitor in lineup])
    duration_s = time.perf_counter() - started

    final = env.steps[-1] if env.steps else []
    rewards = [state.reward for state in final]
    statuses = [state.status for state in final]
    scores = final_scores(final[0].observation, len(lineup)) if final else [0 for _ in lineup]

    return adjusted_scores(scores, statuses), rewards, statuses, duration_s, len(env.steps)


def run_league_episode_worker(
    payload: tuple[list[Competitor], int, int, int | None, bool, bool, str, float],
) -> tuple[int, list[int], list[float | int | None], list[str], float, int]:
    lineup, episode, seed, episode_steps, debug, verbose_env, backend, fast_act_timeout = payload
    make = None if backend == "fast" else load_make(verbose_env)
    scores, rewards, statuses, duration_s, steps = run_league_episode(
        make=make,
        lineup=lineup,
        episode=episode,
        seed=seed,
        episode_steps=episode_steps,
        debug=debug,
        verbose_env=verbose_env,
        backend=backend,
        fast_act_timeout=fast_act_timeout,
    )
    return episode, scores, rewards, statuses, duration_s, steps


def preplanned_lineups(
    competitors: list[Competitor],
    games: int,
    players: int,
    seed_start: int,
    schedule: str,
) -> list[list[Competitor]]:
    if schedule == "ladder":
        raise ValueError("ladder schedule depends on live ratings and cannot be safely preplanned.")

    rng = random.Random(seed_start)
    matchups = build_matchups(competitors, players)
    ratings = {competitor.label: Rating() for competitor in competitors}
    lineups: list[list[Competitor]] = []
    for episode in range(games):
        matchup = choose_matchup(competitors, matchups, ratings, episode, players, schedule, rng)
        lineups.append(rotate_lineup(matchup, episode))
    return lineups


def run_preplanned_league(
    competitors: list[Competitor],
    lineups: list[list[Competitor]],
    games: int,
    seed_start: int,
    mu0: float,
    sigma0: float,
    beta: float,
    tau: float,
    episode_steps: int | None,
    debug: bool,
    verbose_env: bool,
    quiet: bool,
    jobs: int,
    backend: str,
    fast_act_timeout: float,
) -> tuple[dict[str, Rating], dict[str, CompetitorStats], list[LeagueEpisode]]:
    ratings = {competitor.label: Rating(mu0, sigma0) for competitor in competitors}
    stats = {competitor.label: CompetitorStats() for competitor in competitors}
    episodes_by_number: dict[int, tuple[list[int], list[float | int | None], list[str], float, int]] = {}

    payloads = [
        (
            lineups[episode],
            episode + 1,
            seed_start + episode,
            episode_steps,
            debug,
            verbose_env,
            backend,
            fast_act_timeout,
        )
        for episode in range(games)
    ]

    if jobs <= 1:
        make = None if backend == "fast" else load_make(verbose_env)
        for lineup, episode, seed, episode_steps, debug, verbose_env, backend, fast_act_timeout in payloads:
            episodes_by_number[episode] = run_league_episode(
                make=make,
                lineup=lineup,
                episode=episode,
                seed=seed,
                episode_steps=episode_steps,
                debug=debug,
                verbose_env=verbose_env,
                backend=backend,
                fast_act_timeout=fast_act_timeout,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_league_episode_worker, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                episode, scores, rewards, statuses, duration_s, steps = future.result()
                episodes_by_number[episode] = (scores, rewards, statuses, duration_s, steps)

    episodes: list[LeagueEpisode] = []
    for episode in range(1, games + 1):
        lineup = lineups[episode - 1]
        labels = [competitor.label for competitor in lineup]
        seed = seed_start + episode - 1
        before = rating_snapshot(labels, ratings)
        scores, rewards, statuses, duration_s, steps = episodes_by_number[episode]
        update_pairwise_ratings(labels, scores, ratings, beta, tau)
        classify_outcomes(labels, scores, statuses, stats)
        after = rating_snapshot(labels, ratings)
        row = LeagueEpisode(
            episode=episode,
            seed=seed,
            lineup=labels,
            specs=[competitor.spec for competitor in lineup],
            scores=scores,
            ranks=rank_scores(scores),
            rewards=rewards,
            statuses=statuses,
            ratings_before=before,
            ratings_after=after,
            duration_s=duration_s,
            steps=steps,
        )
        episodes.append(row)

        if not quiet:
            print(
                f"episode={row.episode} seed={seed} lineup={','.join(labels)} "
                f"scores={scores} ranks={row.ranks} statuses={statuses} "
                f"duration={duration_s:.2f}s"
            )

    return ratings, stats, episodes


def run_league(
    make: Callable[..., Any] | None,
    competitors: list[Competitor],
    games: int,
    players: int,
    seed_start: int,
    schedule: str,
    mu0: float,
    sigma0: float,
    beta: float,
    tau: float,
    episode_steps: int | None,
    debug: bool,
    verbose_env: bool,
    quiet: bool,
    jobs: int = 1,
    backend: str = "kaggle",
    fast_act_timeout: float = 0.08,
) -> tuple[dict[str, Rating], dict[str, CompetitorStats], list[LeagueEpisode]]:
    if jobs > 1 and schedule != "ladder":
        return run_preplanned_league(
            competitors=competitors,
            lineups=preplanned_lineups(competitors, games, players, seed_start, schedule),
            games=games,
            seed_start=seed_start,
            mu0=mu0,
            sigma0=sigma0,
            beta=beta,
            tau=tau,
            episode_steps=episode_steps,
            debug=debug,
            verbose_env=verbose_env,
            quiet=quiet,
            jobs=jobs,
            backend=backend,
            fast_act_timeout=fast_act_timeout,
        )

    rng = random.Random(seed_start)
    matchups = build_matchups(competitors, players)
    ratings = {competitor.label: Rating(mu0, sigma0) for competitor in competitors}
    stats = {competitor.label: CompetitorStats() for competitor in competitors}
    episodes: list[LeagueEpisode] = []

    for episode in range(games):
        matchup = choose_matchup(competitors, matchups, ratings, episode, players, schedule, rng)
        lineup = rotate_lineup(matchup, episode)
        labels = [competitor.label for competitor in lineup]
        seed = seed_start + episode
        before = rating_snapshot(labels, ratings)
        scores, rewards, statuses, duration_s, steps = run_league_episode(
            make=make,
            lineup=lineup,
            episode=episode + 1,
            seed=seed,
            episode_steps=episode_steps,
            debug=debug,
            verbose_env=verbose_env,
            backend=backend,
            fast_act_timeout=fast_act_timeout,
        )

        update_pairwise_ratings(labels, scores, ratings, beta, tau)
        classify_outcomes(labels, scores, statuses, stats)
        after = rating_snapshot(labels, ratings)
        row = LeagueEpisode(
            episode=episode + 1,
            seed=seed,
            lineup=labels,
            specs=[competitor.spec for competitor in lineup],
            scores=scores,
            ranks=rank_scores(scores),
            rewards=rewards,
            statuses=statuses,
            ratings_before=before,
            ratings_after=after,
            duration_s=duration_s,
            steps=steps,
        )
        episodes.append(row)

        if not quiet:
            print(
                f"episode={row.episode} seed={seed} lineup={','.join(labels)} "
                f"scores={scores} ranks={row.ranks} statuses={statuses} "
                f"duration={duration_s:.2f}s"
            )

    return ratings, stats, episodes


def leaderboard_rows(ratings: dict[str, Rating], stats: dict[str, CompetitorStats]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, rating in ratings.items():
        row = stats[label]
        margins = row.score_margins or []
        rows.append(
            {
                "agent": label,
                "mu": rating.mu,
                "sigma": rating.sigma,
                "conservative": rating.mu - 3.0 * rating.sigma,
                "games": row.games,
                "wins": row.wins,
                "ties": row.ties,
                "losses": row.losses,
                "errors": row.errors,
                "avg_score_margin": statistics.fmean(margins) if margins else None,
            }
        )

    return sorted(rows, key=lambda row: (row["mu"], -row["sigma"], row["agent"]), reverse=True)


def print_leaderboard(rows: list[dict[str, Any]]) -> None:
    print()
    print("Local Ladder Ratings")
    print("rank  agent                         mu      sigma   games  W-T-L    avg_margin")
    for rank, row in enumerate(rows, start=1):
        margin = row["avg_score_margin"]
        margin_text = "n/a" if margin is None else f"{margin:.1f}"
        record = f"{row['wins']}-{row['ties']}-{row['losses']}"
        print(
            f"{rank:>4}  {row['agent'][:28]:<28} "
            f"{row['mu']:>7.1f}  {row['sigma']:>7.1f}  {row['games']:>5}  "
            f"{record:<7}  {margin_text:>10}"
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, episodes: list[LeagueEpisode]) -> None:
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
                "ratings_before",
                "ratings_after",
                "duration_s",
                "steps",
            ],
        )
        writer.writeheader()
        for episode in episodes:
            row = asdict(episode)
            for field in ("lineup", "specs"):
                row[field] = "|".join(row[field])
            for field in ("scores", "ranks", "rewards", "statuses", "ratings_before", "ratings_after"):
                row[field] = json.dumps(row[field])
            writer.writerow(row)


def resolve_output_paths(out_dir: Path | None, json_path: Path | None, csv_path: Path | None) -> tuple[Path | None, Path | None]:
    if out_dir is None:
        return json_path, csv_path

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        json_path or out_dir / f"league_{stamp}.json",
        csv_path or out_dir / f"league_{stamp}.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local Orbit Wars ladder with Gaussian skill ratings.")
    parser.add_argument("--agent", action="append", default=[], help="Agent file or built-in agent. Repeat for the pool.")
    parser.add_argument("--games", type=int, default=20, help="Total league games to run.")
    parser.add_argument("--players", type=int, default=2, choices=sorted(VALID_PLAYER_COUNTS), help="Agents per game.")
    parser.add_argument("--seed", type=int, default=42, help="First seed to use.")
    parser.add_argument(
        "--schedule",
        choices=["round-robin", "ladder", "random"],
        default="round-robin",
        help="Matchmaking policy. 'ladder' picks currently similar ratings.",
    )
    parser.add_argument("--mu0", type=float, default=600.0, help="Initial Gaussian mean.")
    parser.add_argument("--sigma0", type=float, default=200.0, help="Initial Gaussian uncertainty.")
    parser.add_argument("--beta", type=float, default=100.0, help="Performance variance scale.")
    parser.add_argument("--tau", type=float, default=0.0, help="Uncertainty drift added before each pair update.")
    parser.add_argument("--episode-steps", type=int, help="Override episodeSteps for faster smoke runs.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel worker processes for round-robin/random schedules. Ladder schedule remains serial.",
    )
    parser.add_argument("--debug", action="store_true", help="Run the Kaggle environment in debug mode.")
    parser.add_argument("--verbose-env", action="store_true", help="Do not suppress Kaggle environment stdout/stderr.")
    parser.add_argument(
        "--backend",
        choices=["kaggle", "fast"],
        default="kaggle",
        help="Simulation backend. 'fast' bypasses the Kaggle wrapper with orbit_fast.",
    )
    parser.add_argument(
        "--fast-act-timeout",
        type=float,
        default=0.08,
        help="Synthetic actTimeout passed to agents when --backend fast is used.",
    )
    parser.add_argument("--quiet", action="store_true", help="Print only the final leaderboard.")
    parser.add_argument("--out-dir", type=Path, help="Write timestamped JSON and CSV results into this directory.")
    parser.add_argument("--json", type=Path, help="Write JSON results to this path.")
    parser.add_argument("--csv", type=Path, help="Write CSV episode rows to this path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    agent_specs = args.agent or ["main.py", "random"]
    competitors = build_competitors(agent_specs)
    jobs = max(1, min(args.jobs, os.cpu_count() or 1))
    if args.jobs > 1 and args.schedule == "ladder":
        print("league_backtest: --jobs is ignored for ladder schedule because matchmaking depends on live ratings.", file=sys.stderr)

    try:
        build_matchups(competitors, args.players)
    except ValueError as exc:
        print(f"league_backtest: {exc}", file=sys.stderr)
        return 2

    try:
        make = None if args.backend == "fast" else load_make(args.verbose_env)
    except ImportError:
        print("Missing dependency. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    ratings, stats, episodes = run_league(
        make=make,
        competitors=competitors,
        games=args.games,
        players=args.players,
        seed_start=args.seed,
        schedule=args.schedule,
        mu0=args.mu0,
        sigma0=args.sigma0,
        beta=args.beta,
        tau=args.tau,
        episode_steps=args.episode_steps,
        debug=args.debug,
        verbose_env=args.verbose_env,
        quiet=args.quiet,
        jobs=jobs,
        backend=args.backend,
        fast_act_timeout=args.fast_act_timeout,
    )
    rows = leaderboard_rows(ratings, stats)
    print_leaderboard(rows)

    json_path, csv_path = resolve_output_paths(args.out_dir, args.json, args.csv)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": "orbit_wars",
        "agents": [asdict(competitor) for competitor in competitors],
        "games": args.games,
        "players": args.players,
        "seed_start": args.seed,
        "schedule": args.schedule,
        "jobs": jobs,
        "backend": args.backend,
        "fast_act_timeout": args.fast_act_timeout,
        "rating_model": {
            "type": "pairwise_trueskill_style",
            "mu0": args.mu0,
            "sigma0": args.sigma0,
            "beta": args.beta,
            "tau": args.tau,
            "score_margin_used": False,
        },
        "leaderboard": rows,
        "episodes": [asdict(episode) for episode in episodes],
    }

    if json_path is not None:
        write_json(json_path, payload)
        print(f"wrote {json_path}")
    if csv_path is not None:
        write_csv(csv_path, episodes)
        print(f"wrote {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
