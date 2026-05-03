#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_feedback import download_replay, load_kaggle_api  # noqa: E402


EPISODE_RE = re.compile(r"(?<!\d)(\d{6,})(?!\d)")


def ids_from_text(text: str) -> list[int]:
    seen: set[int] = set()
    ids: list[int] = []
    for match in EPISODE_RE.finditer(text):
        episode_id = int(match.group(1))
        if episode_id in seen:
            continue
        seen.add(episode_id)
        ids.append(episode_id)
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public Kaggle Orbit Wars replays by episode ID or URL.")
    parser.add_argument("items", nargs="*", help="Episode IDs, replay URLs, or text containing episode IDs.")
    parser.add_argument("--file", type=Path, help="Text file containing episode IDs or replay URLs.")
    parser.add_argument("--out-dir", type=Path, default=Path("backtests/public_replays"), help="Replay output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text_parts = list(args.items)
    if args.file is not None:
        text_parts.append(args.file.read_text(encoding="utf-8"))

    episode_ids = ids_from_text("\n".join(text_parts))
    if not episode_ids:
        print("No episode IDs found. Paste replay URLs or numeric episode IDs.", file=sys.stderr)
        return 1

    api = load_kaggle_api()
    downloaded = 0
    for episode_id in episode_ids:
        replay_path = args.out_dir / f"episode-{episode_id}-replay.json"
        if download_replay(api, episode_id, replay_path):
            downloaded += 1
            status = "downloaded"
        else:
            status = "cached"
        print(f"{status}: {episode_id} -> {replay_path}")

    print(f"episodes: {len(episode_ids)}  new downloads: {downloaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
