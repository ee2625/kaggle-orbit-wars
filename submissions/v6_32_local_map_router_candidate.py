"""Local-only router for testing map-profile policy selection.

This file imports sibling submission files, so it is not a Kaggle submission
artifact. If it validates, generate a standalone embedded version.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load_agent(filename: str, module_name: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.agent


_V59 = _load_agent("v5_9_partial_volume_candidate.py", "_orbit_v59")
_V629 = _load_agent("v6_29_4p_leader_conversion_candidate.py", "_orbit_v629")
_V630 = _load_agent("v6_30_v59_leader_conversion_candidate.py", "_orbit_v630")
_V627 = _load_agent("v6_27_gated_v517_cheap_highprod_2p_candidate.py", "_orbit_v627")

_ROUTES = {}


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _dist(a, b):
    import math

    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def _state_key(obs):
    player = int(_read(obs, "player", 0) or 0)
    initial = _read(obs, "initial_planets", []) or []
    signature = []
    for row in initial:
        try:
            signature.append((
                int(row[0]),
                int(row[1]),
                round(float(row[2]), 3),
                round(float(row[3]), 3),
                round(float(row[4]), 3),
                int(row[6]),
            ))
        except (TypeError, ValueError, IndexError):
            continue
    return player, tuple(signature)


def _players(obs):
    initial = _read(obs, "initial_planets", []) or []
    owners = {int(row[1]) for row in initial if len(row) > 1 and int(row[1]) != -1}
    if owners:
        return max(2, len(owners))
    current = _read(obs, "planets", []) or []
    owners = {int(row[1]) for row in current if len(row) > 1 and int(row[1]) != -1}
    return max(2, len(owners))


def _route_for_game(obs):
    if _players(obs) < 4:
        return "v627"

    player = int(_read(obs, "player", 0) or 0)
    initial = _read(obs, "initial_planets", []) or _read(obs, "planets", []) or []
    my_home = [row for row in initial if int(row[1]) == player]
    enemy_home = [row for row in initial if int(row[1]) not in (-1, player)]
    if not my_home or not enemy_home:
        current = _read(obs, "planets", []) or []
        my_home = [row for row in current if int(row[1]) == player]
        enemy_home = [row for row in current if int(row[1]) not in (-1, player)]
        initial = current
    neutrals = [row for row in initial if int(row[1]) == -1]
    high = [row for row in neutrals if int(row[6]) >= 4]

    if not high:
        return "v630"
    if not my_home or not enemy_home:
        return "v59"

    enemy_dist = min(_dist(home, enemy) for home in my_home for enemy in enemy_home)
    cheap_high_exists = any(
        int(neutral[6]) >= 4
        and int(neutral[5]) <= 8
        and _dist(home, neutral) <= 24
        for home in my_home
        for neutral in high
    )
    if cheap_high_exists and enemy_dist >= 60:
        return "v629"
    return "v59"


def agent(obs, config=None):
    key = _state_key(obs)
    route = _ROUTES.get(key)
    if route is None:
        route = _route_for_game(obs)
        _ROUTES[key] = route
    if route == "v629":
        return _V629(obs, config)
    if route == "v630":
        return _V630(obs, config)
    if route == "v627":
        return _V627(obs, config)
    return _V59(obs, config)
