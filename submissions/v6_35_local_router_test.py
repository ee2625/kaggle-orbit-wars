"""Local-only v6.35 router for faster validation.

This imports sibling submission files, so it is not a Kaggle submission artifact.
The standalone candidate is v6_35_mode_router_v627_2p_v634_4p_candidate.py.
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


_V2P_AGENT = _load_agent("v6_27_gated_v517_cheap_highprod_2p_candidate.py", "_orbit_v635_local_2p")
_V4P_AGENT = _load_agent("v6_34_4p_packet_aggregation_candidate.py", "_orbit_v635_local_4p")


def _read_field(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _detect_game_players(obs):
    initial = _read_field(obs, "initial_planets", []) or []
    owners = set()
    for row in initial:
        try:
            owner = int(row[1])
        except (IndexError, TypeError, ValueError):
            continue
        if owner != -1:
            owners.add(owner)
    if len(owners) >= 4:
        return 4

    planets = _read_field(obs, "planets", []) or []
    fleets = _read_field(obs, "fleets", []) or []
    owners = set()
    for row in planets:
        try:
            owner = int(row[1])
        except (IndexError, TypeError, ValueError):
            continue
        if owner != -1:
            owners.add(owner)
    for row in fleets:
        try:
            owner = int(row[1])
        except (IndexError, TypeError, ValueError):
            continue
        if owner != -1:
            owners.add(owner)
    return max(2, len(owners))


def agent(obs, config=None):
    if _detect_game_players(obs) >= 4:
        return _V4P_AGENT(obs, config)
    return _V2P_AGENT(obs, config)
