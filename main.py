from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple


BOARD_SIZE = 100.0
CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0


@dataclass(frozen=True)
class Planet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


def _agent_impl(obs: Any) -> List[List[float]]:
    """Kaggle entrypoint.

    Returns moves in the form [from_planet_id, angle_in_radians, num_ships].
    This baseline aims for robust legal play over cleverness.
    """
    planets = [Planet(*p) for p in get_field(obs, "planets", [])]
    player = int(get_field(obs, "player", 0))

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not my_planets or not targets:
        return []

    angular_velocity = float(get_field(obs, "angular_velocity", 0.0))
    comet_ids = set(get_field(obs, "comet_planet_ids", []) or [])
    moves: List[List[float]] = []

    # Strong planets should decide first, before smaller worlds spend themselves.
    for source in sorted(my_planets, key=lambda p: (p.ships, p.production), reverse=True):
        reserve = reserve_for(source)
        available = int(source.ships - reserve)
        if available <= 1:
            continue

        choice = choose_target(source, targets, planets, obs, angular_velocity, comet_ids, available)
        if choice is None:
            continue

        target, angle, ships = choice
        if ships <= 0:
            continue

        moves.append([source.id, angle, ships])

    return moves


def choose_target(
    source: Planet,
    targets: Sequence[Planet],
    all_planets: Sequence[Planet],
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
    available: int,
) -> Tuple[Planet, float, int] | None:
    best: Tuple[float, Planet, float, int] | None = None

    for target in targets:
        base_needed = ships_needed(target)
        if base_needed > max(available, int(available * 1.35)):
            continue

        ships = min(available, max(base_needed, int(base_needed * 1.12)))
        if ships <= 0:
            continue

        tx, ty = predicted_target_position(target, source, ships, obs, angular_velocity, comet_ids)
        distance = math.hypot(tx - source.x, ty - source.y)
        if distance <= source.radius + target.radius:
            continue

        if crosses_sun(source.x, source.y, tx, ty):
            continue

        if not path_clear(source, target, tx, ty, all_planets):
            continue

        eta = distance / fleet_speed(ships)
        score = target_score(target, distance, eta, base_needed, comet_ids)
        angle = math.atan2(ty - source.y, tx - source.x)

        if best is None or score > best[0]:
            best = (score, target, angle, ships)

    if best is not None:
        _, target, angle, ships = best
        return target, angle, ships

    return pressure_weak_enemy(source, targets, all_planets, available)


def ships_needed(target: Planet) -> int:
    if target.owner == -1:
        return int(target.ships + 1)
    return int(target.ships + max(2, target.production + 1))


def reserve_for(source: Planet) -> int:
    return max(4, int(8 + source.production * 2))


def target_score(target: Planet, distance: float, eta: float, needed: int, comet_ids: set) -> float:
    owner_bonus = 12.0 if target.owner != -1 else 0.0
    comet_penalty = 10.0 if target.id in comet_ids else 0.0
    payoff = target.production * 22.0 + target.radius * 2.0 + owner_bonus
    cost = needed * 0.55 + distance * 0.35 + eta * 0.2 + comet_penalty
    return payoff - cost


def pressure_weak_enemy(
    source: Planet,
    targets: Sequence[Planet],
    all_planets: Sequence[Planet],
    available: int,
) -> Tuple[Planet, float, int] | None:
    enemies = [p for p in targets if p.owner != -1]
    if not enemies or available < 8:
        return None

    for target in sorted(enemies, key=lambda p: (p.ships, distance_between(source, p))):
        if crosses_sun(source.x, source.y, target.x, target.y):
            continue
        if not path_clear(source, target, target.x, target.y, all_planets):
            continue

        ships = max(1, int(available * 0.55))
        angle = math.atan2(target.y - source.y, target.x - source.x)
        return target, angle, ships

    return None


def predicted_target_position(
    target: Planet,
    source: Planet,
    ships: int,
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
) -> Tuple[float, float]:
    speed = fleet_speed(ships)
    guess_x, guess_y = target.x, target.y

    for _ in range(3):
        eta = math.hypot(guess_x - source.x, guess_y - source.y) / speed
        guess_x, guess_y = position_after(target, eta, obs, angular_velocity, comet_ids)

    return guess_x, guess_y


def position_after(
    planet: Planet,
    turns: float,
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
) -> Tuple[float, float]:
    if planet.id in comet_ids:
        comet_position = comet_position_after(planet.id, turns, obs)
        if comet_position is not None:
            return comet_position

    dx = planet.x - CENTER[0]
    dy = planet.y - CENTER[1]
    orbital_radius = math.hypot(dx, dy)
    if orbital_radius + planet.radius >= ROTATION_RADIUS_LIMIT or angular_velocity == 0.0:
        return planet.x, planet.y

    angle = math.atan2(dy, dx) + angular_velocity * turns
    return (
        CENTER[0] + math.cos(angle) * orbital_radius,
        CENTER[1] + math.sin(angle) * orbital_radius,
    )


def comet_position_after(planet_id: int, turns: float, obs: Any) -> Tuple[float, float] | None:
    groups = get_field(obs, "comets", []) or []
    for group in groups:
        planet_ids = get_field(group, "planet_ids", []) or []
        if planet_id not in planet_ids:
            continue

        comet_index = planet_ids.index(planet_id)
        paths = get_field(group, "paths", []) or []
        if comet_index >= len(paths):
            return None

        path = paths[comet_index]
        if not path:
            return None

        path_index = int(get_field(group, "path_index", 0))
        future_index = min(len(path) - 1, max(0, path_index + int(round(turns))))
        point = path[future_index]
        return float(point[0]), float(point[1])

    return None


def fleet_speed(ships: int) -> float:
    ships = max(1, int(ships))
    scaled = (math.log(ships) / math.log(1000.0)) ** 1.5
    return 1.0 + (MAX_SPEED - 1.0) * min(1.0, scaled)


def crosses_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    return distance_to_segment(CENTER[0], CENTER[1], x1, y1, x2, y2) <= SUN_RADIUS + 0.25


def path_clear(
    source: Planet,
    target: Planet,
    tx: float,
    ty: float,
    all_planets: Iterable[Planet],
) -> bool:
    total_distance = math.hypot(tx - source.x, ty - source.y)
    if total_distance <= 0:
        return False

    for planet in all_planets:
        if planet.id in (source.id, target.id):
            continue

        closest_distance = distance_to_segment(planet.x, planet.y, source.x, source.y, tx, ty)
        if closest_distance > planet.radius + 0.2:
            continue

        along = projection_fraction(planet.x, planet.y, source.x, source.y, tx, ty)
        hit_distance = total_distance * along
        if 0.0 < along < 1.0 and hit_distance < total_distance - target.radius:
            return False

    return True


def distance_to_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    t = projection_fraction(px, py, x1, y1, x2, y2)
    closest_x = x1 + (x2 - x1) * t
    closest_y = y1 + (y2 - y1) * t
    return math.hypot(px - closest_x, py - closest_y)


def projection_fraction(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return 0.0

    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    return max(0.0, min(1.0, t))


def distance_between(a: Planet, b: Planet) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def agent(obs: Any) -> List[List[float]]:
    return _agent_impl(obs)
