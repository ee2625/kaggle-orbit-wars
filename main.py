from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple


BOARD_SIZE = 100.0
CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
MIN_ATTACK_SHIPS = 6
CAPTURE_SAFETY_FACTOR = 1.1
PROJECTED_FLEET_TURNS = 55
MAX_INTERCEPT_TURNS = 80


@dataclass(frozen=True)
class Planet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


@dataclass(frozen=True)
class Fleet:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


def _agent_impl(obs: Any) -> List[List[float]]:
    """Kaggle entrypoint.

    Returns moves in the form [from_planet_id, angle_in_radians, num_ships].
    This baseline aims for robust legal play over cleverness.
    """
    planets = [Planet(*p) for p in get_field(obs, "planets", [])]
    fleets = [Fleet(*f) for f in get_field(obs, "fleets", [])]
    player = int(get_field(obs, "player", 0))

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not my_planets or not targets:
        return []

    step = int(get_field(obs, "step", 0) or 0)
    angular_velocity = float(get_field(obs, "angular_velocity", 0.0))
    comet_ids = set(get_field(obs, "comet_planet_ids", []) or [])
    incoming = build_incoming_map(fleets, planets, obs, angular_velocity, comet_ids)
    planned_by_target: dict[int, int] = {}
    defense_needs = build_defense_needs(my_planets, incoming, player)
    my_durable_planets = [p for p in my_planets if p.id not in comet_ids]
    owned_ratio = len(my_durable_planets) / max(1, len([p for p in planets if p.id not in comet_ids]))
    moves: List[List[float]] = []

    # Strong planets should decide first, before smaller worlds spend themselves.
    for source in sorted(my_planets, key=lambda p: (p.ships, p.production), reverse=True):
        reserve = reserve_for(source, step, len(my_durable_planets)) + threatened_reserve(source, incoming, player)
        available = int(source.ships - reserve)
        if available <= 1:
            continue

        defense = choose_defense_target(
            source,
            my_planets,
            defense_needs,
            planets,
            obs,
            angular_velocity,
            comet_ids,
            available,
        )
        if defense is not None:
            target, angle, ships = defense
            moves.append([source.id, angle, ships])
            defense_needs[target.id] = max(0, defense_needs.get(target.id, 0) - ships)
            planned_by_target[target.id] = planned_by_target.get(target.id, 0) + ships
            continue

        choice = choose_target(
            source,
            targets,
            planets,
            obs,
            angular_velocity,
            comet_ids,
            incoming,
            planned_by_target,
            player,
            owned_ratio,
            len(my_durable_planets),
            step,
            available,
        )
        if choice is None:
            continue

        target, angle, ships = choice
        if ships <= 0:
            continue

        moves.append([source.id, angle, ships])
        planned_by_target[target.id] = planned_by_target.get(target.id, 0) + ships

    return moves


def choose_target(
    source: Planet,
    targets: Sequence[Planet],
    all_planets: Sequence[Planet],
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
    incoming: dict[int, dict[int, int]],
    planned_by_target: dict[int, int],
    player: int,
    owned_ratio: float,
    my_planet_count: int,
    step: int,
    available: int,
) -> Tuple[Planet, float, int] | None:
    best: Tuple[float, Planet, float, int] | None = None

    for target in targets:
        incoming_by_owner = incoming.get(target.id, {})
        friendly_incoming = incoming_by_owner.get(player, 0) + planned_by_target.get(target.id, 0)
        opposing_incoming = sum(ships for owner, ships in incoming_by_owner.items() if owner != player)
        needed = remaining_ships_needed(target, player, friendly_incoming, opposing_incoming)
        if needed <= 0:
            continue

        if target.owner != -1 and should_delay_enemy_attack(step, my_planet_count, owned_ratio):
            continue

        if needed > available:
            continue

        ships = ships_to_send(needed, available)
        if ships <= 0:
            continue

        aim = aim_target_position(target, source, ships, obs, angular_velocity, comet_ids)
        if aim is None:
            continue
        tx, ty = aim
        distance = math.hypot(tx - source.x, ty - source.y)
        if distance <= source.radius + target.radius:
            continue

        if crosses_sun(source.x, source.y, tx, ty):
            continue

        if not path_clear(source, target, tx, ty, all_planets):
            continue

        eta = distance / fleet_speed(ships)
        arrival_needed = needed_after_travel(target, needed, eta)
        if arrival_needed > ships and arrival_needed <= available:
            ships = ships_to_send(arrival_needed, available)
            aim = aim_target_position(target, source, ships, obs, angular_velocity, comet_ids)
            if aim is None:
                continue
            tx, ty = aim
            distance = math.hypot(tx - source.x, ty - source.y)
            if crosses_sun(source.x, source.y, tx, ty):
                continue
            if not path_clear(source, target, tx, ty, all_planets):
                continue
            eta = distance / fleet_speed(ships)
            arrival_needed = needed_after_travel(target, needed, eta)
        if arrival_needed > ships:
            continue

        score = target_score(target, distance, eta, arrival_needed, comet_ids, owned_ratio, my_planet_count, step)
        angle = math.atan2(ty - source.y, tx - source.x)

        if best is None or score > best[0]:
            best = (score, target, angle, ships)

    if best is not None:
        _, target, angle, ships = best
        return target, angle, ships

    return pressure_weak_enemy(
        source,
        targets,
        all_planets,
        obs,
        angular_velocity,
        comet_ids,
        incoming,
        planned_by_target,
        player,
        my_planet_count,
        step,
        available,
    )


def remaining_ships_needed(
    target: Planet,
    player: int,
    friendly_incoming: int = 0,
    opposing_incoming: int = 0,
) -> int:
    if target.owner == -1:
        needed = target.ships + 1
    elif target.owner == player:
        needed = 0
    else:
        needed = target.ships + max(2, target.production + 1)

    return int(math.ceil(needed + opposing_incoming - friendly_incoming))


def needed_after_travel(target: Planet, needed: int, eta: float) -> int:
    if target.owner == -1:
        return needed

    production_buffer = int(math.ceil(eta * target.production * 0.65))
    return needed + production_buffer


def ships_to_send(needed: int, available: int) -> int:
    if needed > available:
        return 0

    padded = max(needed, int(math.ceil(needed * CAPTURE_SAFETY_FACTOR)), MIN_ATTACK_SHIPS)
    return min(available, padded)


def reserve_for(source: Planet, step: int, my_planet_count: int) -> int:
    if my_planet_count <= 2 or step < 50:
        return 1
    if my_planet_count <= 5 or step < 120:
        return max(1, min(3, source.production))

    return max(2, source.production + 1)


def should_delay_enemy_attack(step: int, my_planet_count: int, owned_ratio: float) -> bool:
    if my_planet_count < 4 and step < 90:
        return True
    if owned_ratio < 0.22 and step < 120:
        return True
    return False


def target_score(
    target: Planet,
    distance: float,
    eta: float,
    needed: int,
    comet_ids: set,
    owned_ratio: float,
    my_planet_count: int,
    step: int,
) -> float:
    neutral_bonus = 18.0 if target.owner == -1 else 0.0
    enemy_bonus = 10.0 if target.owner != -1 else 0.0
    early_economy_bonus = max(0, 5 - my_planet_count) * 8.0 if target.owner == -1 else 0.0
    comet_penalty = 24.0 if target.id in comet_ids else 0.0
    enemy_timing_penalty = max(0.0, 0.45 - owned_ratio) * 90.0 if target.owner != -1 else 0.0
    late_enemy_bonus = min(20.0, max(0, step - 100) * 0.12) if target.owner != -1 else 0.0
    payoff = target.production * 34.0 + target.radius * 2.0 + neutral_bonus + enemy_bonus + early_economy_bonus + late_enemy_bonus
    cost = needed * 0.46 + distance * 0.22 + eta * 0.34 + comet_penalty + enemy_timing_penalty
    return payoff - cost


def pressure_weak_enemy(
    source: Planet,
    targets: Sequence[Planet],
    all_planets: Sequence[Planet],
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
    incoming: dict[int, dict[int, int]],
    planned_by_target: dict[int, int],
    player: int,
    my_planet_count: int,
    step: int,
    available: int,
) -> Tuple[Planet, float, int] | None:
    enemies = [p for p in targets if p.owner != -1]
    if not enemies or available < 8:
        return None
    if my_planet_count < 5 and step < 110:
        return None

    for target in sorted(enemies, key=lambda p: (p.ships, distance_between(source, p))):
        incoming_by_owner = incoming.get(target.id, {})
        friendly_incoming = incoming_by_owner.get(player, 0) + planned_by_target.get(target.id, 0)
        if friendly_incoming >= target.ships:
            continue

        ships = max(1, int(available * 0.55))
        aim = aim_target_position(target, source, ships, obs, angular_velocity, comet_ids)
        if aim is None:
            continue
        tx, ty = aim

        if crosses_sun(source.x, source.y, tx, ty):
            continue
        if not path_clear(source, target, tx, ty, all_planets):
            continue

        angle = math.atan2(ty - source.y, tx - source.x)
        return target, angle, ships

    return None


def build_defense_needs(
    my_planets: Sequence[Planet],
    incoming: dict[int, dict[int, int]],
    player: int,
) -> dict[int, int]:
    needs: dict[int, int] = {}
    for planet in my_planets:
        incoming_by_owner = incoming.get(planet.id, {})
        friendly = incoming_by_owner.get(player, 0)
        hostile = sum(ships for owner, ships in incoming_by_owner.items() if owner != player)
        need = hostile + 2 - friendly - planet.ships
        if need > 0:
            needs[planet.id] = int(need)
    return needs


def threatened_reserve(planet: Planet, incoming: dict[int, dict[int, int]], player: int) -> int:
    incoming_by_owner = incoming.get(planet.id, {})
    friendly = incoming_by_owner.get(player, 0)
    hostile = sum(ships for owner, ships in incoming_by_owner.items() if owner != player)
    return max(0, hostile - friendly)


def choose_defense_target(
    source: Planet,
    my_planets: Sequence[Planet],
    defense_needs: dict[int, int],
    all_planets: Sequence[Planet],
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
    available: int,
) -> Tuple[Planet, float, int] | None:
    best: Tuple[float, Planet, float, int] | None = None

    for target in my_planets:
        needed = defense_needs.get(target.id, 0)
        if target.id == source.id or needed <= 0:
            continue

        ships = min(available, needed)
        if ships <= 0:
            continue

        aim = aim_target_position(target, source, ships, obs, angular_velocity, comet_ids)
        if aim is None:
            continue
        tx, ty = aim
        if crosses_sun(source.x, source.y, tx, ty):
            continue
        if not path_clear(source, target, tx, ty, all_planets):
            continue

        distance = math.hypot(tx - source.x, ty - source.y)
        eta = distance / fleet_speed(ships)
        score = needed * 3.0 + target.production * 10.0 - eta - distance * 0.15
        angle = math.atan2(ty - source.y, tx - source.x)

        if best is None or score > best[0]:
            best = (score, target, angle, ships)

    if best is None:
        return None

    _, target, angle, ships = best
    return target, angle, ships


def aim_target_position(
    target: Planet,
    source: Planet,
    ships: int,
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
) -> Tuple[float, float] | None:
    if is_moving_planet(target, angular_velocity, comet_ids):
        return intercept_target_position(target, source, ships, obs, angular_velocity, comet_ids)

    return predicted_target_position(target, source, ships, obs, angular_velocity, comet_ids)


def is_moving_planet(planet: Planet, angular_velocity: float, comet_ids: set) -> bool:
    if planet.id in comet_ids:
        return True

    if angular_velocity == 0.0:
        return False

    return distance_from_center(planet.x, planet.y) + planet.radius < ROTATION_RADIUS_LIMIT


def intercept_target_position(
    target: Planet,
    source: Planet,
    ships: int,
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
) -> Tuple[float, float] | None:
    speed = fleet_speed(ships)
    best: Tuple[float, float, float] | None = None

    for turn in range(1, MAX_INTERCEPT_TURNS + 1):
        tx, ty = position_after(target, float(turn), obs, angular_velocity, comet_ids)
        travel_turns = math.hypot(tx - source.x, ty - source.y) / speed
        error = abs(travel_turns - turn)

        if best is None or error < best[0]:
            best = (error, tx, ty)

    if best is None:
        return None

    tolerance = max(0.65, min(1.35, (target.radius + 0.35) / speed + 0.35))
    if best[0] > tolerance:
        return None

    return best[1], best[2]


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
    orbital_radius = distance_from_center(planet.x, planet.y)
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


def build_incoming_map(
    fleets: Sequence[Fleet],
    planets: Sequence[Planet],
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
) -> dict[int, dict[int, int]]:
    incoming: dict[int, dict[int, int]] = {}
    for fleet in fleets:
        target = projected_fleet_target(fleet, planets, obs, angular_velocity, comet_ids)
        if target is None:
            continue

        by_owner = incoming.setdefault(target.id, {})
        by_owner[fleet.owner] = by_owner.get(fleet.owner, 0) + int(fleet.ships)

    return incoming


def projected_fleet_target(
    fleet: Fleet,
    planets: Sequence[Planet],
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
    max_turns: int = PROJECTED_FLEET_TURNS,
) -> Planet | None:
    speed = fleet_speed(fleet.ships)
    dx = math.cos(fleet.angle) * speed
    dy = math.sin(fleet.angle) * speed
    x = fleet.x
    y = fleet.y

    for turn in range(1, max_turns + 1):
        nx = x + dx
        ny = y + dy

        if crosses_sun(x, y, nx, ny):
            return None

        hits: List[Tuple[float, Planet]] = []
        for planet in planets:
            if planet.id == fleet.from_planet_id:
                continue

            px, py = position_after(planet, turn, obs, angular_velocity, comet_ids)
            if distance_to_segment(px, py, x, y, nx, ny) <= planet.radius + 0.2:
                along = projection_fraction(px, py, x, y, nx, ny)
                hits.append((turn + along, planet))

        if hits:
            return min(hits, key=lambda item: item[0])[1]

        if nx < 0.0 or nx > BOARD_SIZE or ny < 0.0 or ny > BOARD_SIZE:
            return None

        x, y = nx, ny

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


def distance_from_center(x: float, y: float) -> float:
    return math.hypot(x - CENTER[0], y - CENTER[1])


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def agent(obs: Any) -> List[List[float]]:
    return _agent_impl(obs)
