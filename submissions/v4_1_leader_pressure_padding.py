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
LEADER_PRESSURE_SAFETY_FACTOR = 1.18
PROJECTED_FLEET_TURNS = 55
MAX_INTERCEPT_TURNS = 80
MAX_GAME_TURNS = 500

_TURN_BY_GAME_PLAYER: dict[tuple[Any, ...], int] = {}


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
    angular_velocity = float(get_field(obs, "angular_velocity", 0.0))
    comet_ids = set(get_field(obs, "comet_planet_ids", []) or [])
    step = current_step(obs, planets, fleets, player, angular_velocity, comet_ids)

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not my_planets or not targets:
        return []

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
            fleets,
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
    fleets: Sequence[Fleet],
    planned_by_target: dict[int, int],
    player: int,
    owned_ratio: float,
    my_planet_count: int,
    step: int,
    available: int,
) -> Tuple[Planet, float, int] | None:
    best: Tuple[float, Planet, float, int] | None = None
    has_affordable_neutral = any(
        is_affordable_neutral(target, player, incoming, planned_by_target, available)
        for target in targets
    )
    gateway_target_id = opening_gateway_target(
        source,
        targets,
        all_planets,
        obs,
        angular_velocity,
        comet_ids,
        player,
        incoming,
        planned_by_target,
        available,
        step,
        my_planet_count,
    )

    for target in targets:
        if gateway_target_id is not None and target.id != gateway_target_id:
            continue

        if target.id != gateway_target_id and should_save_for_high_value_opening(
            source,
            target,
            targets,
            all_planets,
            obs,
            angular_velocity,
            comet_ids,
            player,
            incoming,
            planned_by_target,
            available,
            step,
            my_planet_count,
        ):
            continue

        if should_save_for_leader_strike(
            source,
            target,
            targets,
            all_planets,
            obs,
            angular_velocity,
            comet_ids,
            player,
            incoming,
            planned_by_target,
            fleets,
            available,
            step,
        ):
            continue

        incoming_by_owner = incoming.get(target.id, {})
        friendly_incoming = incoming_by_owner.get(player, 0) + planned_by_target.get(target.id, 0)
        opposing_incoming = sum(ships for owner, ships in incoming_by_owner.items() if owner != player)
        needed = remaining_ships_needed(target, player, friendly_incoming, opposing_incoming)
        if needed <= 0:
            continue

        leader_pressure = should_pressure_leader(target, all_planets, fleets, player, comet_ids, step, my_planet_count)
        if target.owner != -1 and not leader_pressure and should_delay_enemy_attack(step, my_planet_count, owned_ratio, has_affordable_neutral):
            continue

        if needed > available:
            continue

        safety_factor = LEADER_PRESSURE_SAFETY_FACTOR if leader_pressure else CAPTURE_SAFETY_FACTOR
        ships = ships_to_send(needed, available, safety_factor)
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
        angle = math.atan2(ty - source.y, tx - source.x)
        if not route_hits_target(source, target, angle, ships, all_planets, obs, angular_velocity, comet_ids, eta):
            continue

        arrival_needed = needed_after_travel(target, needed, eta)
        if arrival_needed > ships and arrival_needed <= available:
            ships = ships_to_send(arrival_needed, available, safety_factor)
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
            angle = math.atan2(ty - source.y, tx - source.x)
            if not route_hits_target(source, target, angle, ships, all_planets, obs, angular_velocity, comet_ids, eta):
                continue
            arrival_needed = needed_after_travel(target, needed, eta)
        if arrival_needed > ships:
            continue

        score = target_score(target, distance, eta, arrival_needed, comet_ids, owned_ratio, my_planet_count, step)
        if target.id == gateway_target_id:
            score += 110.0
        if leader_pressure:
            score += 150.0

        if best is None or score > best[0]:
            best = (score, target, angle, ships)

    if best is not None:
        _, target, angle, ships = best
        return target, angle, ships

    return None


def is_affordable_neutral(
    target: Planet,
    player: int,
    incoming: dict[int, dict[int, int]],
    planned_by_target: dict[int, int],
    available: int,
) -> bool:
    if target.owner != -1:
        return False

    incoming_by_owner = incoming.get(target.id, {})
    friendly_incoming = incoming_by_owner.get(player, 0) + planned_by_target.get(target.id, 0)
    opposing_incoming = sum(ships for owner, ships in incoming_by_owner.items() if owner != player)
    needed = remaining_ships_needed(target, player, friendly_incoming, opposing_incoming)
    return 0 < needed <= available


def should_save_for_high_value_opening(
    source: Planet,
    target: Planet,
    targets: Sequence[Planet],
    all_planets: Sequence[Planet],
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
    player: int,
    incoming: dict[int, dict[int, int]],
    planned_by_target: dict[int, int],
    available: int,
    step: int,
    my_planet_count: int,
) -> bool:
    if step >= 18 or my_planet_count > 1:
        return False
    if target.owner != -1 or target.production >= 5:
        return False

    wait_turns = 7 if source.production >= 4 and target.production <= 1 else 4
    if source.production <= 2:
        wait_turns = 5

    future_available = available + source.production * wait_turns
    target_distance = distance_between(source, target)
    for candidate in targets:
        if candidate.id == target.id or candidate.owner != -1:
            continue
        if candidate.production <= target.production:
            continue
        if candidate.production < 3:
            continue

        incoming_by_owner = incoming.get(candidate.id, {})
        friendly_incoming = incoming_by_owner.get(player, 0) + planned_by_target.get(candidate.id, 0)
        opposing_incoming = sum(ships for owner, ships in incoming_by_owner.items() if owner != player)
        needed = remaining_ships_needed(candidate, player, friendly_incoming, opposing_incoming)
        if needed > future_available:
            continue

        candidate_distance = distance_between(source, candidate)
        if candidate_distance > target_distance + 30.0 and candidate.production < target.production + 2:
            continue
        if candidate_distance > 65.0 and candidate.production < 5:
            continue

        aim = aim_target_position(candidate, source, needed, obs, angular_velocity, comet_ids)
        if aim is None:
            continue

        tx, ty = aim
        if crosses_sun(source.x, source.y, tx, ty):
            continue
        if not path_clear(source, candidate, tx, ty, all_planets):
            continue
        angle = math.atan2(ty - source.y, tx - source.x)
        eta = math.hypot(tx - source.x, ty - source.y) / fleet_speed(needed)
        if not route_hits_target(source, candidate, angle, needed, all_planets, obs, angular_velocity, comet_ids, eta):
            continue

        if candidate.production >= target.production + 1:
            return True

    return False


def should_save_for_leader_strike(
    source: Planet,
    target: Planet,
    targets: Sequence[Planet],
    all_planets: Sequence[Planet],
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
    player: int,
    incoming: dict[int, dict[int, int]],
    planned_by_target: dict[int, int],
    fleets: Sequence[Fleet],
    available: int,
    step: int,
) -> bool:
    if step < 70 or step > 210 or source.production < 4:
        return False
    if target.production >= 3 or target.owner == player or target.id in comet_ids:
        return False

    leader = pressure_leader_owner(all_planets, fleets, player, comet_ids, step)
    if leader is None:
        return False

    wait_turns = 18 if step < 140 else 12
    future_available = available + source.production * wait_turns
    for candidate in targets:
        if candidate.owner != leader or candidate.production < 4 or candidate.id in comet_ids:
            continue

        incoming_by_owner = incoming.get(candidate.id, {})
        friendly_incoming = incoming_by_owner.get(player, 0) + planned_by_target.get(candidate.id, 0)
        opposing_incoming = sum(ships for owner, ships in incoming_by_owner.items() if owner != player)
        needed = remaining_ships_needed(candidate, player, friendly_incoming, opposing_incoming)
        if needed <= available or needed > future_available:
            continue

        ships = max(needed, MIN_ATTACK_SHIPS)
        aim = aim_target_position(candidate, source, ships, obs, angular_velocity, comet_ids)
        if aim is None:
            continue

        tx, ty = aim
        distance = math.hypot(tx - source.x, ty - source.y)
        if distance > 78.0 or crosses_sun(source.x, source.y, tx, ty):
            continue
        if not path_clear(source, candidate, tx, ty, all_planets):
            continue

        eta = distance / fleet_speed(ships)
        arrival_needed = needed_after_travel(candidate, needed, eta)
        if arrival_needed > future_available:
            continue

        angle = math.atan2(ty - source.y, tx - source.x)
        if route_hits_target(source, candidate, angle, ships, all_planets, obs, angular_velocity, comet_ids, eta):
            return True

    return False


def opening_gateway_target(
    source: Planet,
    targets: Sequence[Planet],
    all_planets: Sequence[Planet],
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
    player: int,
    incoming: dict[int, dict[int, int]],
    planned_by_target: dict[int, int],
    available: int,
    step: int,
    my_planet_count: int,
) -> int | None:
    if step >= 12 or my_planet_count > 1 or source.production < 3:
        return None

    best: Tuple[float, int] | None = None
    future_available = available + source.production * 5
    for gateway in targets:
        if gateway.owner != -1 or gateway.production > 1 or gateway.id in comet_ids:
            continue

        incoming_by_owner = incoming.get(gateway.id, {})
        friendly_incoming = incoming_by_owner.get(player, 0) + planned_by_target.get(gateway.id, 0)
        opposing_incoming = sum(ships for owner, ships in incoming_by_owner.items() if owner != player)
        needed = remaining_ships_needed(gateway, player, friendly_incoming, opposing_incoming)
        if needed > future_available:
            continue

        ships = max(needed, MIN_ATTACK_SHIPS)
        aim = aim_target_position(gateway, source, ships, obs, angular_velocity, comet_ids)
        if aim is None:
            continue
        tx, ty = aim
        if crosses_sun(source.x, source.y, tx, ty):
            continue
        if not path_clear(source, gateway, tx, ty, all_planets):
            continue

        distance = math.hypot(tx - source.x, ty - source.y)
        eta = distance / fleet_speed(ships)
        angle = math.atan2(ty - source.y, tx - source.x)
        if not route_hits_target(source, gateway, angle, ships, all_planets, obs, angular_velocity, comet_ids, eta):
            continue

        blocked_value = best_blocked_value_behind_gateway(source, gateway, targets, all_planets)
        if blocked_value <= 0.0:
            continue

        score = blocked_value - needed * 0.8 - distance_between(source, gateway) * 0.35
        if best is None or score > best[0]:
            best = (score, gateway.id)

    return None if best is None else best[1]


def best_blocked_value_behind_gateway(
    source: Planet,
    gateway: Planet,
    targets: Sequence[Planet],
    all_planets: Sequence[Planet],
) -> float:
    best = 0.0
    for candidate in targets:
        if candidate.id == gateway.id or candidate.owner != -1 or candidate.production < 5:
            continue
        if distance_between(gateway, candidate) > 26.0:
            continue

        total_distance = distance_between(source, candidate)
        if total_distance <= distance_between(source, gateway):
            continue
        along = projection_fraction(gateway.x, gateway.y, source.x, source.y, candidate.x, candidate.y)
        if not 0.1 < along < 0.95:
            continue
        if distance_to_segment(gateway.x, gateway.y, source.x, source.y, candidate.x, candidate.y) > gateway.radius + 0.8:
            continue
        if path_clear(source, candidate, candidate.x, candidate.y, all_planets):
            continue

        value = candidate.production * 45.0 - candidate.ships * 0.35 - total_distance * 0.2
        best = max(best, value)

    return best


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


def ships_to_send(needed: int, available: int, safety_factor: float = CAPTURE_SAFETY_FACTOR) -> int:
    if needed > available:
        return 0

    padded = max(needed, int(math.ceil(needed * safety_factor)), MIN_ATTACK_SHIPS)
    return min(available, padded)


def reserve_for(source: Planet, step: int, my_planet_count: int) -> int:
    if my_planet_count <= 2 or step < 50:
        return 1
    if my_planet_count <= 5 or step < 120:
        return max(1, min(3, source.production))

    return max(2, source.production + 1)


def should_delay_enemy_attack(
    step: int,
    my_planet_count: int,
    owned_ratio: float,
    has_affordable_neutral: bool,
) -> bool:
    if has_affordable_neutral and (my_planet_count < 10 or step < 170):
        return True
    if my_planet_count < 6 and step < 140:
        return True
    if owned_ratio < 0.28 and step < 170:
        return True
    return False


def should_pressure_leader(
    target: Planet,
    all_planets: Sequence[Planet],
    fleets: Sequence[Fleet],
    player: int,
    comet_ids: set,
    step: int,
    my_planet_count: int,
) -> bool:
    if target.owner in (-1, player) or step < 35 or my_planet_count < 3:
        return False

    leader = pressure_leader_owner(all_planets, fleets, player, comet_ids, step)
    return leader is not None and target.owner == leader


def pressure_leader_owner(
    all_planets: Sequence[Planet],
    fleets: Sequence[Fleet],
    player: int,
    comet_ids: set,
    step: int,
) -> int | None:
    production = owner_production(all_planets, comet_ids)
    my_prod = production.get(player, 0)
    leader_owner = None
    leader_prod = 0
    for owner, prod in production.items():
        if owner == player or owner < 0:
            continue
        if prod > leader_prod:
            leader_owner = owner
            leader_prod = prod

    production_pressure = False
    if leader_owner is not None:
        lead = leader_prod - my_prod
        production_pressure = lead >= max(3, int(my_prod * 0.15))

    score_leader_owner = None
    score_leader = 0
    scores = owner_ship_scores(all_planets, fleets)
    my_score = scores.get(player, 0)
    for owner, score in scores.items():
        if owner == player or owner < 0:
            continue
        if score > score_leader:
            score_leader_owner = owner
            score_leader = score

    score_lead = score_leader - my_score
    score_pressure = (
        score_leader_owner is not None
        and step >= 90
        and score_lead >= max(80, int(my_score * 0.28))
    )

    if score_pressure:
        return score_leader_owner
    if production_pressure:
        return leader_owner

    return None


def owner_production(all_planets: Sequence[Planet], comet_ids: set) -> dict[int, int]:
    production: dict[int, int] = {}
    for planet in all_planets:
        if planet.id in comet_ids or planet.owner < 0:
            continue
        production[planet.owner] = production.get(planet.owner, 0) + int(planet.production)
    return production


def owner_ship_scores(all_planets: Sequence[Planet], fleets: Sequence[Fleet]) -> dict[int, int]:
    scores: dict[int, int] = {}
    for planet in all_planets:
        if planet.owner < 0:
            continue
        scores[planet.owner] = scores.get(planet.owner, 0) + int(planet.ships)
    for fleet in fleets:
        if fleet.owner < 0:
            continue
        scores[fleet.owner] = scores.get(fleet.owner, 0) + int(fleet.ships)
    return scores


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
        angle = math.atan2(ty - source.y, tx - source.x)
        if not route_hits_target(source, target, angle, ships, all_planets, obs, angular_velocity, comet_ids, eta):
            continue

        score = needed * 3.0 + target.production * 10.0 - eta - distance * 0.15

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

            start_px, start_py = position_after(planet, turn - 1, obs, angular_velocity, comet_ids)
            end_px, end_py = position_after(planet, turn, obs, angular_velocity, comet_ids)
            if distance_to_segment(start_px, start_py, x, y, nx, ny) <= planet.radius + 0.2:
                along = projection_fraction(start_px, start_py, x, y, nx, ny)
                hits.append((turn + along, planet))
                continue

            if distance_to_segment(nx, ny, start_px, start_py, end_px, end_py) <= planet.radius + 0.2:
                hits.append((turn + 0.95, planet))

        if hits:
            return min(hits, key=lambda item: item[0])[1]

        if nx < 0.0 or nx > BOARD_SIZE or ny < 0.0 or ny > BOARD_SIZE:
            return None

        x, y = nx, ny

    return None


def route_hits_target(
    source: Planet,
    target: Planet,
    angle: float,
    ships: int,
    planets: Sequence[Planet],
    obs: Any,
    angular_velocity: float,
    comet_ids: set,
    eta: float,
) -> bool:
    fleet = Fleet(
        -1,
        source.owner,
        source.x + math.cos(angle) * (source.radius + 0.05),
        source.y + math.sin(angle) * (source.radius + 0.05),
        angle,
        source.id,
        ships,
    )
    max_turns = max(PROJECTED_FLEET_TURNS, int(math.ceil(eta)) + 8)
    hit = projected_fleet_target(fleet, planets, obs, angular_velocity, comet_ids, max_turns=max_turns)
    return hit is not None and hit.id == target.id


def current_step(
    obs: Any,
    planets: Sequence[Planet],
    fleets: Sequence[Fleet],
    player: int,
    angular_velocity: float,
    comet_ids: set,
) -> int:
    raw_step = get_field(obs, "step", None)
    if raw_step is not None:
        step = safe_int(raw_step, 0)
        remember_step(obs, player, angular_velocity, step)
        return step

    inferred = infer_step_from_orbits(obs, planets, angular_velocity, comet_ids)
    key = turn_state_key(obs, player, angular_velocity)
    if key is None:
        return max(0, int(round(inferred or 0)))

    prior = _TURN_BY_GAME_PLAYER.get(key)
    if prior is not None and inferred is not None and inferred <= 1.5 and prior > 20 and not fleets:
        prior = None

    if prior is None:
        step = max(0, int(round(inferred or 0)))
    elif inferred is None or abs(angular_velocity) < 1e-9:
        step = prior + 1
    else:
        expected = prior + 1
        period = (2.0 * math.pi) / abs(angular_velocity)
        wraps = int(MAX_GAME_TURNS / period) + 3
        candidates = [inferred + period * wrap for wrap in range(wraps)]
        step = int(round(min(candidates, key=lambda candidate: abs(candidate - expected))))
        if abs(step - expected) > 4:
            step = expected

    step = max(0, min(MAX_GAME_TURNS, step))
    _TURN_BY_GAME_PLAYER[key] = step
    return step


def infer_step_from_orbits(
    obs: Any,
    planets: Sequence[Planet],
    angular_velocity: float,
    comet_ids: set,
) -> float | None:
    if abs(angular_velocity) < 1e-9:
        return None

    initial_rows = get_field(obs, "initial_planets", []) or []
    if not initial_rows:
        return None

    current_by_id = {planet.id: planet for planet in planets}
    inferred_steps: List[float] = []
    for row in initial_rows:
        try:
            initial = Planet(*row)
        except (TypeError, ValueError):
            continue

        if initial.id in comet_ids:
            continue
        current = current_by_id.get(initial.id)
        if current is None:
            continue
        if distance_from_center(initial.x, initial.y) + initial.radius >= ROTATION_RADIUS_LIMIT:
            continue
        if distance_from_center(current.x, current.y) + current.radius >= ROTATION_RADIUS_LIMIT + 1.0:
            continue

        start_angle = math.atan2(initial.y - CENTER[1], initial.x - CENTER[0])
        current_angle = math.atan2(current.y - CENTER[1], current.x - CENTER[0])
        if angular_velocity > 0:
            delta = (current_angle - start_angle) % (2.0 * math.pi)
        else:
            delta = (start_angle - current_angle) % (2.0 * math.pi)
        inferred_steps.append(delta / abs(angular_velocity))

    if not inferred_steps:
        return None

    inferred_steps.sort()
    return inferred_steps[len(inferred_steps) // 2]


def turn_state_key(obs: Any, player: int, angular_velocity: float) -> tuple[Any, ...] | None:
    initial_rows = get_field(obs, "initial_planets", []) or []
    if not initial_rows:
        return None

    signature = []
    for row in initial_rows:
        try:
            signature.append(
                (
                    int(row[0]),
                    int(row[1]),
                    round(float(row[2]), 3),
                    round(float(row[3]), 3),
                    round(float(row[4]), 3),
                    int(row[6]),
                )
            )
        except (TypeError, ValueError, IndexError):
            continue

    if not signature:
        return None

    return (player, round(float(angular_velocity), 8), tuple(signature))


def remember_step(obs: Any, player: int, angular_velocity: float, step: int) -> None:
    key = turn_state_key(obs, player, angular_velocity)
    if key is not None:
        _TURN_BY_GAME_PLAYER[key] = max(0, min(MAX_GAME_TURNS, step))


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
