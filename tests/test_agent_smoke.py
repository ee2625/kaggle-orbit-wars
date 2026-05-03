import math
import unittest

from main import (
    Fleet,
    Planet,
    agent,
    current_step,
    path_clear,
    projected_fleet_target,
    route_hits_target,
    should_save_for_high_value_opening,
    should_save_for_low_production_breakout,
    should_save_for_leader_strike,
    should_siege_remaining_enemy,
    should_prioritize_leader_pressure_over_rebalance,
)


class AgentSmokeTest(unittest.TestCase):
    def test_returns_legal_looking_moves(self):
        obs = {
            "player": 0,
            "angular_velocity": 0.03,
            "comet_planet_ids": [],
            "comets": [],
            "planets": [
                [0, 0, 15.0, 15.0, 2.0, 60, 3],
                [1, -1, 25.0, 18.0, 2.1, 12, 4],
                [2, -1, 85.0, 85.0, 2.1, 12, 4],
                [3, 1, 80.0, 80.0, 2.0, 60, 3],
            ],
            "fleets": [],
        }

        moves = agent(obs)

        self.assertIsInstance(moves, list)
        self.assertGreaterEqual(len(moves), 1)
        for move in moves:
            self.assertEqual(len(move), 3)
            self.assertEqual(move[0], 0)
            self.assertTrue(math.isfinite(move[1]))
            self.assertIsInstance(move[2], int)
            self.assertGreater(move[2], 0)
            self.assertLessEqual(move[2], 60)

    def test_empty_when_no_planets(self):
        self.assertEqual(agent({"player": 0, "planets": []}), [])

    def test_does_not_duplicate_sufficient_friendly_incoming(self):
        obs = {
            "player": 0,
            "angular_velocity": 0.0,
            "comet_planet_ids": [],
            "comets": [],
            "planets": [
                [0, 0, 10.0, 10.0, 2.0, 60, 3],
                [1, -1, 30.0, 10.0, 2.0, 20, 3],
            ],
            "fleets": [
                [99, 0, 20.0, 10.0, 0.0, 0, 25],
            ],
        }

        self.assertEqual(agent(obs), [])

    def test_reinforces_threatened_owned_planet_first(self):
        obs = {
            "player": 0,
            "angular_velocity": 0.0,
            "comet_planet_ids": [],
            "comets": [],
            "planets": [
                [0, 0, 10.0, 10.0, 2.0, 60, 3],
                [1, 0, 30.0, 10.0, 2.0, 5, 3],
                [2, 1, 80.0, 80.0, 2.0, 60, 3],
            ],
            "fleets": [
                [99, 1, 20.0, 10.0, 0.0, 2, 20],
            ],
        }

        moves = agent(obs)

        self.assertGreaterEqual(len(moves), 1)
        self.assertEqual(moves[0][0], 0)
        self.assertAlmostEqual(moves[0][1], 0.0)
        self.assertGreaterEqual(moves[0][2], 17)

    def test_orbiting_target_launch_uses_real_intercept(self):
        obs = {
            "player": 0,
            "angular_velocity": 0.046110546288126206,
            "comet_planet_ids": [],
            "comets": [],
            "planets": [
                [12, 0, 64.70017556467475, 79.95155009265068, 1.0, 18, 1],
                [14, -1, 79.95155009265068, 35.29982443532525, 1.0, 8, 1],
                [15, 1, 35.29982443532525, 20.04844990734932, 1.0, 18, 1],
            ],
            "fleets": [],
        }

        moves = agent(obs)
        planets = [Planet(*planet) for planet in obs["planets"]]
        source = planets[0]
        move = moves[0]
        fleet = Fleet(
            -1,
            0,
            source.x + math.cos(move[1]) * (source.radius + 0.05),
            source.y + math.sin(move[1]) * (source.radius + 0.05),
            move[1],
            source.id,
            move[2],
        )

        target = projected_fleet_target(
            fleet,
            planets,
            obs,
            obs["angular_velocity"],
            set(),
            max_turns=80,
        )

        self.assertIsNotNone(target)
        self.assertEqual(target.id, 14)

    def test_infers_step_when_observation_omits_step(self):
        angular_velocity = 0.04
        turn = 23
        orbital_radius = 30.0
        x = 50.0 + math.cos(angular_velocity * turn) * orbital_radius
        y = 50.0 + math.sin(angular_velocity * turn) * orbital_radius
        obs = {
            "player": 2,
            "angular_velocity": angular_velocity,
            "initial_planets": [
                [7, -1, 80.0, 50.0, 1.0, 12, 3],
            ],
            "planets": [
                [7, -1, x, y, 1.0, 12, 3],
            ],
            "fleets": [],
            "comet_planet_ids": [],
        }

        planets = [Planet(*planet) for planet in obs["planets"]]

        self.assertEqual(current_step(obs, planets, [], 2, angular_velocity, set()), turn)

    def test_route_validation_catches_moving_blocker(self):
        source = Planet(1, 0, 1.0, 50.0, 1.0, 1000, 1)
        target = Planet(2, -1, 99.0, 50.0, 1.0, 10, 1)
        blocker = Planet(
            3,
            -1,
            50.0 + math.cos(-1.0) * 20.0,
            50.0 + math.sin(-1.0) * 20.0,
            1.0,
            10,
            1,
        )
        planets = [source, target, blocker]
        obs = {
            "player": 0,
            "angular_velocity": 0.1,
            "comet_planet_ids": [],
            "comets": [],
            "planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production] for p in planets],
        }

        self.assertTrue(path_clear(source, target, target.x, target.y, planets))
        self.assertFalse(route_hits_target(source, target, 0.0, 1000, planets, obs, 0.1, set(), 17.0))

    def test_saves_low_value_spend_for_leader_strike(self):
        source = Planet(1, 0, 10.0, 10.0, 2.0, 50, 5)
        low_value = Planet(2, -1, 10.0, 25.0, 1.0, 5, 1)
        leader_anchor = Planet(3, 1, 80.0, 10.0, 2.0, 45, 5)
        planets = [source, low_value, leader_anchor]
        obs = {
            "player": 0,
            "step": 100,
            "angular_velocity": 0.0,
            "comet_planet_ids": [],
            "comets": [],
            "planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production] for p in planets],
        }
        leader_fleet = Fleet(99, 1, 70.0, 80.0, 0.0, 3, 100)

        self.assertTrue(
            should_save_for_leader_strike(
                source,
                low_value,
                [low_value, leader_anchor],
                planets,
                obs,
                0.0,
                set(),
                0,
                {},
                {},
                [leader_fleet],
                35,
                100,
            )
        )

    def test_rebalances_surplus_from_rear_to_frontier(self):
        obs = {
            "player": 0,
            "step": 80,
            "angular_velocity": 0.0,
            "comet_planet_ids": [],
            "comets": [],
            "initial_planets": [
                [0, 0, 15.0, 85.0, 2.0, 10, 5],
                [4, 1, 82.0, 80.0, 2.0, 10, 5],
                [5, 2, 85.0, 15.0, 2.0, 10, 5],
                [6, 3, 15.0, 15.0, 2.0, 10, 5],
            ],
            "planets": [
                [0, 0, 15.0, 85.0, 2.0, 90, 5],
                [1, 0, 70.0, 80.0, 2.0, 12, 4],
                [2, 0, 15.0, 95.0, 2.0, 20, 2],
                [3, 0, 5.0, 80.0, 2.0, 20, 2],
                [4, 1, 82.0, 80.0, 2.0, 200, 5],
            ],
            "fleets": [],
        }

        moves = agent(obs)

        self.assertTrue(any(move[0] == 0 and move[2] >= 6 for move in moves))

    def test_skips_forward_rebalance_in_two_player_games(self):
        obs = {
            "player": 0,
            "step": 80,
            "angular_velocity": 0.0,
            "comet_planet_ids": [],
            "comets": [],
            "initial_planets": [
                [0, 0, 15.0, 85.0, 2.0, 10, 5],
                [4, 1, 82.0, 80.0, 2.0, 10, 5],
            ],
            "planets": [
                [0, 0, 15.0, 85.0, 2.0, 90, 5],
                [1, 0, 70.0, 80.0, 2.0, 12, 4],
                [2, 0, 15.0, 95.0, 2.0, 20, 2],
                [3, 0, 5.0, 80.0, 2.0, 20, 2],
                [4, 1, 82.0, 80.0, 2.0, 200, 5],
            ],
            "fleets": [],
        }

        moves = agent(obs)

        self.assertFalse(any(move[0] == 0 and move[2] >= 6 for move in moves))

    def test_prioritizes_leader_pressure_over_rebalance_when_behind(self):
        planets = [
            Planet(0, 0, 15.0, 85.0, 2.0, 90, 4),
            Planet(1, 0, 70.0, 80.0, 2.0, 12, 3),
            Planet(2, 1, 82.0, 80.0, 2.0, 200, 5),
            Planet(3, 1, 75.0, 70.0, 2.0, 200, 5),
            Planet(4, 2, 15.0, 15.0, 2.0, 10, 1),
            Planet(5, 3, 85.0, 15.0, 2.0, 10, 1),
        ]

        self.assertTrue(should_prioritize_leader_pressure_over_rebalance(planets, set(), 0, 90))

    def test_four_player_rich_opening_does_not_wait_on_close_prod_two(self):
        source = Planet(5, 0, 35.0, 96.0, 2.6, 15, 5)
        close = Planet(25, -1, 40.0, 86.0, 2.1, 9, 2)
        high_value = Planet(1, -1, 45.0, 75.0, 2.6, 20, 5)
        obs = {
            "player": 0,
            "step": 1,
            "angular_velocity": 0.0,
            "comet_planet_ids": [],
            "comets": [],
            "planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production] for p in [source, close, high_value]],
        }

        self.assertFalse(
            should_save_for_high_value_opening(
                source,
                close,
                [close, high_value],
                [source, close, high_value],
                obs,
                0.0,
                set(),
                0,
                {},
                {},
                14,
                1,
                1,
                4,
            )
        )

    def test_low_production_opening_saves_for_breakout_target(self):
        source = Planet(0, 0, 90.0, 90.0, 1.0, 7, 1)
        first_low = Planet(1, -1, 90.0, 82.0, 1.0, 7, 1)
        second_low = Planet(2, -1, 80.0, 82.0, 1.0, 7, 1)
        breakout = Planet(3, -1, 70.0, 90.0, 2.0, 16, 3)
        planets = [source, first_low, second_low, breakout]
        obs = {
            "player": 0,
            "step": 12,
            "angular_velocity": 0.0,
            "comet_planet_ids": [],
            "comets": [],
            "planets": [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production] for p in planets],
        }

        self.assertTrue(
            should_save_for_low_production_breakout(
                source,
                second_low,
                [first_low, second_low, breakout],
                planets,
                obs,
                0.0,
                set(),
                0,
                {first_low.id: {0: 8}},
                {},
                6,
                12,
                2,
            )
        )

    def test_sieges_remaining_enemy_when_far_ahead(self):
        my_planets = [
            Planet(0, 0, 10.0, 10.0, 2.0, 220, 8),
            Planet(1, 0, 20.0, 10.0, 2.0, 180, 8),
        ]
        enemy_planets = [
            Planet(2, 1, 80.0, 80.0, 2.0, 150, 3),
            Planet(3, 1, 75.0, 75.0, 2.0, 120, 2),
        ]

        self.assertTrue(should_siege_remaining_enemy(enemy_planets, my_planets + enemy_planets, [], set(), 0))


if __name__ == "__main__":
    unittest.main()
