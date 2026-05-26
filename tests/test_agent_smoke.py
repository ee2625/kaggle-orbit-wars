import math
import unittest

from main import (
    Fleet,
    Planet,
    agent,
    build_world,
    crosses_sun,
    fleet_speed,
    projected_fleet_target,
)


def assert_legal_moves(testcase, obs, moves):
    owned = {int(planet[0]): int(planet[5]) for planet in obs.get("planets", []) if int(planet[1]) == obs.get("player", 0)}
    spent = {}
    testcase.assertIsInstance(moves, list)
    for move in moves:
        testcase.assertEqual(len(move), 3)
        source_id = int(move[0])
        ships = int(move[2])
        testcase.assertIn(source_id, owned)
        testcase.assertTrue(math.isfinite(float(move[1])))
        testcase.assertGreater(ships, 0)
        spent[source_id] = spent.get(source_id, 0) + ships
        testcase.assertLessEqual(spent[source_id], owned[source_id])


class AgentSmokeTest(unittest.TestCase):
    def test_returns_legal_looking_moves(self):
        obs = {
            "player": 0,
            "step": 0,
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

        self.assertGreaterEqual(len(moves), 1)
        assert_legal_moves(self, obs, moves)

    def test_empty_when_no_owned_planets(self):
        self.assertEqual(agent({"player": 0, "planets": []}), [])

    def test_four_player_opening_returns_legal_moves(self):
        obs = {
            "player": 2,
            "step": 0,
            "angular_velocity": 0.025,
            "initial_planets": [],
            "comet_planet_ids": [],
            "comets": [],
            "planets": [
                [0, 0, 15.0, 15.0, 2.0, 10, 4],
                [1, 1, 85.0, 15.0, 2.0, 10, 4],
                [2, 2, 15.0, 85.0, 2.0, 60, 4],
                [3, 3, 85.0, 85.0, 2.0, 10, 4],
                [4, -1, 24.0, 78.0, 2.2, 9, 5],
                [5, -1, 44.0, 82.0, 1.7, 8, 3],
                [6, -1, 75.0, 24.0, 2.2, 9, 5],
            ],
            "fleets": [],
        }

        moves = agent(obs)

        self.assertGreaterEqual(len(moves), 1)
        assert_legal_moves(self, obs, moves)

    def test_world_model_reads_observation(self):
        obs = {
            "player": 1,
            "step": 12,
            "angular_velocity": 0.04,
            "comet_planet_ids": [9],
            "comets": [],
            "planets": [
                [0, 1, 20.0, 20.0, 2.0, 30, 3],
                [9, -1, 40.0, 20.0, 1.0, 5, 1],
            ],
            "fleets": [],
        }

        world = build_world(obs)

        self.assertEqual(world.player, 1)
        self.assertEqual(world.step, 12)
        self.assertEqual(len(world.my_planets), 1)
        self.assertEqual(world.comet_ids, {9})

    def test_world_model_infers_step_from_orbiting_planets(self):
        angular_velocity = 0.04
        turn = 23
        radius = 24.0
        x = 50.0 + math.cos(angular_velocity * turn) * radius
        y = 50.0 + math.sin(angular_velocity * turn) * radius
        obs = {
            "player": 0,
            "angular_velocity": angular_velocity,
            "initial_planets": [
                [0, 0, 74.0, 50.0, 1.0, 10, 3],
            ],
            "comet_planet_ids": [],
            "comets": [],
            "planets": [
                [0, 0, x, y, 1.0, 20, 3],
            ],
            "fleets": [],
        }

        world = build_world(obs)

        self.assertEqual(world.step, turn)

    def test_fleet_speed_scales_with_size(self):
        self.assertLess(fleet_speed(1), fleet_speed(50))
        self.assertLess(fleet_speed(50), fleet_speed(1000))
        self.assertLessEqual(fleet_speed(1000), 6.0)

    def test_projected_target_compatibility_helper(self):
        source = Planet(0, 0, 10.0, 10.0, 2.0, 50, 3)
        target = Planet(1, -1, 30.0, 10.0, 2.0, 5, 2)
        fleet = Fleet(99, 0, 12.1, 10.0, 0.0, source.id, 20)

        hit = projected_fleet_target(fleet, [source, target], max_turns=20)

        self.assertIsNotNone(hit)
        self.assertEqual(hit.id, target.id)
        self.assertFalse(crosses_sun(source.x, source.y, target.x, target.y))


if __name__ == "__main__":
    unittest.main()
