import math
import unittest

from main import Fleet, Planet, agent, projected_fleet_target


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


if __name__ == "__main__":
    unittest.main()
