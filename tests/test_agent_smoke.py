import math
import unittest

from main import agent


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


if __name__ == "__main__":
    unittest.main()
