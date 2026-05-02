import unittest

from scripts.backtest import (
    EpisodeResult,
    build_lineup,
    classify_result,
    final_scores,
    reward_margin,
    score_margin,
    summarize_results,
)


class BacktestTest(unittest.TestCase):
    def test_build_lineup_places_agent_at_requested_seat(self):
        lineup, index = build_lineup("main.py", ["random", "random", "random"], 2)

        self.assertEqual(index, 2)
        self.assertEqual(lineup, ["random", "random", "main.py", "random"])

    def test_rejects_invalid_player_count(self):
        with self.assertRaises(ValueError):
            build_lineup("main.py", ["random", "random"], 0)

    def test_classifies_result(self):
        self.assertEqual(classify_result([1, -1], ["DONE", "DONE"], 0), "win")
        self.assertEqual(classify_result([1, 1], ["DONE", "DONE"], 0), "tie")
        self.assertEqual(classify_result([-1, 1], ["DONE", "DONE"], 0), "loss")
        self.assertEqual(classify_result([None, 1], ["DONE", "DONE"], 0), "unknown")
        self.assertEqual(classify_result([-1, 1], ["ERROR", "DONE"], 0), "error")

    def test_reward_margin_compares_against_best_opponent(self):
        self.assertEqual(reward_margin([3, 1, 2, -1], 0), 1.0)

    def test_final_scores_add_planets_and_fleets(self):
        observation = {
            "planets": [
                [0, 0, 0.0, 0.0, 1.0, 10, 1],
                [1, 1, 0.0, 0.0, 1.0, 5, 1],
                [2, -1, 0.0, 0.0, 1.0, 99, 1],
            ],
            "fleets": [
                [10, 0, 0.0, 0.0, 0.0, 0, 3],
                [11, 1, 0.0, 0.0, 0.0, 1, 7],
            ],
        }

        self.assertEqual(final_scores(observation, 2), [13, 12])
        self.assertEqual(score_margin([13, 12], 0), 1)

    def test_summarize_results(self):
        results = [
            EpisodeResult("main.py", 1, 42, 0, ["random"], ["main.py", "random"], "win", [1, -1], ["DONE", "DONE"], [100, 10], 2.0, 90, 0.1, 501),
            EpisodeResult("main.py", 2, 43, 0, ["random"], ["main.py", "random"], "loss", [-1, 1], ["DONE", "DONE"], [10, 100], -2.0, -90, 0.2, 501),
        ]

        summary = summarize_results(results)

        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertEqual(summary["win_rate"], 0.5)
        self.assertEqual(summary["avg_margin"], 0.0)
        self.assertEqual(summary["avg_score_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
