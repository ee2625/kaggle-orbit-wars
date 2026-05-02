import unittest

from scripts.evaluate_replay_suite import classify_result, find_player_indices, trace_agent


class EvaluateReplaySuiteTest(unittest.TestCase):
    def test_trace_agent_uses_one_turn_offset(self):
        actions = [[], [["late"]], [["target"]]]
        agent = trace_agent(actions, offset=1)

        self.assertEqual(agent({}), [["late"]])
        self.assertEqual(agent({}), [["target"]])
        self.assertEqual(agent({}), [])

    def test_find_player_indices_by_team(self):
        replay = {"info": {"TeamNames": ["a", "ours", "b", "ours"]}, "steps": [[{}, {}, {}, {}]]}

        self.assertEqual(find_player_indices(replay, team="ours", explicit_index=None), [1, 3])

    def test_classify_result(self):
        self.assertEqual(classify_result([-1, 1], 1), "win")
        self.assertEqual(classify_result([1, 1], 0), "tie")
        self.assertEqual(classify_result([-1, 1], 0), "loss")


if __name__ == "__main__":
    unittest.main()
