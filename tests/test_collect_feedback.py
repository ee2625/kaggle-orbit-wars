import unittest

from scripts.collect_feedback import matching_agent_indices


class CollectFeedbackTest(unittest.TestCase):
    def test_matches_submission_id(self):
        episode = {
            "agents": [
                {"submissionId": 111, "index": 0, "teamName": "other"},
                {"submissionId": 222, "index": 1, "teamName": "ours"},
            ]
        }

        self.assertEqual(matching_agent_indices(episode, submission_id=222, team=None), [1])

    def test_team_name_can_match_self_play_or_unknown_submission(self):
        episode = {
            "agents": [
                {"submissionId": 111, "index": 0, "teamName": "ours"},
                {"submissionId": 222, "index": 1, "teamName": "ours"},
            ]
        }

        self.assertEqual(matching_agent_indices(episode, submission_id=333, team="ours"), [0, 1])


if __name__ == "__main__":
    unittest.main()
