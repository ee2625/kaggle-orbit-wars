import unittest

from scripts.league_backtest import (
    Rating,
    build_competitors,
    build_matchups,
    rank_scores,
    rate_draw,
    rate_win,
    update_pairwise_ratings,
)


class LeagueBacktestTest(unittest.TestCase):
    def test_rate_win_moves_winner_up_and_loser_down(self):
        winner = Rating(600.0, 200.0)
        loser = Rating(600.0, 200.0)

        updated_winner, updated_loser = rate_win(winner, loser, beta=100.0, tau=0.0)

        self.assertGreater(updated_winner.mu, winner.mu)
        self.assertLess(updated_loser.mu, loser.mu)
        self.assertLess(updated_winner.sigma, winner.sigma)
        self.assertLess(updated_loser.sigma, loser.sigma)

    def test_rate_draw_moves_ratings_toward_each_other(self):
        favorite = Rating(800.0, 120.0)
        underdog = Rating(600.0, 120.0)

        updated_favorite, updated_underdog = rate_draw(favorite, underdog, beta=100.0, tau=0.0)

        self.assertLess(updated_favorite.mu, favorite.mu)
        self.assertGreater(updated_underdog.mu, underdog.mu)
        self.assertLess(updated_favorite.mu - updated_underdog.mu, favorite.mu - underdog.mu)

    def test_update_pairwise_ratings_ignores_score_margin(self):
        tight = {"a": Rating(600.0, 200.0), "b": Rating(600.0, 200.0)}
        blowout = {"a": Rating(600.0, 200.0), "b": Rating(600.0, 200.0)}

        update_pairwise_ratings(["a", "b"], [101, 100], tight, beta=100.0, tau=0.0)
        update_pairwise_ratings(["a", "b"], [1000, 0], blowout, beta=100.0, tau=0.0)

        self.assertAlmostEqual(tight["a"].mu, blowout["a"].mu)
        self.assertAlmostEqual(tight["b"].mu, blowout["b"].mu)
        self.assertAlmostEqual(tight["a"].sigma, blowout["a"].sigma)
        self.assertAlmostEqual(tight["b"].sigma, blowout["b"].sigma)

    def test_rank_scores_handles_ties(self):
        self.assertEqual(rank_scores([10, 20, 20, 5]), [2, 1, 1, 3])

    def test_build_competitors_makes_duplicate_labels_unique(self):
        competitors = build_competitors(["random", "random"])

        self.assertEqual([competitor.label for competitor in competitors], ["random", "random#2"])

    def test_build_matchups_rejects_too_small_pool(self):
        with self.assertRaises(ValueError):
            build_matchups(build_competitors(["main.py", "random"]), players=4)


if __name__ == "__main__":
    unittest.main()
