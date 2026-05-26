import math
import unittest
from types import SimpleNamespace

from kaggle_environments.envs.orbit_wars.orbit_wars import interpreter

from orbit_fast.sim import GameState


def _official_state(
    planets,
    fleets,
    actions=None,
    step=1,
    num_agents=2,
    angular_velocity=0.01,
    next_fleet_id=100,
    comets=None,
    comet_planet_ids=None,
):
    if actions is None:
        actions = [[] for _ in range(num_agents)]
    comets = [] if comets is None else comets
    comet_planet_ids = [] if comet_planet_ids is None else comet_planet_ids
    state = [
        SimpleNamespace(
            observation=SimpleNamespace(
                step=step,
                planets=[p[:] for p in planets],
                fleets=[f[:] for f in fleets],
                next_fleet_id=next_fleet_id,
                angular_velocity=angular_velocity,
                initial_planets=[p[:] for p in planets],
                comets=comets,
                comet_planet_ids=comet_planet_ids[:],
            ),
            action=actions[0],
            status="ACTIVE",
            reward=0,
        )
    ]
    for player in range(1, num_agents):
        state.append(
            SimpleNamespace(
                observation=SimpleNamespace(player=player),
                action=actions[player],
                status="ACTIVE",
                reward=0,
            )
        )
    return state


def _env(seed=None, ship_speed=6, episode_steps=500, comet_speed=4):
    return SimpleNamespace(
        configuration=SimpleNamespace(
            seed=seed,
            shipSpeed=ship_speed,
            episodeSteps=episode_steps,
            cometSpeed=comet_speed,
        ),
        done=False,
        info={} if seed is None else {"seed": seed},
    )


def _advance_official_step(state):
    next_step = getattr(state[0].observation, "step", 0) + 1
    for agent_state in state:
        agent_state.observation.step = next_step


class TestFastSimParity(unittest.TestCase):
    def assert_close(self, got, expected, path="value"):
        if isinstance(expected, float) or isinstance(got, float):
            self.assertAlmostEqual(got, expected, places=9, msg=path)
        elif isinstance(expected, list):
            self.assertIsInstance(got, list, path)
            self.assertEqual(len(got), len(expected), path)
            for i, (g_item, e_item) in enumerate(zip(got, expected)):
                self.assert_close(g_item, e_item, f"{path}[{i}]")
        elif isinstance(expected, dict):
            self.assertIsInstance(got, dict, path)
            self.assertEqual(set(got), set(expected), path)
            for key in expected:
                self.assert_close(got[key], expected[key], f"{path}.{key}")
        else:
            self.assertEqual(got, expected, path)

    def assert_state_matches(self, fast, official):
        obs = official[0].observation
        self.assert_close(fast.planets, obs.planets, "planets")
        self.assert_close(fast.fleets, obs.fleets, "fleets")
        self.assert_close(fast.initial_planets, obs.initial_planets, "initial_planets")
        self.assert_close(fast.comets, obs.comets, "comets")
        self.assert_close(fast.comet_planet_ids, obs.comet_planet_ids, "comet_planet_ids")
        self.assertEqual(fast.next_fleet_id, obs.next_fleet_id)
        self.assertEqual(fast.rewards, [s.reward for s in official])
        self.assertEqual(fast.status, [s.status for s in official])

    def run_one_step(self, planets, fleets, actions=None, step=1, num_agents=2):
        official = _official_state(
            planets,
            fleets,
            actions=actions,
            step=step,
            num_agents=num_agents,
        )
        env = _env()
        official = interpreter(official, env)

        fast = GameState(
            planets=[p[:] for p in planets],
            fleets=[f[:] for f in fleets],
            initial_planets=[p[:] for p in planets],
            angular_velocity=0.01,
            next_fleet_id=100,
            step_index=step,
            num_agents=num_agents,
        )
        fast.step(actions)
        self.assert_state_matches(fast, official)

    def test_combat_user_example_matches_official(self):
        planets = [[0, -1, 80, 80, 5, 10, 0]]
        fleets = [
            [0, 0, 76.0, 80.0, 0.0, 1, 41],
            [1, 1, 76.0, 80.0, 0.0, 2, 20],
            [2, 1, 76.0, 80.0, 0.0, 2, 20],
            [3, 2, 76.0, 80.0, 0.0, 3, 42],
        ]
        self.run_one_step(planets, fleets, num_agents=4)

    def test_launch_production_and_fleet_motion_match_official(self):
        planets = [
            [0, 0, 80.0, 80.0, 3.0, 50, 2],
            [1, 1, 20.0, 20.0, 3.0, 50, 2],
        ]
        actions = [[[0, math.pi, 12]], []]
        self.run_one_step(planets, [], actions=actions)

    def test_planet_first_collision_order_matches_official(self):
        planets = [[0, 1, 98.0, 50.0, 2.0, 50, 1]]
        fleets = [[0, 0, 95.0, 50.0, 0.0, 99, 1000]]
        self.run_one_step(planets, fleets)

    def test_sun_removal_matches_official(self):
        planets = [[0, 0, 80.0, 50.0, 3.0, 50, 1]]
        fleets = [[0, 0, 60.0, 50.0, math.pi, 0, 10]]
        self.run_one_step(planets, fleets)

    def test_seeded_empty_rollout_matches_official_through_comet_spawn(self):
        seed = 123456
        official = [
            SimpleNamespace(
                observation=SimpleNamespace(step=0),
                action=[],
                status="ACTIVE",
                reward=0,
            )
        ] + [
            SimpleNamespace(
                observation=SimpleNamespace(player=i),
                action=[],
                status="ACTIVE",
                reward=0,
            )
            for i in range(1, 4)
        ]
        env = _env(seed=seed, episode_steps=120)
        official = interpreter(official, env)
        fast = GameState.initialize(num_agents=4, seed=seed, episode_steps=120)
        self.assert_state_matches(fast, official)

        for _ in range(60):
            for agent_state in official:
                agent_state.action = []
            official = interpreter(official, env)
            fast.step([[], [], [], []])
            self.assert_state_matches(fast, official)
            _advance_official_step(official)
