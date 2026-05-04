# Submission Snapshots

This folder keeps stable copies of submitted or submission-candidate agents.
Kaggle submissions still use the root `main.py`; these files are for rollback,
diffing, and local league comparisons.

| file | Kaggle message / role |
| --- | --- |
| `v1_target_scoring_safe_reserve.py` | v1 target scoring safe reserve |
| `v2_moving_target_intercepts.py` | v2 moving target intercepts |
| `v3_anti_trickle_replay_suite.py` | v3 anti-trickle replay-suite 13-0 |
| `v4_1_leader_pressure_padding.py` | v4.1 leader pressure padding replay 49-0 |
| `v5_2_conservative_4p_tempo.py` | v5.2 conservative 4p tempo gates |

When a new bot is submitted, copy the exact submitted `main.py` here with a
versioned filename before continuing experiments.
