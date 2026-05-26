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
| `v5_0_moving_ledger_highprod_source_save.py` | recovered live-best v5 line; Kaggle public score 899.6 on 2026-05-05 |
| `v5_1_submitted.py` | v5.1 4p reserve unlock high-prod race |
| `v5_2_conservative_4p_tempo.py` | v5.2 conservative 4p tempo gates |
| `v5_3_threshold_volume_candidate.py` | unsubmitted v5.1 + lower small-fleet thresholds candidate |
| `v6_2_opening_gate_no_spam_candidate.py` | unsubmitted v5.1 + 4P high-prod opening gate, no small-fleet spam |
| `v6_3_route_validated_candidate.py` | discarded experiment: route validation inside shot search was too slow |
| `v6_4_route_validated_only_candidate.py` | discarded experiment: route validation only, kept for reference |
| `v6_13_starved_economy_unlock_320_candidate.py` | 4P starved-economy exact reserve unlock |
| `v6_19_highprod_opening_gated_wave_candidate.py` | submitted v6.19 high-prod opening + gated wave |
| `v6_20_2p_4p_profile_split_candidate.py` | candidate: separate 2P/4P profiles |
| `v6_21_profile_split_endgame_refiner_candidate.py` | candidate: v6.20 + gated endgame refiner |
| `v6_22_4p_anti_funnel_experiment.py` | experiment: 4P-only anti-funnel disruption, not promoted |
| `v6_23_inline_optuna_trial0002_candidate.py` | candidate: inline Optuna constants, no wrapper/import penalty |
| `v6_24_economy_bias_experiment.py` | discarded experiment: live economy-search/bias hurt validation |
| `v5_10_projected_cache_experiment.py` | discarded experiment: projected timeline cache was not a clean speed win |
| `v5_11_microcache_candidate.py` | candidate: v5 live baseline plus stdlib-safe fleet-speed cache and timeline loop cleanup |
| `v5_12_microcache_defense_priority_candidate.py` | candidate: v5.11 plus defense-first mission execution priority |
| `v5_13_route_position_cache_candidate.py` | candidate: v5.12 plus exact per-turn route position/arrival cache |
| `v5_14_2p_routecache_4p_baseline_hybrid_candidate.py` | experiment: v5.13 route/defense changes gated to 2P, 4P reverts to v5-style aiming/order; not promoted |
| `v5_15_2p_partial_floor_candidate.py` | current candidate: v5.0 plus v5.9's 2P partial-source floor, 4P kept baseline-like |
| `v5_16_v515_microcache_candidate.py` | candidate: v5.15 plus stdlib-safe fleet-speed cache and timeline loop cleanup |
| `v5_17_opening_static_focus_candidate.py` | candidate: v5.16 plus 2P opening gates against enemy-favored static neutrals and low-prod second-wave nibbling |
| `v5_18_game_mode_split_candidate.py` | candidate: v5.0 timing/4P behavior plus stable game-format detection and v5.17 2P-only opening gates |
| `v5_19_v517_stable_mode_candidate.py` | candidate: exact v5.17 behavior plus stable initial game-format detection |
| `v6_25_mode_router_v517_2p_v621_4p_candidate.py` | experiment: standalone literal router; 2P dispatches to v5.17, 4P dispatches to v6.21; not promoted after active 4P gate failed |
| `v6_26_mode_router_v517_2p_v50_4p_candidate.py` | experiment: standalone literal router; 2P dispatches to v5.17, 4P dispatches to live-best v5.0; not promoted after latest-loss guard trailed v5.0 |
| `v6_27_gated_v517_cheap_highprod_2p_candidate.py` | current candidate: v5.0 default; 2P dispatches to v5.17 only when a cheap prod-4+ neutral exists |

When a new bot is submitted, copy the exact submitted `main.py` here with a
versioned filename before continuing experiments.
