# Orbit Wars Alpha Research

Last updated: 2026-05-11

Purpose: find the next large strategic opportunity. This file is not a changelog
of constant tweaks. It records experiments, what they prove, what they fail to
prove, and what to build next.

## Current Baseline

Live production baseline: `v6.71 homeprod3 v656 router`.

Known status:

- Kaggle public score check on 2026-05-11 showed `v6.85` stuck at `823.9`,
  while `v6.71` remained much healthier at `940.2`. The older `v6.35`
  safety-anchor rerun remains the single best historical score at `959.4`, but
  it is no longer in the latest-two window.
- Root `main.py` has been rolled back to the saved
  `submissions/v6_71_homeprod3_v656_router_candidate.py` candidate so the
  workspace no longer defaults to the known-bad v6.85 portfolio router.
- v6.35, v6.53, v6.56, and v6.59 remain useful 4P router components. Do not
  promote a new router unless it beats v6.71 on fresh replay guards and active
  2P/4P gates.
- Full Claude v7 should not replace production; it went `3W-0T-5L` on the
  compact guard because it was based on an older v6.13 policy and dropped later
  opening/profile work.
- Replay-suite performance is saturated and cannot be the main optimization
  target anymore.

## Experiment Log

### E11: v6.85 Live Regression And Opening-Failsafe Probe

Context:

- v6.85 completed 49 public games: `19W-30L`, score `823.9`.
- Breakdown: 2P was roughly break-even (`13W-12L`), while 4P collapsed
  (`6W-18L`).
- 4P losses averaged only about `21.3` max production and first capture around
  turn `14.8`; 4P wins averaged about `68.2` max production and first capture
  around turn `8.7`.
- Representative losses show the same shape: delayed or misdirected opening
  capital, then a low-production lockout. The miss-rate hypothesis is not the
  main driver; several losses have low projected miss rates.

Counterfactual checks:

- On the 18 v6.85 4P loss replays, fixed-opponent replay simulation gave:
  - `v6.71`: `12W-6L`
  - `v6.56`: `12W-6L`
  - `v6.35`: `12W-6L`
- This does not pick a clear 4P component, but it does prove v6.85's live 4P
  choice is not worth preserving.

Candidate probes:

- `v6_87_v671_4p_opening_failsafe_candidate.py` added a broad high-production
  neutral opener. It flipped some lockouts but regressed the 18-game guard to
  `11W-7L`, mainly because exact/padded early sends disrupted already-good
  openings.
- `v6_88_v671_4p_capital_unlock_candidate.py` narrowed the idea to hoarded
  65+ ship capital with no base move. It is safer, but too narrow to justify a
  submission yet.

Decision:

- Do not submit v6.87 or v6.88.
- Roll root `main.py` back to v6.71.
- Next alpha should be a direct 4P opening-book/router built from top replay
  opening states, not another generic rule layered over all 4P maps.

### E10: v6.85 Portfolio Router

Context:

- Fresh Kaggle feedback for the latest two submissions collected 139 public
  episodes. v6.71 was `30W-49L`; the 4P pocket was the worst at `14W-35L`.
- Corrected diagnostics again showed this is not mainly an aiming issue:
  v6.71 4P losses averaged about `1.7%` projected miss rate, but only `20.9`
  max production.
- Failed probes: `v6_82` home-prod-2 dense v6.56 routing, `v6_83` lockout
  rescue wrapper, and `v6_84` v6.56 default 4P.

Candidate:

- `submissions/v6_85_v671_2p_v635_4p_portfolio_candidate.py`
- 2P -> v6.71 agent.
- 4P -> v6.35 safety-anchor router.

Validation:

- compile + submission contract: passed.
- active 4P gate, 20 fast games: v6.85 ranked first (`mu 601.6`) over v6.56,
  v6.35, and v6.71.
- confirmation 4P gate, 12 fast games: v6.85 ranked first again (`mu 636.5`).
- direct 2P vs v6.71, 20 fast games: v6.85 led `12-2-6`, average margin
  `+224.9`.
- recent v6.71 4P loss replay guard, 10 seats: `7W-3L`.

Decision:

- Promoted v6.85 to root `main.py`.
- Submitted to Kaggle with message `v6.85 v671 2p v635 4p portfolio`.
- Next action is to monitor the score before spending another submit. If it
  underperforms, the likely problem is local 4P fast-sim bias, not a syntax or
  submission-contract issue.

### E8: Top-10 Replay Sample And Enemy-Pressure Probe

Context:

- Downloaded the public Kaggle dataset
  `bovard/orbit-wars-top10-episodes-2026-05-04`.
- Full extraction filled the workspace, so the partial `episodes/` extraction
  was deleted and analysis now streams from the retained zip.
- Sampled the top 40 manifest episodes, all 4P, and compared top winners with
  our latest v6.71 4P winners.

Result:

| metric | top 40 4P winners | our v6.71 4P winners |
|---|---:|---:|
| avg max prod | 69.9 | 76.3 |
| avg launches | 284.2 | 705.5 |
| avg ships launched | 24,950 | 17,128 |
| avg send | 103.1 | 22.9 |
| median p90 send | 98.6 | 39.0 |
| enemy ship share | 52.1% | 23.4% |
| friendly ship share | 37.9% | 67.4% |

Interpretation:

- Our good games are not losing because the economy cannot grow. In 4P wins,
  our max production is already comparable to top games.
- The bigger gap is conversion: top bots move fewer fleets, but each fleet is
  much larger and over half of ship volume hits enemy planets. Our successful
  games spend about two-thirds of ship volume on friendly transfers.
- A wrapper experiment (`v6_75_enemy_pressure_router_candidate.py`) appended
  late enemy packets to v6.71. It flipped one fixed-trace midgame conversion
  replay but did not beat the baselines in active fast 4P.
- The same pressure layer on v6.35 (`v6_76_v635_late_enemy_pressure_candidate.py`)
  also failed the active fast gate. The idea is directionally right, but the
  wrapper form is too bolted-on and likely sends from the wrong sources/timings.

Next:

- Do not submit v6.75 or v6.76.
- Treat v6.35 as the healthiest 4P base until a candidate beats it active.
- Build conversion inside mission scoring rather than appending moves after the
  base policy. The target is top-player ratios: fewer friendly shuttles, larger
  enemy-bound packets, and enemy ship share closer to 45-55%.

### E7: v6.71 Latest Feedback And Miss-Rate Audit

Context:

- A fresh v6.71 pull collected 65 public episodes: 27 wins and 38 losses.
- The largest loss pocket is 4P. v6.71 went 12W-26L in the collected 4P games.
- Claude suggested a high-miss-rate aiming bug, centered on episode `76227647`.

Result:

| diagnostic | old read | corrected read |
|---|---:|---:|
| `76227647` v6.71 projected miss rate | about 46.7% | 0.0% |
| `76227647` v6.71 launches | 30 | 30 |
| `76227647` winner launches | 751 | 751 |
| `76227647` v6.71 max production | 9 | 9 |
| `76227647` winner max production | 66 | 66 |

Interpretation:

- The old replay miner projected actions against same-step observations and used
  static planet targeting. That falsely marked valid moving-planet shots as
  whiffs.
- `scripts/analyze_replay.py` and `scripts/mine_replay_strategy.py` now project
  launch actions from the previous observation and use the agent's dynamic
  moving-planet projector when available.
- The actual gap is not aiming in these losses. It is 4P economy collapse:
  low production by turn 50, low launch count, and failure to match opponent
  funnel volume.

Alpha signal:

- Stop optimizing against static miss rate.
- Focus on opening profile routing and bad-map salvage in 4P.
- Current experiment: `v6_72_expanded_v656_profile_candidate.py`, which routes
  more v6.71 4P loss profiles through the stronger v6.56 4P component.

### E9: Router Probe Failures After Top-Replay Audit

Context:

- The top-replay sample showed our 4P wins over-funnel friendly ships compared
  with top-10 games, but direct pressure appenders failed active gates.
- Latest Kaggle score check on 2026-05-10 still had `v6.71` at `954.6` and
  older `v6.35` at `959.4`, close enough that local gates need to be decisive.

Results:

| candidate | change | active result |
|---|---|---|
| `v6_77_4p_mature_conversion_candidate.py` | mature-game pressure packets and lower rear funneling | failed 4P gate, ranked last |
| `v6_78_4p_fast_prod3_opening_candidate.py` | base 4P planner allows early cheap prod-3 captures | fixed one replay opening, failed 4P gate |
| `v6_79_barren_fast_prod3_router_candidate.py` | v6.71 barren router sends cheap prod-3 before fixed 21-packet | improved one loss shape, failed 4P gate |
| `v6_80_no_barren_packet_router_candidate.py` | disable v6.71 barren override | failed 4P gate |
| `v6_81_homeprod3_sparse_v517_router_candidate.py` | route sparse home-prod-3 maps to embedded v5.17 | overfit portfolio seeds, failed 4P gate |

Portfolio check:

- 4P, 16 fast games, strongest components: `v6.71` ranked first over
  `v6.56`, `v6.35`, and `v5.17`.
- 2P, 16 fast games: `v6.71` was clearly strongest (`4-1-0` in sampled
  pairings), so a full rollback to `v6.35` would likely hurt 2P.

Decision:

- Do not submit v6.77-v6.81.
- Keep production on `v6.71` unless a candidate beats it in both 2P and 4P or
  gives a very clear 4P gain without touching 2P.
- The next real alpha should be a higher-quality local evaluator/portfolio
  router trained on many saved top replay/opening features, not another
  single-feature route rule.

### E6: Kaggle-Safe Architecture Cache Pass

Context:

- qihuazhong's Kore-2022 architecture suggests timeline arrays, cached world
  properties, defense-first dispatch, and precomputed routes.
- Orbit Wars submission code must remain standalone and standard-library only,
  so we tested pure-Python adaptations.

Result:

| candidate | result | status |
|---|---|---|
| `v5_10_projected_cache_experiment.py` | 1-1 vs v5 in a tiny duel, but very slow (`62.6s` avg backtest duration) | discarded |
| `v5_11_microcache_candidate.py` | stdlib-safe `fleet_speed` cache plus timeline loop cleanup; random smoke passed | kept as architecture candidate |
| `v5_12_microcache_defense_priority_candidate.py` | rescue/reinforce/recapture execute before normal attacks; random smoke passed | kept as architecture candidate |
| `v5_13_route_position_cache_candidate.py` | exact per-turn target-position/arrival cache; profile calls dropped from ~51M to ~45M on one random run | kept as research candidate, not submit-ready |
| `v5_14_2p_routecache_4p_baseline_hybrid_candidate.py` | gates v5.13 route/defense changes to 2P and leaves 4P closer to v5.0 | saved, validation was too slow/incomplete |
| `v5_15_2p_partial_floor_candidate.py` | v5.0 plus v5.9's partial-source floor in 2P only | current candidate; strong single full 2P sanity and full 4P sanity |
| `v5_16_v515_microcache_candidate.py` | v5.15 plus stdlib-safe speed/timeline microcache | tests pass; sampled 2P mixed, sampled 4P near v5.0 |
| `v5_17_opening_static_focus_candidate.py` | v5.16 plus narrow 2P opening filters for enemy-favored static neutrals and low-prod second waves | flipped the v6.23 starvation seed; 2P random gate improved to 4-1 |
| `v5_18_game_mode_split_candidate.py` | v5.0 timing/4P behavior plus stable initial game-format detection and v5.17 2P gates | current root; 2P starvation seed stays green, paired 4P smoke matches v5.0 |

Interpretation:

- Timeline simulation is not currently the main bottleneck.
- Profiling points to route aiming:
  `aim_with_prediction -> search_safe_intercept -> estimate_arrival`.
- Exact route/position caching helps, but the candidate needs ladder-like
  validation before promotion because runtime and score movement are noisy.
- v5.13 showed a strong 2P gate but a 4P regression versus v5.0, so root was
  restored to recovered v5.0 after saving v5.14.
- v5.15 is the cleaner lesson from v5.9: keep the live-best v5 4P behavior, but
  lower the 2P partial-source floor from 6 to 4.
- v5.16 ports the safest part of the architecture-cache work onto v5.15: cache
  `fleet_speed` and use indexed per-turn arrival buckets inside timeline
  simulation. This is mainly an execution-budget improvement, not new strategy.
- v5.17 addresses the real 2P loss mode from seed `189001`: after taking a
  cheap prod-2 neutral, the bot sent ships to an enemy-favored static prod-5 and
  kept nibbling prod-2 targets while a safe high-production target was available.
  The fix is intentionally opening-only so 4P and mature 2P behavior stay close
  to v5.16.
- v5.18 makes the split explicit: game format is derived from `initial_planets`
  and remains stable after eliminations. It also removes the microcache pass so
  4P timing is closer to v5.0, while retaining the 2P-only opening filters.

Next:

- Expand-test `v5_17` against recovered v5.0, v5.9, and recent ladder loss maps.
- Submit only if the user explicitly wants to spend a Kaggle slot; current
  evidence is promising but still sample-small.
- If route caching holds, explore conservative ship-count bucketing for aim
  probes only; do not bucket final committed shots.

### E1: Replay Feature Mining On v6.13 Peak Feedback

Command:

```bash
.venv/bin/python scripts/aggregate_replay_alpha.py \
  backtests/v613_peak_feedback/replays/submission_52335247/*replay.json \
  --markdown backtests/alpha_research_20260505/v613_peak_replay_alpha.md
```

Output:

- `backtests/alpha_research_20260505/v613_peak_replay_alpha.md`

Result:

| group | seats | avg max prod | avg launches | avg ships | avg send | enemy ship share | friendly ship share | miss ship share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| winners | 14 | 72.3 | 507.4 | 11,352 | 21.4 | 22.1% | 42.4% | 24.6% |
| non-winners | 20 | 25.1 | 179.6 | 3,852 | 20.3 | 41.3% | 18.9% | 19.2% |
| 4P winners | 3 | 72.0 | 551.0 | 10,118 | 18.7 | 31.3% | 37.5% | 24.6% |
| 2P winners | 11 | 72.4 | 495.5 | 11,689 | 22.1 | 19.6% | 43.8% | 24.6% |

Interpretation:

- The strongest wins are economy-volume wins, not just target-score wins.
- Winners reach roughly `70+` max production and then move huge ship volume.
- Friendly funneling is not optional; winners route about `40%+` of ship volume
  through owned planets.
- Non-winners often attack more directly but never build the production base
  needed to sustain pressure.

Alpha signal:

- The next big edge is likely reliable production growth plus funnel volume, not
  a prettier one-shot attack score.

### E2: Replay Guard Saturation

Commands:

```bash
.venv/bin/python scripts/evaluate_replay_suite.py \
  backtests/v613_peak_feedback/replays/submission_52335247/*replay.json \
  --agent main.py \
  --team orf527 \
  --json backtests/alpha_research_20260505/v621_all_v613_replays.json

.venv/bin/python scripts/evaluate_replay_suite.py \
  backtests/v613_peak_feedback/replays/submission_52335247/*replay.json \
  --agent submissions/v6_20_2p_4p_profile_split_candidate.py \
  --team orf527 \
  --json backtests/alpha_research_20260505/v620_all_v613_replays.json
```

Result:

| candidate | replay result | note |
|---|---:|---|
| v6.21 current `main.py` | 14W-0T-0L | passes all known v6.13 feedback maps |
| v6.20 snapshot | 14W-0T-0L | identical replay-suite result |

Interpretation:

- The replay suite is now a regression guard, not an alpha detector.
- Any candidate that loses here is probably unsafe.
- Any candidate that wins here is not necessarily better.

Alpha signal:

- Stop using replay win rate as the primary objective. Use it as a hard filter,
  then rank candidates by active local leagues and style-bucket performance.

### E3: Same-Band Opponent Style Analysis

Command:

```bash
.venv/bin/python scripts/analyze_elo_competitors.py \
  backtests/kaggle_feedback_heartbeat_20260504_1900/replays \
  backtests/v613_peak_feedback/replays \
  --leaderboard backtests/alpha_research_20260505/orbit-wars-publicleaderboard-2026-05-03T19:28:22.csv \
  --our-team orf527 \
  --score-low 700 \
  --score-high 1100 \
  --markdown backtests/alpha_research_20260505/elo_700_1100_opponent_styles.md
```

Output:

- `backtests/alpha_research_20260505/elo_700_1100_opponent_styles.md`

Result, grouped by opponent style. W-L-T is from the opponent perspective:

| style | teams | opponent W-L-T | what it means for us |
|---|---:|---:|---|
| balanced | 90 | 36-70-0 | we usually beat these |
| reckless pressure | 43 | 0-48-0 | we crush these |
| noisy routes | 27 | 3-25-0 | we usually beat these |
| heavy pressure | 20 | 7-15-0 | mixed but favorable |
| passive funnel | 14 | 13-4-0 | they beat us often |
| slow opener | 12 | 2-12-0 | we usually beat these |
| controlled funnel | 4 | 5-1-0 | they beat us hard |

Interpretation:

- The loss pattern is very specific: funnel bots beat us.
- We already handle reckless/noisy/balanced opponents much better than expected.
- The highest-leverage target is anti-funnel disruption and matching their
  funnel volume, not more anti-rush defense.

Alpha signal:

- Build a behavior-driven style switcher, with the first real counter being
  anti-passive-funnel / anti-controlled-funnel.

### E4: Active Local Mini-Leagues

Source:

- `backtests/alpha_research_20260505/v621_2p_mini_league.json`
- `backtests/alpha_research_20260505/v621_4p_mini_league.json`

2P result:

| rank | agent | mu | games | W-T-L | avg score margin |
|---:|---|---:|---:|---:|---:|
| 1 | main | 657.6 | 8 | 6-0-2 | 2359.6 |
| 2 | v6.20 profile split | 600.8 | 5 | 2-0-3 | -1351.0 |
| 3 | v5.9 partial volume | 592.5 | 7 | 3-0-4 | -918.0 |
| 4 | v6.13 starved unlock | 466.0 | 4 | 1-0-3 | -1424.0 |

4P result:

| rank | agent | mu | games | W-T-L | avg score margin |
|---:|---|---:|---:|---:|---:|
| 1 | main | 682.1 | 8 | 2-0-6 | -3738.8 |
| 2 | v5.9 partial volume | 673.8 | 8 | 4-0-4 | -1754.2 |
| 3 | v6.13 starved unlock | 582.5 | 8 | 2-0-6 | -660.9 |
| 4 | v6.20 profile split | 530.9 | 8 | 0-0-8 | -6148.6 |

Interpretation:

- v6.21 is a reasonable current baseline.
- The 4P W-L labels are misleading because only one bot can win a game; use mu
  and head-to-head rating movement more than raw win count.
- The sample is too small for submission decisions, but it supports keeping
  v6.21 as the active branch.

Alpha signal:

- The next experiment should be style-aware and tested against style buckets,
  not only against random local snapshots.

### E5: v6.22 Anti-Funnel Prototype

Implementation:

- Added active enemy-fleet funnel detection.
- Added high-priority anti-funnel capture missions.
- Added a rear-forwarding volume bump when a production-threatening funnel
  opponent is detected.
- First version triggered in 2P and 4P. Second version required the funneling
  opponent to be a production threat. Final experiment gated anti-funnel to 4P
  only and was saved as `submissions/v6_22_4p_anti_funnel_experiment.py`.

Outputs:

- `backtests/alpha_research_20260505/v622_anti_funnel_compact_guard.json`
- `backtests/alpha_research_20260505/v622_anti_funnel_full_guard.json`
- `backtests/alpha_research_20260505/v622_strict_vs_v621_2p_league.json`
- `backtests/alpha_research_20260505/v622_strict_vs_v621_4p_league.json`
- `backtests/alpha_research_20260505/v622_4p_only_vs_v621_2p_league.json`
- `backtests/alpha_research_20260505/v622_4p_only_vs_v621_4p_league.json`

Result:

| candidate | test | result |
|---|---|---|
| v6.22 first anti-funnel | compact guard | 8W-0T-0L |
| v6.22 first anti-funnel | full v6.13 guard | 14W-0T-0L |
| v6.22 first anti-funnel | 2P mini-league | trailed v6.21; not safe |
| v6.22 strict prod-threat gate | 2P mini-league | improved but still trailed v6.21 |
| v6.22 strict prod-threat gate | 4P mini-league | beat v6.21 but not v6.20 |
| v6.22 4P-only gate | compact guard | 8W-0T-0L |
| v6.22 4P-only gate | short 4P mini-league | slightly above v6.21, below v6.20 |

Interpretation:

- The anti-funnel idea is real enough to keep, but not production-ready.
- The first prototype overreacted to harmless 2P friendly transfers.
- 4P-only gating is safer, but current local evidence is not strong enough to
  submit it over v6.21.
- Root `main.py` was restored to v6.21 after saving the experiment snapshot.

Alpha signal:

- Anti-funnel needs a narrower target: disrupt funnel destinations only when
  they are both production-leading and receiving/launching meaningful volume.
  The current prototype is too blunt.

## Ranked Alpha Opportunities

### 1. Anti-Funnel Disruption

Why:

- Passive funnel and controlled funnel opponents are the clearest losing bucket.
- Same-band analysis shows passive funnel opponents went `13-4` against us and
  controlled funnel opponents went `5-1`.
- These bots spend huge ship volume on friendly transfers. That creates a window
  where their rear and frontier planets are both temporarily under-defended.

Candidate behavior:

- During turns `25-120`, estimate each opponent's style from observed launches:
  friendly ship share, p90 send, max production, launch volume, and enemy share.
- If opponent is funneling, switch from normal growth to disruption:
  - send small probes at their frontier production planets every few turns;
  - force them to spend ships defending instead of consolidating;
  - prefer attacks that interrupt a transfer chain, not just highest raw value;
  - pressure the funnel destination before the transferred volume arrives.

Test:

- Build replay/style buckets from `elo_700_1100_opponent_styles.md`.
- Run the candidate against passive/controlled-funnel replays first.
- Candidate must not regress balanced/reckless/noisy buckets.

Expected upside:

- Highest. This targets the style that actually beats us.

### 2. Reliable Self-Funnel Volume

Why:

- Winners average `507 launches`, `11.3k` launched ships, and `42%` friendly
  ship share.
- Non-winners average `180 launches`, `3.9k` launched ships, and only `19%`
  friendly ship share.

Candidate behavior:

- Add an explicit funnel objective, separate from emergency defense:
  - identify one or two frontier sink planets;
  - every rear planet above reserve sends periodic packets toward the sink;
  - scale packet cadence with production lead/deficit;
  - in 4P, avoid funneling into a planet exposed to two enemy fronts.

Test:

- Instrument launch volume by bucket for our bot in local leagues.
- Require candidate to increase friendly transfer volume without delaying first
  high-production neutral capture.

Expected upside:

- Very high. This is the macro pattern of the winning seats.

### 3. Production-Floor Emergency Mode

Why:

- The most important separator is max production: winners average `72`, losers
  average `25`.
- When we fail, we often never reach the economy needed for mature packets.

Candidate behavior:

- Add production milestones:
  - by turn `35`, target production >= map-adjusted threshold;
  - by turn `65`, target production >= second threshold.
- If below milestone:
  - temporarily override conservative reserves;
  - prefer guaranteed high-production neutrals over enemy pressure;
  - allow multi-source captures of production-4/5 planets even when no single
    source can pay alone.

Test:

- Use replays where original max production stayed under `35`.
- Compare first high-prod neutral timing and max production curve.

Expected upside:

- High, especially for low/mid Elo climb.

### 4. Style Switcher

Why:

- Opponent style has predictive value.
- We beat reckless/noisy/balanced more often than funnel styles, so one global
  strategy is leaving EV on the table.

Candidate behavior:

- Classify opponents after turns `20-30`:
  - passive funnel: high friendly share, low enemy share;
  - controlled funnel: high max production, high friendly share, mature p90;
  - reckless pressure: high enemy share, low max production;
  - noisy routes: high miss share;
  - heavy pressure: high enemy share and high p90.
- Feed the style into target scoring, reserves, pressure packets, and rear
  funnel policy.

Test:

- Offline: compare style label against replay outcomes.
- Online/local: run separate bucket mini-suites.

Expected upside:

- High, but only after anti-funnel and self-funnel policies exist.

### 5. Expensive-Packet Route Validation

Why:

- Winners can have high miss share, so route validation for every launch is too
  blunt.
- Huge early misses are still bad. We should care much more about a 90-ship miss
  than a 3-ship probe miss.

Candidate behavior:

- Keep cheap probes permissive.
- For launches above a size threshold or before turn `50`, retry aim with
  alternate ship sizes and reject only expensive misses.
- Add route validation telemetry to count rejected vs accepted big packets.

Test:

- Compare miss ships, not miss count.
- Candidate should reduce early big-miss ship volume without lowering launch
  count too much.

Expected upside:

- Medium. Useful, but not enough alone for 1200.

### 6. Optuna With Style-Aware Objective

Why:

- Hand tuning is slow and noisy.
- Replay-only tuning will overfit a saturated guard.

Candidate objective:

1. Hard filter: no losses on compact replay guard.
2. Score: active 2P league mu.
3. Score: active 4P league mu.
4. Penalty: passive/controlled-funnel bucket losses.
5. Penalty: production max below milestone.

Test:

```bash
.venv/bin/python scripts/tune_constants.py \
  --trials 30 \
  --games-2p 8 \
  --games-4p 8 \
  --jobs 4 \
  --out-dir backtests/optuna_constants
```

Expected upside:

- Medium to high, but only if the objective includes the style gap.

## Immediate Next Experiments

### 2026-05-06 Mode Split Result

The partial transplant idea was rejected: v6.21 plus the v5.17 2P guards still
lost the targeted 2P seed `189001` against v6.23 by `-6690`, while literal
v5.17 wins it by `+4949`.

Current router candidate:

- `2P -> v5_17_opening_static_focus_candidate.py`
- `4P -> v6_21_profile_split_endgame_refiner_candidate.py`
- snapshot: `submissions/v6_25_mode_router_v517_2p_v621_4p_candidate.py`

Validation:

- unit contract: `26/26 OK`
- targeted 2P seed `189001`: `1-0-0`, score margin `+4949`
- paired 2P vs v5.17, seeds `189000-189001`: identical `1-0-1`
- paired 4P vs v6.21, seeds `194100-194101`: both won; router average margin
  lower but same win/loss result
- active 4P random gate, seed `195100`: failed; router went `0-2` while v6.21
  went `2-3` and v5.0 ranked first on that sample

Decision: do not submit this router as-is. Root `main.py` was restored to the
live-best v5.0 baseline after saving the router snapshot. The next mode-router
attempt should use a fairer 4P branch selection, likely comparing v5.0 and v6.21
with shorter fixed-step paired gates before another full run.

### 2026-05-06 Mode Router v6.26

Candidate:

- `2P -> v5_17_opening_static_focus_candidate.py`
- `4P -> v5_0_moving_ledger_highprod_source_save.py`
- snapshot: `submissions/v6_26_mode_router_v517_2p_v50_4p_candidate.py`

Validation:

- unit contract: `26/26 OK`
- targeted 2P seed `189001`: `1-0-0`, score margin `+4949`
- paired 4P vs v5.0, seeds `195102-195103`: identical `1-0-1`, same average
  score margin `+1660.5`
- compact active 4P gate, seed `197100`: router ranked first, `3-0-1`, average
  margin `+3296.8`
- latest v5.17 loss replay guard, 10 seats:
  - v6.26 router: `4-0-6`
  - v5.0 baseline: `5-0-5`
  - regression: v6.26 lost episode `75955120`, which v5.0 won in the same
    counterfactual guard

Decision: do not submit v6.26. Root `main.py` was restored to the live-best
v5.0 baseline. The split idea improved some local gates, but latest feedback
does not beat the current live-best baseline.

### 2026-05-06 Mode Router v6.27

Candidate:

- default: `v5_0_moving_ledger_highprod_source_save.py`
- `2P -> v5_17_opening_static_focus_candidate.py` only when the opening has a
  neutral planet with `production >= 4` and `ships <= 35`
- `4P -> v5.0`
- snapshot: `submissions/v6_27_gated_v517_cheap_highprod_2p_candidate.py`

Why:

- v5.17 beats v5.0 on seed `189001` because cheap high-production neutrals can
  be raced early.
- v5.17 loses replay `75955120` because high-production neutrals are expensive
  (`54+` ships), causing low-volume trickle and long-term production starvation.

Validation:

- unit contract: `26/26 OK`
- targeted 2P seed `189001`: win, score margin `+4949`
- regression replay `75955120`: win, matches v5.0 outcome
- latest v5.17 loss replay guard, 10 seats: `5-0-5`, tying v5.0 and improving
  over v6.26's `4-0-6`
- paired 4P vs v5.0, seeds `195102-195103`: identical `1-0-1`, same average
  score margin `+1660.5`
- active 2P random gate, seed `198000`: v6.27 ranked first, `2-0-1`

Decision: current root `main.py` is v6.27. It is a real candidate because it
keeps v5.0's latest-loss guard while preserving the known v5.17 2P upside.

0. Recover and protect the exact v5 production baseline, or identify the closest
   saved snapshot if the exact submitted file is unavailable.
1. Implement an opponent style telemetry pass in `main.py` without changing
   behavior. Validate that online classifications match replay analysis.
2. Add anti-funnel disruption behind a gate:
   - only after turn `25`;
   - only against opponents with high friendly-transfer share;
   - only if our production is not already collapsing.
3. Add self-funnel volume instrumentation:
   - launches by target kind;
   - ships by target kind;
   - first high-prod neutral turn;
   - max production by turn bucket.
4. Build a small bucketed test set:
   - passive/control funnel losses;
   - balanced wins;
   - reckless/noisy wins;
   - 4P pressure games.
5. Use replay guard only as a safety check, then rank candidates by bucketed
   active-game tests.

## Submission Rule

Do not submit a candidate just because it wins the replay suite.

Submit only if:

- it passes the compact replay guard;
- it passes the full v6.13 replay guard;
- it beats v6.21 in 2P mini-league;
- it beats or ties v6.21 in 4P mini-league;
- it specifically improves passive/controlled-funnel bucket performance.

### 2026-05-11 Replay Alignment Correction And v6.35 Anchor

Finding:

- Kaggle replay rows store `action` on the row after the action has been
  applied. The legal source state for `steps[t][player].action` is
  `steps[t - 1][player].observation`, not `steps[t][player].observation`.
- The previous legal-action filters undercounted top-player launches and made
  many valid large sends look like invalid oversends.
- `scripts/phase_metrics.py` was corrected to evaluate actions against the
  previous observation. `scripts/compare_policy_to_replays.py` already uses this
  alignment.

Corrected aligned top-player sample, `backtests/top10_sample_120` winners:

- `launches=262.8`, `ships=13053`, `avg_send=50.7`, `p90_send=70.7`
- bucket `t0_49`: `launches=18.5`, `ships=412`, `p90=38.6`
- bucket `t50_99`: `launches=46.6`, `ships=1528`, `p90=59.9`
- bucket `t100_199`: `launches=103.9`, `ships=3933`, `p90=75.6`
- target share: neutral `7.7%`, enemy `38.1%`, friendly `44.4%`, comet `1.6%`

Corrected latest v6.85 feedback for `orf527`:

- 4P wins: first capture `8.7`, max production `68.2`, avg send `32.0`, p90
  send `68.7`
- 4P losses: first capture `14.8`, max production `21.3`, avg send `19.0`,
  p90 send `35.1`
- The real 4P loss signature is early economy lockout: late first capture and
  low max production. Friendly transfers are high in wins because winning games
  have a real economy to move around.

Candidate outcomes:

- `v6.91_v671_4p_fast_patience_candidate`: fixed-trace guard `14-4`, but active
  4P gate `0-8`; rejected.
- `v6.92_v671_4p_friendly_chatter_filter_candidate`: subset guard `7-1`, but
  active 4P gate `0-8`; rejected.
- `v6.93_v671_miss_salvage_candidate`: subset guard `5-3`; rejected.
- `v6.94_v671_4p_enemy_packet_boost_candidate`: subset guard `1-7`; rejected.
- `v6.95_v671_4p_first_capture_breaker_candidate`: subset guard `5-3`, active
  4P gate `0-8`; rejected.

Decision:

- Stop building on v6.71 as the default baseline.
- Restore root `main.py` to exact
  `submissions/v6_35_mode_router_v627_2p_v634_4p_candidate.py`.
- Verified: `py_compile`, `tests/test_submission_contract.py`, and `cmp` all
  pass.
- Do not submit the v6.91-v6.95 candidates.
