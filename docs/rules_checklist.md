# Orbit Wars Rules Checklist

This is an engineering checklist distilled from the competition rules the team
provided. It is not legal advice; use the Kaggle rules page as the source of
truth if anything changes.

## Submission Limits

- Use one Kaggle account only.
- Maximum team size is 5.
- Maximum submissions: 5 per day.
- Up to 2 final submissions may be selected for judging.
- Final ranking uses the simulation leaderboard; there is no private leaderboard
  for this competition type.

## Submission Runtime

- During an evaluated episode, `main.py` may not pull information from outside
  the submission and environment.
- During an evaluated episode, `main.py` may not send information out.
- Keep `main.py` standalone and deterministic from the observation plus local
  constants.
- Do not depend on files that are not packaged into the submitted artifact.
- Avoid network, filesystem state, credentials, web APIs, environment variables,
  wall-clock time, or randomness that is not explicitly controlled.

## Code And Data Sharing

- Do not privately share competition code outside the official team.
- Public sharing is allowed only when made available to all competitors through
  Kaggle competition forums/notebooks and under an OSI-approved license.
- Open-source dependencies must allow commercial use.
- External data/tools must be publicly available and equally accessible at no
  cost, or otherwise reasonably accessible under the rules.

## Winner Obligations

- Winning submission and source code must be licensable under CC-BY 4.0.
- Competition data is available under Apache 2.0 terms.
- Winners may need to publish a reproducible method description and repository.
- Keep enough notes, seeds, scripts, and result artifacts to explain how final
  submissions were produced.

## Practical Guardrails For This Repo

- The only file submitted by default is root `main.py`.
- Raw reference files stay in ignored `references/raw/`.
- Backtest outputs stay in ignored `backtests/`.
- Do not copy private code from other teams into this repo.
- Keep any non-standard dependency out of `main.py` unless we intentionally move
  to a bundled multi-file submission and can justify the license/accessibility.
