# Orbit Wars Agent Notes

## Commands

```bash
python -m unittest discover -s tests
python scripts/run_local.py --episodes 3
```

Use Python 3.12 for local evaluation. The default macOS Python 3.9 cannot install
the required Orbit Wars Kaggle environment version.

## Submission Contract

- `main.py` must remain standalone and import only the Python standard library.
- Kaggle loads the last callable in a submitted file, so keep the public
  `agent(obs)` wrapper as the final function in `main.py`.
- The agent must return `[[from_planet_id, angle_in_radians, num_ships], ...]`.
- Keep tests fast; use `scripts/run_local.py` for actual environment matches.

## Strategy Direction

Start by improving tactical reliability before adding broad strategy:

- account for incoming friendly and enemy fleets
- better predict moving planet intercepts
- coordinate multi-source captures
- tune neutral expansion separately from enemy attacks
- parse replay JSONs for ladder diagnostics
