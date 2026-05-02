# Reference Files

Raw uploaded reference material lives under `references/raw/` and is ignored by
git. This keeps the active bot submission clean while preserving local material
for strategy research.

## Layout

- `raw/official-starter/orbit-wars/` - official Kaggle starter kit files.
- `raw/notebooks/getting-started.ipynb` - large getting-started notebook.
- `raw/notebooks/orbit-wars-agent-ow-proto.ipynb` - prototype strategy notebook.
- `raw/downloads/` - downloaded zip, logs, and helper text.
- `raw/duplicates/` - duplicate README/main files from repeated downloads.

## Notes

The active submission entrypoint remains `main.py` at the repo root. Keep raw
notebooks, downloaded archives, generated logs, and duplicated starter files out
of the root so Kaggle submissions and local tests stay predictable.
