# Results

One directory per experiment; inside it, one directory per run, named by configuration and seed.

Each run directory may contain `config.json`, `run.py`, `run.sh`, `submit_vertex.sh` (how it was launched), `metrics.jsonl` (one line per evaluation), `summary.json` (final evaluation) and `instances.jsonl` (per-instance telemetry). No model checkpoints are stored.

`addition/` holds the union of the dated snapshots taken between 2026-09-20 and 2026-10-02: the most recent copy of every file for every run. Earlier snapshots are still in git history. Launch scripts from before the yessir restructure import the old `transformadder` package. They are kept as a record; re-run with `experiments/addition/run.sh` and the same flags.
