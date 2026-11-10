# TransformAdder (prototype)

> **Reference only. Allowed to rot.** This prototype was written quickly, mostly by Claude, as an exploration of whether the idea was worth pursuing, and it was not reviewed line by line. It was last made runnable on 2026-11-10 and is not maintained. The current addition experiment is in [`experiments/addition/`](../../experiments/addition/).

**Question:** can a transformer add in a numeral system it is only told about in the prompt?

Each problem comes with an **axiom**: an ordered alphabet of digit symbols, e.g. `[q, #, 7, Z]`. The axiom defines both the base (here 4) and what each symbol is worth (`q`=0, `#`=1, ...). Symbols are random bytes, resampled for every problem, so no symbol has a fixed meaning the model could memorise. Models were trained from scratch on a few bases ({2, 10} or {2, 4, 8, 10}) and tested zero-shot on:

- **unseen bases**: 3 and 12;
- **longer operands**: 21–40 digits, against 1–20 in training;
- **familiar symbols in an unfamiliar order**: `0`–`9` permuted.

[`spec/`](spec/README.md) records what was being tested. [`notes/`](notes/) has the analysis write-ups from each round of runs (`round1.md`–`round3.md`).

## What it found

Final results (runs up to 2026-10-02):

- **In-domain addition was solved** (≥ 99.6% exact match) by several setups: `SM-MH` T1, `SM-Seq` / `SM-SeqRev` T4, `SM-SeqRev` T5 after 500k steps, and `AR` T1–T3. `SM-SeqRev` T7, which has no axiom and infers the alphabet, reached 87–96%.
- **Transfer to unseen bases was partial and unstable.** Base-3 exact match was mostly 5–35%, with outliers at 81% (`SM-Seq` T4, seed 39032) and 77% (`SM-SeqRev` T7 at 100k steps). Base-12 was mostly ≤ 11%; the best was `AR` T2 at 41%. Seeds of the same configuration differed a lot (`SM-Seq` T4: 81% vs. 61% on Base-3), and longer training did not reliably help (the `SM-SeqRev` T7 run trained to 500k steps was at 6% on Base-3).
- **Length extrapolation was ≤ 7%** everywhere (`AR` T2 was best).
- The fixed-canonical-symbols control (`-NC1Canon`) collapsed to 0% on random axioms, as expected.

By the prototype's own criteria ([`spec/01`](spec/01_research_goal.md) §5) these are partial outcomes: the core hypothesis was not supported. These numbers come from unreviewed code and haven't been checked independently, so treat them as leads, not findings.

## Results

[`results/`](results/) has one directory per run, named by configuration and seed. The [spec README](spec/README.md#3-run-naming-paradigms-and-tiers) decodes the names, e.g. `SM-SeqRev-T5-L6-SwiGLU-Causal-multimodal-seed5329`. Each directory may contain `config.json`, `run.py`, `run.sh`, `submit_vertex.sh` (how it was launched), `metrics.jsonl` (one line per evaluation), `summary.json` (final evaluation) and `instances.jsonl` (per-instance telemetry). No checkpoints were saved.

They are the union of the dated snapshots taken between 2026-09-20 and 2026-10-02: the most recent copy of every file for every run. Earlier snapshots are in git history (before commit `b35c755`). Caveats:
- launch scripts from before the restructure import the old top-level `transformadder` package; re-run with `run.sh` below and the same flags instead (they also look for the old `.venv-<os>` environments);
- several directories are 1-step smoke tests (`step` = 1 in `metrics.jsonl`);
- `SM-MH-T1-L6-SwiGLU-Causal-bimodal-seed42` stopped at step 1,000.

## Running (as of 2026-11-10)

From the repo root:

```bash
prototypes/transformadder/run.sh                       # interactive wizard: configures and launches a run
prototypes/transformadder/run.sh --paradigm sm_seq_rev --tier 4 --seed 1 --dry-run   # 1-step smoke test
uv run python -m prototypes.transformadder.cli --help  # the underlying CLI (also the Vertex AI entrypoint)
uv run python prototypes/transformadder/scripts/summarize_results.py   # leaderboard over results/
uv run pytest prototypes/transformadder                # tests (not part of the repo's default test run)
```

New runs are written to `prototypes/transformadder/results/<run-name>/`, with `run.sh` for local re-runs and `submit_vertex.sh` for Google Cloud Vertex AI.

## Layout

```
cli.py         CLI / Vertex AI entrypoint
core/          vocabulary, rule + oracle, prompt builders, data generator, transformer blocks, model
harnesses/     state-machine and autoregressive execution drivers
utils/         training runner, evaluation, telemetry, metrics logger, interactive stepper
scripts/       run configurator and results summariser
tests/
spec/          what the prototype tested (record)
notes/         analysis write-ups from each round of runs
results/       every run
```
