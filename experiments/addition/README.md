# Addition

**Can a transformer add in a numeral system it is only told about in the prompt?**

Each problem comes with an **axiom**: an ordered alphabet of digit symbols, e.g. `[q, #, 7, Z]`. The axiom defines both the base (here 4) and what each symbol is worth (`q`=0, `#`=1, ...). Symbols are random bytes, resampled for every problem, so no symbol has a fixed meaning the model could memorise. Models are trained from scratch on a few bases ({2, 10} or {2, 4, 8, 10}) and tested zero-shot on:

- **unseen bases**: 3 and 12;
- **longer operands**: 21–40 digits, against 1–20 in training;
- **familiar symbols in an unfamiliar order**: `0`–`9` permuted, which tests whether the prompt overrides a learned prior.

The full design is in [`spec/`](spec/README.md). Start with [`spec/01_research_goal.md`](spec/01_research_goal.md).

## Where things stand

From the latest results (`results/addition/`, last updated 2026-10-02):

- **In-domain addition is solved** (≥ 99.6% exact match) by several setups: `SM-MH` T1, `SM-Seq` / `SM-SeqRev` T4, `SM-SeqRev` T5 after 500k steps, and `AR` T1–T3. `SM-SeqRev` T7, which has no axiom and infers the alphabet, reaches 87–96%.
- **Transfer to unseen bases is partial and unstable.** Base-3 exact match is mostly 5–35%, with outliers at 81% (`SM-Seq` T4, seed 39032) and 77% (`SM-SeqRev` T7 at 100k steps). Base-12 is mostly ≤ 11%; the best is `AR` T2 at 41%. Seeds of the same configuration differ a lot (`SM-Seq` T4: 81% vs. 61% on Base-3), and longer training does not reliably help (the `SM-SeqRev` T7 run trained to 500k steps is at 6% on Base-3).
- **Length extrapolation is ≤ 7%** everywhere (`AR` T2 is best).
- The fixed-canonical-symbols control (`-NC1Canon`) collapses to 0% on random axioms, as expected.

So the core hypothesis is not supported yet; by [`spec/01`](spec/01_research_goal.md) §5 these are partial outcomes. Run-by-run analysis is in [`notes/`](notes/) (`round1.md`–`round3.md`). Several directories in `results/addition/` are 1-step smoke tests (`step` = 1 in `metrics.jsonl`), and `SM-MH-T1-L6-SwiGLU-Causal-bimodal-seed42` stopped at step 1,000.

## Running

From the repo root:

```bash
experiments/addition/run.sh                       # interactive wizard: sets up the venv, configures a run
experiments/addition/run.sh --paradigm sm_seq_rev --tier 4 --seed 1 --dry-run   # 1-step smoke test
python -m experiments.addition.cli --help         # the underlying CLI (also the Vertex AI entrypoint)
python experiments/addition/scripts/summarize_results.py   # leaderboard over results/addition/
pytest experiments/addition                       # tests
```

The wizard writes each run to `results/addition/<run-name>/`, with `run.sh` for local re-runs and `submit_vertex.sh` for Google Cloud Vertex AI. Run names encode the configuration, e.g. `SM-SeqRev-T5-L6-SwiGLU-Causal-multimodal-seed5329`. The [spec README](spec/README.md#3-run-naming-paradigms-and-tiers) decodes them.

## Layout

```
cli.py         CLI / Vertex AI entrypoint
core/          vocabulary, rule + oracle, prompt builders, data generator, model
harnesses/     state-machine and autoregressive execution drivers
utils/         training runner, evaluation, telemetry, interactive stepper
scripts/       run configurator and results summariser
tests/
spec/          the specification
notes/         analysis write-ups from each round of runs
```

Shared transformer blocks, training utilities and the metrics logger live in the top-level [`yessir/`](../../yessir/) package.
