# Addition: evaluation only happens once, at the end of training

Open issue found in review on 2026-11-16. Nothing here is implemented or decided.

## What happens now

`train_and_evaluate` in `experiments/addition/train_and_evaluate.py` makes one `train(...)` call for all `config.max_steps` steps, then loops over `config.eval_splits()` once and writes `eval.json` and `eval_splits.json`. During training, `train.jsonl` records only `step`, `elapsed_sec` and `loss`.

So a run gives one accuracy reading per split, taken at `max_steps`.

## Why it matters

Two leads in [`addition.md`](addition.md) (item 2) are about how accuracy moves over training steps, and neither can be seen from a single final reading:

- **Late grokking**: some setups sat at `0%` in-distribution exact match for 70k–120k steps and then reached `96%`–`100%` by 425k–500k. A run stopped at 100k reads as a failure.
- **OOD overtraining**: on runs that converged early, exact match on unseen bases peaked around 75k–100k steps and then decayed as training continued. A run evaluated only at 500k reports the decayed value and hides the peak.

`studies/addition/grokking-and-overtraining.json` (status `draft`) asks exactly this: how in-distribution exact match and OOD-base transfer evolve "across checkpoints from 25k to 500k steps". With the current loop that study needs one full run per checkpoint step count, and those runs would not share a training trajectory unless everything else is held identical.

It also affects every other study: comparing two variants at a fixed `max_steps` can compare one before its peak with one after it.

## What already exists to build on

- `yessir.training.train` accepts an existing `optimizer` and a persistent iterator (`stream = iter(dataset)`), and its docstring says this is for resuming "between evaluations" without restarting the data stream or resetting Adam state.
- `train_step` calls `model.train()` on every step, so an evaluator leaving the model in `eval()` mode is harmless.
- Evaluation is cheap relative to training: with the defaults it is 200 problems per split, every column of each, in batches of 64.

## Options

1. **Periodic evaluation inside one run.** Add an `eval_every` setting (or `None` to keep today's behaviour). Train in chunks of `eval_every` steps that share one optimizer and one data stream, run all eval splits after each chunk, and append one record per evaluation to a new file (for example `eval_history.jsonl`, with `step` plus the same keys as `eval.json`). The final evaluation still writes `eval.json` and `eval_splits.json` as now.
2. **Checkpoints, evaluated afterwards.** Save model weights every N steps and evaluate them in a separate pass. More flexible (new splits can be scored later) but adds weight files to `results/` and a second entry point.
3. **Leave the loop alone** and run one training run per step count. No code change, but costs a full run per point and gives no shared trajectory.

Option 1 is the smallest change that answers the study's question.

## Open questions for Louis

- Should each evaluation use the same eval problems (same split seeds) every time? Reusing them makes the curve comparable from step to step; it is what the split seeds give by default.
- Where should the history live: a new file, or extra rows in `train.jsonl`? `web/` types a `train.jsonl` row as `{ step, elapsed_sec, loss }` (`TrainPoint` in `web/src/lib/data.ts`), so rows without `loss` would need checking against the loss chart first.
- `web/` treats a run with no `eval.json` as still in progress. Should a partial history be visible before the run finishes?
- `sir_yes_sir` has the same single final evaluation. Is this an `addition`-only change, or shared?
