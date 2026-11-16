# Addition: most evaluation metrics are not visible in `web/`

Open issue found in review on 2026-11-16. Nothing here is implemented or decided.

## What happens now

An `addition` run writes two evaluation files (`train_and_evaluate` in `experiments/addition/train_and_evaluate.py`):

| File | Contents | Read by `web/` |
| :--- | :--- | :--- |
| `eval.json` | Two numbers per split: `{prefix}exact_match` and `{prefix}step_accuracy` (prefix is empty for `id`, otherwise `ood_base3_`, `extrap_len_`, `permuted_`, ...) | Yes |
| `eval_splits.json` | All nine `EvalMetrics` fields per split, keyed by split name | No |

`loadRun` in `web/src/lib/data.ts` reads only `config.json`, `eval.json`, `train.jsonl` and `README.md`. Nothing under `web/src` mentions `eval_splits.json`, and no spec or README describes the file either; the only description is the docstring of `train_and_evaluate`.

## What is hidden as a result

For every split, these seven values exist on disk but cannot be seen in the browser:

- `digit_accuracy`, `carry_accuracy`, `halt_accuracy`
- the four first-error rates: `out_of_alphabet`, `prior_intrusion`, `procedural_error`, `syntax_malformation`

That matters for reading results. `exact_match` says a run failed but not how: a model that adds correctly and never halts, and a model that cannot carry, show the same low `exact_match`. The `permuted` split in particular exists to measure `prior_intrusion`, which is only in `eval_splits.json`.

## Constraints in `web/` to keep in mind

- The headline metric is the first key in `eval.json` (`headlineMetric` in `web/src/lib/data.ts`), so `exact_match` must stay first.
- The run page draws one tile per metric name, and the runs table one column per metric name, taken as the union of `eval.json` keys across runs.
- A tile whose value is not a number is shown as "pending eval". `carry_accuracy` is `null` when no evaluated column has a carry, so putting it in `eval.json` would mislabel that case.

## Options

1. **Teach `web/` to read `eval_splits.json`.** Keep `eval.json` as the compact headline file. On the run page, add a per-split table (splits as rows, the nine metrics as columns) and render `null` as "n/a". The runs table and headline are untouched.
2. **Flatten everything into `eval.json`.** No `web/` change, but with the default four splits that is 36 keys, so 36 tiles on the run page and 36 columns in the runs table, plus the `null` problem above.
3. **Leave it** and read `eval_splits.json` by hand when a run needs diagnosing.

Option 1 keeps the tables readable and is the one that makes the first-error rates usable.

## Open questions for Louis

- Should studies be able to chart a diagnostic metric (for example `prior_intrusion` on the `permuted` split), or is the run page enough?
- Should `eval_splits.json` be described in `experiments/addition/spec/task.md` §3 alongside the metrics it holds?
