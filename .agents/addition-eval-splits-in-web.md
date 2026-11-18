# Most evaluation artefacts and new run fields are not visible in `web/`

Open issue originally noted on 2026-11-16, updated on 2026-11-17 after commits `be36867` and `ede2632`. Nothing on the `web/` data-loading side is implemented or decided yet.

## What happens now

[loadRun](file:///atelier/code/workshop/yessir/web/src/lib/data.ts#L168-L216) in `web/src/lib/data.ts` reads only `config.json` (partially), `eval.json`, `train.jsonl` (`{ step, elapsed_sec, loss }` only), and `README.md`.

Meanwhile, `train_and_evaluate` in [addition/train_and_evaluate.py](file:///atelier/code/workshop/yessir/experiments/addition/train_and_evaluate.py#L70-L79) and [sir_yes_sir/train_and_evaluate.py](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/train_and_evaluate.py#L75-L141) (documented in [addition/spec/task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L171-L177) and [sir_yes_sir/spec/task.md](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/spec/task.md#L74-L80)) now writes additional files and fields on disk:

| File / field | Experiment | Contents on disk | Read by `web/` |
| :--- | :--- | :--- | :--- |
| `eval.json` | `addition`, `sir_yes_sir` | Final-step headline pairs (`{prefix}exact_match`, `{prefix}step_accuracy` across enabled splits in `addition`; `exact_match`, `step_accuracy`, `halt_accuracy` in `sir_yes_sir`) | **Yes** |
| `eval_splits.json` | `addition` | Final-step full **11-field** `EvalMetrics` per split (`id`, `ood_base3`, `ood_base12`, `extrap_len`, `permuted`), **plus** `{split}_base{b}` per-base slices for multi-base splits (`id_base2`, `extrap_len_base8`, `permuted_base4`, ...) | **No** |
| `eval.jsonl` | `addition`, `sir_yes_sir` | Periodic checkpoint trajectory (`step`, `elapsed_sec`, top-level headline pairs, and nested `"splits"` with all 11 `EvalMetrics` fields + `{split}_base{b}` slices in `addition`, or `splits.id` including `wrap_around_accuracy` in `sir_yes_sir`) | **No** |
| `train.jsonl` (`tokens_seen`) | `addition`, `sir_yes_sir` | Cumulative supervised target count `tokens_seen` alongside `step`, `elapsed_sec`, `loss` | **No** (`TrainPoint` only types `step`, `elapsed_sec`, `loss`) |
| `config.json` (`started_at`) | `addition`, `sir_yes_sir` | UTC ISO-8601 run start timestamp ([artefacts.py](file:///atelier/code/workshop/yessir/yessir/artefacts.py#L117-L136)) | **No** (`startedAtMs` in [data.ts](file:///atelier/code/workshop/yessir/web/src/lib/data.ts#L214) still uses `config.json` filesystem `mtime`, which `git checkout` resets) |

## What is hidden as a result

1. **Nine of eleven `addition` `EvalMetrics` fields per split (`eval_splits.json` & `eval.jsonl`)**:
   - `oracle_step_accuracy` (single-step teacher-forced transition accuracy, contrasting against free-running rollout `step_accuracy`)
   - `digit_accuracy`, `carry_accuracy`, `halt_accuracy`
   - `address_accuracy` (`0.0..1.0` on `addressed_ar`, `null` on the other five variants)
   - the four first-error rates: `out_of_alphabet`, `prior_intrusion`, `procedural_error`, `syntax_malformation`
2. **Per-base split breakdowns (`{split}_base{b}` in `eval_splits.json` & `eval.jsonl`)**:
   - Needed to compare `permuted_base{b}` `prior_intrusion` against each base's `1 / (B - 1)` chance baseline (`spec/task.md` §3) and to separate per-base unseen-address profiles on `extrap_len_base{b}` (`B in {2, 4, 8, 10}`).
3. **Mid-training accuracy curves (`eval.jsonl` in both experiments)**:
   - Needed to see delayed grokking vs OOD base decay across steps (`grokking-and-overtraining`), convergence speed (`eval_every`), and live checkpoint progress while a run is still `"in_progress"` (`eval.json` is written only at `max_steps`).
4. **`sir_yes_sir` `wrap_around_accuracy` (`eval.jsonl` `splits.id`)**:
   - Needed for `alphabet-length` (where `B = 240` overall `step_accuracy` is `~99.6%` even at `0%` wrap-around accuracy); `sir_yes_sir` writes `wrap_around_accuracy` to `eval.jsonl` `splits.id` rather than `eval.json`.
5. **Token-normalised x-axes (`tokens_seen` in `train.jsonl`) and git-safe run ordering (`started_at` in `config.json`)**:
   - `ar` / `addressed_ar` supervise `~20x–30x` more target tokens per step than step-wise variants, so `tokens_seen` enables data-budget comparisons alongside `step`.

## Constraints in `web/` to keep in mind

- The headline metric is the first key in `eval.json` (`headlineMetric` in [data.ts](file:///atelier/code/workshop/yessir/web/src/lib/data.ts#L114-L115)), so `exact_match` must stay first.
- The run page draws one tile per metric name, and the runs table one column per metric name, taken as the union of `eval.json` keys across runs.
- A tile whose value is not a number is shown as `"pending eval"`. Both `carry_accuracy` (when no evaluated column has a carry) and `address_accuracy` (on all variants except `addressed_ar`) can be `null`, so putting them in `eval.json` would mislabel those cases.
- With default `addition` settings there are now 5 top-level splits (`id`, `ood_base3`, `ood_base12`, `extrap_len`, `permuted`, since `eval_permuted_canonical=True` by default) plus 11 per-base slices (`id_base{2,4,8,10}`, `extrap_len_base{2,4,8,10}`, `permuted_base{4,8,10}`) × 11 fields = **176 values** in `eval_splits.json`, so flattening into `eval.json` is unworkable.

## Options

1. **Teach `web/` to read `eval_splits.json`, `eval.jsonl`, `tokens_seen`, and `started_at`.**
   - Keep `eval.json` as the compact headline file for run tables.
   - On the run page: add a per-split diagnostic table from `eval_splits.json` (top-level splits as rows, collapsible `{split}_base{b}` sub-rows, the 11 metrics as columns, rendering `null` as `"n/a"`), plus checkpoint accuracy charts from `eval.jsonl` (including for `"in_progress"` runs before `eval.json` exists, falling back to the last `eval.jsonl` row when `eval_splits.json` is not written yet).
   - Parse `meta.started_at` in `loadRun` (`Date.parse`) with `fileMtimeMs` as fallback for older runs, and expose `tokens_seen` on `TrainPoint`.
2. **Leave `web/` unchanged for now** and inspect `eval_splits.json` / `eval.jsonl` directly when diagnosing runs.

## Open questions for Louis

- Should study pages be able to chart a diagnostic metric from `eval_splits.json` / `eval.jsonl` (for example `prior_intrusion` on `permuted`, `address_accuracy` on `extrap_len`, or `wrap_around_accuracy` in `sir_yes_sir`, and accuracy-vs-step curves for `grokking-and-overtraining`), or is showing them on the individual run page enough first?
- On the run page, should `{split}_base{b}` slices be folded under their parent split row (`id`, `extrap_len`, `permuted`) or toggled separately?
- Should loss / eval charts offer a `step` vs `tokens_seen` x-axis toggle?
