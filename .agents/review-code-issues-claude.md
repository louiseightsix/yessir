# Code issues to address

A deduplicated list of the issues raised in the eight `.agents/review-code-*-claude.md` files, grouped by kind and ordered from most to least serious. Code reviewed at commit `bc2244b`.

Nothing here makes `exact_match` wrong. Teacher-forced exact match was checked against free-running rollout on partially trained models, problem by problem: no disagreements in about 30,000 comparisons. Every issue below is about other metrics, about what a result can be taken to show, or about the code around the metric.

Each entry gives: where, what goes wrong (with a concrete trigger), which variants it affects, and which `-claude` review files raised it. In the "Raised in" lines, `addition-ar` is short for `review-code-addition-ar-claude.md`, and so on.

---

## 1. Correctness errors

The code does something inconsistent or unintended.

### C1. A premature `[EOS]` in `ar` is filed as `out_of_alphabet` and costs a step

- **Where**: `experiments/addition/variants/ar/evaluation.py:122` (`syntax_ok = True` without query tokens); `experiments/addition/evaluation.py:132-152` (values are checked before `halt_ok`) and `:87-88` (anything outside the alphabet is `OUT_OF_ALPHABET`).
- **Issue**: without query tokens there is nothing between columns, so an `[EOS]` emitted after column `k - 1` lands in column `k`'s first value slot. It is classified `out_of_alphabet` and also fails column `k` in `step_accuracy`. The same early halt is `syntax_malformation` in `ar` with query tokens and in every other variant, and costs no step there. A missing final `[EOS]` is `syntax_malformation` in all variants.
- **Trigger**: one-column problem, model emits `[EOS]` in place of `o_0`: `out_of_alphabet`. Same problem with everything right but no final `[EOS]`: `syntax_malformation`.
- **Affects**: `addition` / `ar` with `use_query_tokens=False`. This matters on `extrap_len`, where stopping early is the expected failure.
- **Note**: the spec can be read either way, so the definition needs deciding first. The test that would need changing is listed as T2.
- **Raised in**: `addition-ar` (A3), `addition-addressed_ar` (A5).

### C2. `_split_seed` can give two splits the same seed, and can move one split in a seed sweep

- **Where**: `experiments/addition/config.py:353-356` and `:385`.
- **Issue**: a split's seed is bumped by one if it equals the training seed, without checking the other splits. This causes two problems:
  1. Two OOD splits can end up with the same seed.
  2. A training seed of 2002, 2011, 2999 or 3999 (at default `eval_seed`) shifts the matching split onto different problems. A seed sweep then no longer shares one evaluation set.
- **Trigger**: `bases="2,8"`, `eval_ood_bases="3,4"`, `eval_seed=999`, `seed=2002` gives both `ood_base3` and `ood_base4` seed 2003. In practice the two streams diverge almost at once (1 of 200 problems shares an alphabet prefix), so problem 2 is the one that matters.
- **Affects**: all addition variants.
- **Raised in**: `addition-cls`, `addition-step_ar`, `addition-step_fork`, `addition-addressed_ar` (all Part B).

### C3. Reusing an output directory mixes two runs' logs

- **Where**: `yessir/metrics.py:40` opens `train.jsonl` and `eval.jsonl` in append mode. `config.json` and `eval.json` are overwritten.
- **Trigger**: run twice with the same `--out-dir`. The JSONL files then hold both runs; the JSON files hold only the second.
- **Affects**: both experiments. Automatic run names avoid it.
- **Raised in**: `addition-cls` (A5).

### C4. `addressed_ar`'s `halt_accuracy` treats a wrong token between columns inconsistently

- **Where**: `experiments/addition/variants/addressed_ar/evaluation.py:149-151`.
- **Issue**: a wrong token where `[IDX]` should follow column `k` counts as a halting error. A wrong token before column 0 counts only as a syntax error.
- **Affects**: `addition` / `addressed_ar`, `halt_accuracy` only.
- **Raised in**: `addition-addressed_ar` (Part B).

---

## 2. Misleading metrics

These are computed as specified, but invite a wrong reading.

### M1. `addressed_ar`'s `step_accuracy` ignores wrong addresses (high)

- **Where**: `experiments/addition/variants/addressed_ar/evaluation.py:71-91` (the input is the reference trace, addresses included) and `:135-137` (a wrong address only sets `syntax_ok`); `experiments/addition/evaluation.py:193-204` (a step counts as correct from digit and carry alone).
- **Trigger**: one column, address wrong, digit and carry right gives `exact_match = 0.0`, but `step_accuracy`, `digit_accuracy` and `halt_accuracy` are all 1.0.
- **Issue**: a model whose address counter fails beyond column 20 still gets the correct address before each digit, and is not penalised for its own. `extrap_len_step_accuracy`, which is written to `eval.json`, then measures what `addressed_step_ar` measures.
- **Fix options**: require `syntax_ok` for a correct step, or report a separate address accuracy (see G4).
- **Raised in**: `addition-addressed_ar` (A1).

### M2. All metrics except `exact_match` are computed from reference state

- **Where**:
  - `experiments/addition/variants/*/evaluation.py`: every evaluator builds its inputs from `reference_add`.
  - `experiments/sir_yes_sir/variants/*/evaluation.py`: likewise from the reference outputs.
  - No run calls `rollout_*` or `greedy_generate_*`.
- **Issue**: after the first mistake the model is given the correct carry and history again, and a premature halt does not stop the scoring. So `step_accuracy` (in `eval.json`), `digit_accuracy`, `carry_accuracy` and `halt_accuracy` mean "accuracy given a correct history", not progress through a rollout.
- **Trigger**: `999 + 1` in base 10, model fails only the first carry. Free-running, every later column is wrong; teacher-forced `step_accuracy` is 0.75. In `cls`, a model that predicts `FLAG_FINISHED` at column 0 still has every later column scored.
- **Affects**: all 8 variants.
- **Fix options**: report free-running rollout metrics alongside, or document the meaning in the spec and the results.
- **Raised in**: all eight files.

### M3. Within-step teacher forcing makes `halt_accuracy` and `digit_accuracy` easier in `step_ar` and `addressed_step_ar`

- **Where**: `experiments/addition/variants/step_ar/evaluation.py:59-77`, `:104-106`; the same structure in `addressed_step_ar/evaluation.py:60-79`, `:84-108`.
- **Issue**: the flag is read after the reference carry is already in the input. Under `carry_first` the digit is read after the reference carry too. In `cls` and `step_fork` neither is.
- **Trigger**: `7 + 5` in base 10 under `carry_first`, model predicts carry `0`. Both the digit and the flag are scored from the reference carry `1`, so both can count as correct.
- **Affects**: comparisons of `halt_accuracy` between the `step_ar` family and `cls` or `step_fork`, and of `digit_accuracy` between `carry_first` and `sum_first`. `step_accuracy` itself is unaffected, because it needs both values right.
- **Raised in**: `addition-step_ar` (A2), `addition-addressed_step_ar` (A2).

### M4. sir_yes_sir `cls` `step_accuracy` ignores a premature halt; `ar` does not

- **Where**: `experiments/sir_yes_sir/variants/cls/evaluation.py:98-135`.
- **Issue**: the digit head and the flag head are read independently. A premature `FINISHED` at step 0 leaves all `L` digits scored in `cls`, but costs a step in `ar`, where `[EOS]` takes the output slot.
- **Affects**: `cls` versus `ar` `step_accuracy` on unconverged models.
- **Raised in**: `sir_yes_sir-cls` (A5).

---

## 3. Experimental confounds and default traps

The code matches its spec, but a run at these settings does not test what its study says it tests.

### D1. Under the default `token_type_pe`, the code tells the model which column it is on

- **Where**:
  - `experiments/addition/config.py:111` sets the default.
  - Per-variant position ids: `ar/inputs.py:21-43`, `cls/inputs.py:53`, `step_ar/inputs.py:82-88`, `step_fork/inputs.py:47,72`, `addressed_ar/inputs.py:89-115`.
  - Shared address entries: `experiments/addition/inputs.py:259-274`.
- **Issue**: every token for column `k` carries `position_id = k`, and so do `a_k` and `b_k`. The model can align columns by matching position ids, without counting.
  - **`ar`**: the column counter is external, even though the spec contrasts `ar` with harnessed variants. The `ar` versus `step_ar` comparison then does not isolate internal state tracking.
  - **`addressed_ar` and `addressed_step_ar`**: addresses are redundant, because a digit can be fetched, and an address copied, by position. In zero-preamble mode the address `s_j` sits at position `j`, which hands the model the rank it is meant to infer.
- **Trigger**: Example 1 at `k = 2`. The predicting token sits at position 2, and so does `a_2 = '9'`.
- **Fix options**: make `nope` (or `token_type_nope`) the default for the variants whose point depends on it, or require studies to state the mode.
- **Raised in**: all six addition files.

### D2. `extrap_len` contains address digits and widths never seen in training

- **Where**: `experiments/addition/inputs.py:115-138` (`encode_address`) combined with the default `max_length=20` and `eval_extrap_max_length=40`.
- **Issue**:

| Base | Addresses on `extrap_len` (columns 21 to 40) |
| :--- | :--- |
| 2 | First 6-digit addresses (training goes up to 5 digits) |
| 4 | Fully covered |
| 8 | New leading digits 3, 4 and 5 |
| 10 | New leading digits 3 and 4; leading digit 2 seen in training only at column 20, on an overflow step |

  The split pools all four bases, so covered and uncovered cases are mixed with no breakdown.
- **Affects**: `addressed_ar`, `addressed_step_ar`.
- **Raised in**: `addition-addressed_ar` (A3), `addition-addressed_step_ar` (A4).

### D3. Zero-preamble mode ties operand length to the base

- **Where**: `experiments/addition/inputs.py:431` (`eff_min = max(min_length, base)`).
- **Trigger**: minimum sampled length is 2, 4, 8, 10 for bases 2, 4, 8, 10; an `ood_base12` split would use lengths 12 to 20 only.
- **Issue**: in this mode a large base always comes with long operands. Turning `include_alphabet` off also changes problem length.
- **Affects**: `addressed_ar`, `addressed_step_ar`; the `zero-preamble` study.
- **Raised in**: `addition-addressed_ar` (A4), `addition-addressed_step_ar` (A3).

### D4. sir_yes_sir: a fixed length makes halting a constant

- **Where**: `experiments/sir_yes_sir/config.py:74`; `variants/*/run.py` pass the same `x_length` to training and evaluation.
- **Trigger**: `B = 4`, `L = 3`, `token_type_pe`. `halt_accuracy` was 100% at every checkpoint while exact match stayed between 0% and 3%, in both variants.
- **Issue**: `halt_accuracy`, and the halting part of `exact_match`, carry no information. The `cls-vs-ar` study asks about "halting reliability".
- **Raised in**: `sir_yes_sir-ar` (A1), `sir_yes_sir-cls` (A1).

### D5. With `p_prior=1.0`, no default split tests in-context reading

- **Where**: `experiments/addition/config.py:158` (`eval_permuted_canonical=False`) and `:358-373` (`id` and `extrap_len` inherit `p_prior`).
- **Issue**: both splits then use only the ascending canonical alphabet, where a model that ignores the prompt's alphabet scores 100%. Only the OOD-base splits use random alphabets, and they change the base at the same time.
- **Affects**: the `prior-override` study, all addition variants.
- **Raised in**: `addition-cls` (A3), `addition-step_fork` (Part B).

### D6. Base 2 is pooled into the `permuted` split

- **Where**: `experiments/addition/config.py:402-415`; `experiments/addition/evaluation.py:78-102`.
- **Issue**: in base 2 every in-alphabet error is classified `prior_intrusion`. With bases 2, 4, 8, 10 pooled, uniform random errors already give a `prior_intrusion` share of 39.7%, and there is no per-base breakdown.
- **Raised in**: `addition-step_ar` (Part B).

### D7. Evaluation sets are independent samples, not held-out sets

- **Where**: `experiments/addition/config.py:358-373` (`id`); `experiments/sir_yes_sir/variants/*/run.py` (evaluation is the same generator with another seed).
- **Measured overlap with training**:

| Setting | Evaluation problems also in training |
| :--- | :--- |
| Addition, defaults | 0 of 200 |
| Addition, `p_prior=1.0` | 6 of 200 |
| Addition, decimal pool, bases 2 and 4, lengths up to 3 | 107 of 200 |
| sir_yes_sir, `resample_alphabet=False` (universe of 64 problems) | almost all |
| sir_yes_sir, pool 3, `B = 3`, `L = 2` (universe of 54 problems) | almost all |
| sir_yes_sir, pool 8 | 15 of 150 |

  At default sir_yes_sir settings the universe is 207,068,682,240, so overlap does not arise.
- **Issue**: in small universes `exact_match` is accuracy on seen problems.
- **Raised in**: `addition-ar` (Part B), `addition-cls` (A3), `sir_yes_sir-ar` (A2), `sir_yes_sir-cls` (A2).

### D8. `all_steps` changes how many distinct problems a batch holds

- **Where**: `experiments/addition/inputs.py:537-539`; `experiments/sir_yes_sir/variants/cls/dataset.py:73-80`; batches are taken in order (`yessir/training.py:302-308`).
- **Issue**: all steps of a problem are consecutive.
  - **Addition**: a batch of 64 holds about 5 problems at default lengths, against 64.
  - **sir_yes_sir `cls`** at `L = 3`: about 21 problems against 64. `single_pass` also supervises `L` targets per example.
  - **Consequence**: comparing supervision modes at equal `max_steps` changes several things at once.
- **Affects**: `cls`, `step_fork`, `step_ar`, `addressed_step_ar`, sir_yes_sir `cls`; the `step-sampling` and `cls-training-passes` studies.
- **Raised in**: `addition-cls` (A4), `addition-step_ar` (A4), `addition-step_fork` (A3), `sir_yes_sir-cls` (A3).

### D9. Full-trace variants train on far more per step than step variants

- **Where**: `yessir/training.py:151-183` (loss is a mean over all supervised tokens in the batch).
- **Issue**: a batch of 64 problems in `ar` or `addressed_ar` holds about 14 times as many column transitions as a batch in the step variants. Within a batch, long problems dominate: a 21-column problem has 43 targets, a 1-column problem 3.
- **Affects**: comparisons between variants "by step".
- **Raised in**: `addition-ar` (A4), `addition-addressed_ar` (Part B).

### D10. sir_yes_sir at `B = 2` does not test reading the alphabet order

- **Issue**: the successor of each symbol is the other one, whichever way round the alphabet is written.
- **Affects**: the lower end of the `alphabet-length` study.
- **Raised in**: `sir_yes_sir-ar` (A3).

### D11. sir_yes_sir `cls` uses a 256-way flag head

- **Where**: `experiments/sir_yes_sir/variants/cls/run.py` (`num_classes=(VOCAB_SIZE, VOCAB_SIZE)`).
- **Issue**: 254 of the flag head's classes are never used. At defaults `cls` has 19,241,472 parameters against 18,978,816 for `ar`.
- **Affects**: parameter-count comparisons in `minimum-model-size` and `cls-vs-ar`.
- **Raised in**: `sir_yes_sir-cls` (Part B).

---

## 4. Config validation and silent no-ops

### K1. `p_terminal` is ignored when `all_steps=True`

- **Where**: `experiments/addition/inputs.py:537-539`.
- **Issue**: the value is still written to `config.json`, so two runs that differ only in `p_terminal` look different but are identical.
- **Affects**: all four addition step variants.
- **Raised in**: `addition-cls`, `addition-step_ar`, `addition-step_fork`, `addition-addressed_step_ar`.

### K2. `fixed_d_model=True` adds width back when `d_scratch=0`

- **Where**: `experiments/addition/config.py:331-340`; the same logic in `experiments/sir_yes_sir/config.py:207-221`.
- **Trigger**: `embedding_mode="nope"`, `d_scratch=0` gives 48 scratch dimensions and `d_model = 304`, not 256.
- **Affects**: capacity sweeps (`harness-vs-capacity`, `minimum-model-size`).
- **Raised in**: `addition-step_ar` (Part B).

### K3. `d_position=0` under a NoPE mode shrinks `d_model` despite `fixed_d_model=True`

- **Where**: `experiments/sir_yes_sir/config.py:218`; the same logic at `experiments/addition/config.py:337`.
- **Trigger**: `nope` or `token_type_nope` with `d_position=0` gives `d_model = 480`, against 512 for `token_type_pe` at the default `d_position=32`.
- **Issue**: an embedding-mode ablation that sets `d_position=0` for the NoPE arms silently changes width.
- **Raised in**: `sir_yes_sir-cls` (Part B).

### K4. `symbol_set` is ignored when `p_prior=1.0`

- **Where**: `experiments/addition/inputs.py:426-427`; `experiments/addition/config.py:199` (the pool-size check is skipped).
- **Trigger**: `symbol_set="decimal"`, `bases="2,12"`, `p_prior=1.0`, `eval_ood_bases=None` is accepted, and base 12 draws `'A'` and `'B'`.
- **Raised in**: `addition-cls`, `addition-step_fork` (Part B).

### K5. sir_yes_sir `symbol_pool_size` and `resample_alphabet` interact silently

- **Where**: `experiments/sir_yes_sir/inputs.py:166`.
- **Issue**:
  - With `resample_alphabet=False`, `symbol_pool_size` has no effect.
  - With `symbol_pool_size=None`, switching `resample_alphabet` also changes how many distinct symbols are used (4 against 10 for `decimal`, `B = 4`). An ablation must set `symbol_pool_size=B` to hold that fixed.
- **Raised in**: `sir_yes_sir-ar`, `sir_yes_sir-cls` (Part B).

### K6. sir_yes_sir at default `max_steps` evaluates only once

- **Where**: `experiments/sir_yes_sir/config.py:94` (`max_steps=1000`); `yessir/defaults.py` (default interval `max(2500, max_steps // 20)`).
- **Issue**: the default interval is 2,500, which is beyond the end of a default run, so `eval.jsonl` holds a single point. Studies about convergence speed need an explicit `eval_every`.
- **Raised in**: `sir_yes_sir-ar` (Part B).

### K7. `lr` is not checked for NaN

- **Where**: `experiments/addition/config.py:269`, `experiments/sir_yes_sir/config.py:163`.
- **Issue**: `lr <= 0` is false for NaN, so `lr=float("nan")` passes validation.
- **Raised in**: `sir_yes_sir-cls` (Part B).

### K8. `AdditionStepForkDataset` needs a batch size that is a multiple of 3

- **Where**: `experiments/addition/variants/step_fork/dataset.py`; only `run.py:47` passes `3 * batch_size`.
- **Issue**: calling `train()` on the dataset directly splits one transition's three branches across batches. A CLI run is not affected.
- **Raised in**: `addition-step_fork` (A4).

---

## 5. Latent issues in code no run calls

### L1. `rollout_addressed_ar` assigns metadata from the expected address width

- **Where**: `experiments/addition/variants/addressed_ar/inputs.py:89-115`.
- **Issue**: if the model emits an address of the wrong width, the position ids and token types of every later token are misaligned. The problem has already failed by then, but any post-error analysis of a rollout would be built on wrong inputs.
- **Raised in**: `addition-addressed_ar` (A2, Part B).

### L2. `parse_addressed_ar_trace` accepts special tokens as address or value

- **Where**: `experiments/addition/variants/addressed_ar/generation.py` (`parse_addressed_ar_trace`).
- **Issue**: it reports the trace as well-formed when `[SEP]`, `[PAD]` or `[BOS]` sit in address or value slots.
- **Raised in**: `addition-addressed_ar` (Part B).

### L3. Rollout helpers continue on any flag that is not `FLAG_FINISHED`

- **Where**:
  - `experiments/addition/variants/cls/generation.py:63`
  - `step_fork/generation.py:64`
  - `step_ar/generation.py:77`
  - `addressed_step_ar/generation.py:81`
  - `experiments/sir_yes_sir/variants/cls/generation.py:70`
- **Issue**: evaluation requires exactly `FLAG_CONTINUE`; rollout treats any other byte as "continue". This is consistent only if callers compare the emitted flags too, as the convergence tests do.
- **Raised in**: `addition-step_ar`, `sir_yes_sir-cls` (Part B).

### L4. Step evaluators build every step of every problem before batching

- **Where**: `experiments/addition/variants/addressed_step_ar/evaluation.py:60-79`. The other step variants have the same structure.
- **Issue**: memory use at length 40 in base 2, where sequences run to about 800 tokens, was not measured.
- **Raised in**: `addition-addressed_step_ar` (Part B).

---

## 6. Missing capability for planned studies

### G1. No wrap-around accuracy in sir_yes_sir

- **Where**: `experiments/sir_yes_sir/variants/*/evaluation.py` report one pooled `step_accuracy`.
- **Issue**: at `B = 240`, a model that never handles wrap-around still scores about 99.6%.
- **Needed by**: `alphabet-length`.
- **Raised in**: `sir_yes_sir-ar` (A3), `sir_yes_sir-cls` (A4).

### G2. No shuffled-alphabet evaluation for a canonical-trained sir_yes_sir model

- **Where**: `experiments/sir_yes_sir/variants/*/run.py` evaluate with the training `resample_alphabet`.
- **Issue**: a model that ignores the prompt's alphabet scores 100%.
- **Needed by**: `memorisation-vs-in-context`.
- **Raised in**: `sir_yes_sir-ar` (A3), `sir_yes_sir-cls` (A4).

### G3. No other-length split and no saved model in sir_yes_sir

- **Issue**: length generalisation cannot be measured, during the run or afterwards.
- **Needed by**: `sequence-length`, `embedding-modes`.
- **Raised in**: `sir_yes_sir-ar` (A3), `sir_yes_sir-cls` (A4).

### G4. No per-base breakdown and no address accuracy in addition

- **Where**: `experiments/addition/train_and_evaluate.py:162-182`; `experiments/addition/evaluation.py` (`EvalMetrics`).
- **Issue**: the evaluation artefacts carry no per-base figures and no address accuracy. Without them D2, D6 and M1 cannot be separated after a run.
- **Raised in**: `addition-addressed_ar` (A1, A3), `addition-step_ar` (Part B).

---

## 7. Test gaps

### T1. No test compares teacher-forced exact match with free-running rollout

- **Issue**: the specs' central argument has no test, in any variant, on a real model that gets some problems wrong.
- **Status**: my scratch checks found no disagreement in every variant. For addition, this was in `token_type_pe` only, on lengths up to 3. `nope` and `token_type_nope` for addition are untested, both in the repo and by me.
- **Raised in**: all eight files report the scratch check; `addition-ar`, `addition-cls` and `addition-step_fork` note that the repo has no such test.

### T2. `test_ar.py` asserts the current premature-`[EOS]` behaviour

- **Where**: `experiments/addition/tests/test_ar.py:489` ("Premature EOS inside column 0 -> ... out_of_alphabet=1.0").
- **Issue**: if C1 is resolved the other way, this test must change with it.
- **Raised in**: `addition-ar` (A3, Part B).

---

## 8. Docs, tooling and readability

### R1. Both task specs mention `bfloat16` autocast

- **Where**: `experiments/addition/spec/task.md:145`, `experiments/sir_yes_sir/spec/task.md:78`.
- **Issue**: there is no autocast anywhere in `yessir/` or `experiments/`. Everything runs in float32.
- **Raised in**: `addition-ar`, `addition-step_ar`, `addition-addressed_step_ar`, `sir_yes_sir-cls` (Part B).

### R2. `studies/addition/query-tokens.json` lists `[F]` for all four variants

- **Issue**: `ar` and `addressed_ar` use only `[S]` and `[C]`, because they halt with `[EOS]`.
- **Raised in**: `addition-ar` (Part B).

### R3. The inspection CLI prints addition tokens as hex

- **Where**: `yessir/inspect_data.py:14` (`format_token`).
- **Issue**: `[S]`, `[C]`, `[F]` and `[IDX]` appear as `0xf8` to `0xfb`.
- **Raised in**: `addition-addressed_step_ar`, `addition-step_ar` (Part B).

### R4. The sir_yes_sir inspection CLI has no symbol-pool flags

- **Where**: `experiments/sir_yes_sir/inputs.py` (`add_common_dataset_args`).
- **Issue**: there is no `--symbol-set` or `--symbol-pool-size`, so the CLI always draws from the full pool.
- **Raised in**: `sir_yes_sir-ar` (Part B).

### R5. TUI float fields cannot be stepped down to 0

- **Where**: `yessir/tui.py:142` (`return max(step_f, next_f)`).
- **Issue**: for example, `p_prior` stops at 0.05. Typing 0 still works.
- **Raised in**: `addition-ar` (Part B).

### R6. `cls` logs a loss three times larger than the language-model variants

- **Where**: `yessir/training.py` (`parallel_classification_cross_entropy` sums the three heads).
- **Issue**: under AdamW the updates are effectively unchanged, but the loss curves cannot be overlaid without dividing by 3.
- **Raised in**: `addition-cls`, `addition-step_fork` (Part B).
