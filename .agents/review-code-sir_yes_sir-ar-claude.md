# Code review: `sir_yes_sir` / `ar`

- **Persona**: `.agents/reviewer-code.md`.
- **Code reviewed**: commit `bc2244b` (the working tree at `89bdcb2` differs only by files under `.agents/`).
- **Runs**: 8 `ar` runs exist under `results/sir_yes_sir/`, built from `586f44d`, `1eb53a1` and `86e46ea`. All are marked `OUTDATED: built against draft spec`, and all 8 `studies/sir_yes_sir/*.json` are `draft` with `runs: []`. No current claim rests on them, so I reviewed the current code and only checked that the `OUTDATED` marking is justified.
- **Order of work**: own reading and checks first, then `.agents/review-code-sir_yes_sir-ar.md` (Part B).

## Verdict

No bug found that would make `exact_match` wrong. The code matches its spec. What a result could support is narrower than the metric names suggest: halting is a constant and tests nothing (finding 1), the evaluation set is not held out when the problem space is small (finding 2), and the runner has no split that could show memorisation, wrap-around failure or length generalisation (finding 3).

## Part A: my review

### Findings

**1. Every problem in a run has the same length, so halting is a constant. Medium; `halt_accuracy` and the halting part of `exact_match` carry no information.**

- Where: `config.py:74` (`x_length` is one integer); `variants/ar/run.py:24-39` passes the same value to the training and evaluation datasets; `inputs.py:150-174` uses it for every problem.
- What happens: `[EOS]` always follows exactly `L` outputs. Under `token_type_pe` the last output token even carries `position_id = L`. The model can learn "emit `[EOS]` at this position" without reading the input.
- Evidence (my scratch run, `B = 4`, `L = 3`, `token_type_pe`): after 100 training steps `exact_match` was 0.7% and `halt_accuracy` was already 100%, and it stayed at 100% at every checkpoint while exact match stayed under 3%.
- Why it matters: the spec describes halting as part of the rule, and the `cls-vs-ar` study asks about "halting reliability". With a fixed length neither can be assessed. A reader seeing `halt_accuracy = 1.0` would take it as evidence the model knows when the input ends.

**2. With a small symbol pool or a fixed alphabet, the evaluation problems have been trained on. Medium for any small-pool or `resample_alphabet=False` result.**

- Where: `variants/ar/run.py:32-39`. Evaluation uses the same generator with a different seed; nothing is held out.
- Size of the problem space: with `resample_alphabet=False` it is `B^L` (64 at the defaults `B = 4`, `L = 3`). With a pool of 3, `B = 3`, `L = 2` it is 54. Both are smaller than one training batch.
- At the default full pool this does not arise: the space is 240 x 239 x 238 x 237 x 64 = 207,068,682,240. In my scratch run with a pool of 8 (space 107,520) 15 of 150 evaluation problems had appeared in 19,200 training problems.
- Consequence: in those regimes `exact_match` is accuracy on seen problems. That is acceptable for a smoke test and is what the `memorisation-vs-in-context` study wants on its memorisation arm, but it must not be described as generalisation.

**3. The runner cannot answer three of the questions the draft studies ask. Medium; not a defect in a number, a gap before those studies can run.**

- Memorisation: a model trained with `resample_alphabet=False` is evaluated on the same fixed alphabet (`run.py:32-39`). A model that ignores the prompt scores 100%. Telling memorisation from lookup needs an evaluation on shuffled alphabets.
- Wrap-around: `evaluate_ar` reports one pooled `step_accuracy` (`variants/ar/evaluation.py:102-120`). At `B = 240` a wrap-around step occurs once in 240, so a model that never handles it still scores about 99.6%.
- Length generalisation: there is no split at another length and no saved model.
- Also, at `B = 2` the successor of each symbol is the other one whichever way round the alphabet is written, so `B = 2` does not test reading the alphabet order at all.

**4. `step_accuracy` is conditional on the reference outputs. Low.** `evaluate_ar` feeds the correct outputs back after an error, and an early `[EOS]` does not stop the scoring of later steps. `exact_match` is unaffected (checked below).

### What I read and found correct

- **Reference** (`reference.py`): successor is `alphabet[(i + 1) % B]`, flag is `FINISHED` only at the last step. Checked against both spec examples by reading.
- **Sampling** (`inputs.py:150-174`): alphabet drawn without replacement from the pool, or the fixed prefix when `resample_alphabet=False`; inputs drawn uniformly from the alphabet.
- **Layout** (`inputs.py:65-99`): output positions start at 1, so the token that predicts `o_k` sits at position `k`, the same as `x_k`.
- **Labels** (`variants/ar/inputs.py`): pre-shifted; only the `L` outputs and `[EOS]` are supervised.
- **Evaluation** (`variants/ar/evaluation.py:102-120`): predictions are taken at exactly the supervised positions; exact match requires all `L + 1`, including `[EOS]`; a premature `[EOS]` fails both exact match and halting.
- **Same problems at every checkpoint**: the evaluation dataset object is re-iterated from its seed each time.
- **Config**: all 20 fields reach a consumer. `seed` and `eval_seed` must differ.
- **Eval mode and state**: `model.eval()` is called; no dropout, no state between batches.

### Historical runs

- I confirmed `experiments/sir_yes_sir` and `yessir` are identical across `586f44d`, `1eb53a1` and `86e46ea` (empty diff).
- The code at `586f44d` implements a different task: three tokens per step including a wrap indicator, a different position layout, and `halt_accuracy` computed per step. `OUTDATED` is the right marking. I did not audit that code further.
- Five of the eight runs record `git_dirty: true`, so even their own commit does not fully identify the code that produced them.

### What I ran

- `./yessir.sh test`: 360 passed.
- Scratch check of teacher-forced exact match against `greedy_generate_ar`, problem by problem, on a partially trained tiny model (`B = 4`, `L = 3`, pool of 8, 150 problems, 8 checkpoints, in all three embedding modes): **0 disagreements in 3,600 comparisons**. Exact match was low throughout (0% to 4%), so this mostly exercises problems that fail early; it is weaker evidence than the addition checks.

### What I did not get to

- No full-size run; nothing here says the model learns the task at the default settings.
- GPU behaviour, `web/`, `yessir/vertex.py`.

## Part B: check of `review-code-sir_yes_sir-ar.md`

I read it in full.

**Correct, and I confirmed independently**

- Section 1 (historical runs): correct. 8 `ar` and 6 `sm_mh` runs, all `OUTDATED`; the three commits are identical in the relevant directories; five `ar` runs are dirty; the old `halt_accuracy` was per step.
- Finding 1 (studies ask what the runner cannot measure): correct; same as my finding 3, including the `B = 240` arithmetic.
- Finding 2 (train and evaluation overlap in small universes): correct in substance; same as my finding 2. The `B = 2` observation is right and useful.
- Finding 3 (teacher-forced `step_accuracy` and `halt_accuracy`): correct. It reports matching exact match in all three modes on a partially trained model, which agrees with my check.
- Finding 4 (one evaluation point at the default `max_steps`; `resample_alphabet` changes the active symbol set unless `symbol_pool_size` is set; inspection CLI lacks pool flags): correct from the code. At `max_steps = 1000` the default interval is 2,500, so only the final evaluation runs.
- Finding 5 (test asymmetry with `cls`): consistent with the test names I listed; I did not open each test.

**Errors**

- Finding 2 gives the default problem space as 207,085,393,920. The product 240 x 239 x 238 x 237 x 64 is 207,068,682,240. The conclusion (no overlap at defaults) is unaffected.

**Missing**

- My finding 1. The review notes that length generalisation cannot be measured, but does not say that a fixed length makes `halt_accuracy`, and the halting requirement inside `exact_match`, uninformative in every run. Its own section 1 cites an old run with `halt_accuracy = 1.0` and `step_accuracy = 0.01` and attributes that to the old per-step metric; my scratch run shows the same picture under the current per-problem metric.
