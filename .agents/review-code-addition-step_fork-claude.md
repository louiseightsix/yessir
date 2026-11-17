# Code review: `addition` / `step_fork`

- **Persona**: `.agents/reviewer-code.md`.
- **Code reviewed**: commit `bc2244b` (the working tree at `89bdcb2` differs only by files under `.agents/`).
- **Runs**: none. `results/addition/` does not exist and every addition study is `draft` with `runs: []`.
- **Order of work**: own reading and checks first, then `.agents/review-code-addition-step_fork.md` (Part B).

## Verdict

No bug found that would make `exact_match` wrong. The one mechanism specific to this variant, grouping three branches per transition into one batch, works. The caveats are the same family as the other step variants: secondary metrics are conditional on reference state (finding 1) and the harness supplies the column index (finding 2).

## Part A: my review

### Findings

**1. `step_accuracy`, `digit_accuracy`, `carry_accuracy` and `halt_accuracy` are computed from reference state at every column. Medium; moves `step_accuracy` in `eval.json`.**

- Where: `variants/step_fork/evaluation.py:52-66`. `rollout_step_fork` is never called by a run.
- The persona asks that each step receive the model's own carry and that evaluation stop on the model's own halt. The spec chooses teacher forcing and argues `exact_match` is unaffected. I tested that (see "What I ran"); it held.
- Concrete case: `9995 + 5` in base 10, model fails only the first carry. Free-running, the later columns are wrong; teacher-forced, they are scored with the correct carry and `step_accuracy` is 0.8.
- One thing is cleaner here than in `step_ar`: the three branches are separate sequences, so nothing within a column is conditioned on a reference value. `halt_accuracy` and `digit_accuracy` are therefore comparable with `cls`, and not with `step_ar`.

**2. Under `token_type_pe` (the default) the harness supplies the column index. Low to medium; affects what the headline means.**

- Where: `variants/step_fork/inputs.py:47` and `:72`. The carry token and the query token both get `position_id = k`.
- Same point as for `cls` and `step_ar`: in this mode the model matches position ids and does not count columns.

**3. `p_terminal` is ignored when `all_steps=True`, and `all_steps` batches are correlated. Low.** Same as my `cls` review, finding 4.

**4. Iterating the dataset directly needs a batch size that is a multiple of 3. Low; cannot affect a run started through the CLI.** `run.py:47` passes `3 * batch_size`. Anyone calling `train()` on `AdditionStepForkDataset` themselves gets branches of one transition split across batches. A guard or a note in the dataset would prevent it.

### What I read and found correct

- **Reference**: hand cases and 20,000 random problems against Python integers, 0 mismatches (details in my `ar` review).
- **Layout** (`variants/step_fork/inputs.py:25-112`): `c_k [SEP] o_0..o_{k-1}` then one of `[S]`, `[C]`, `[F]`; history positions `0..k-1`, query at `k`. Matches the spec.
- **Labels** (`inputs.py:144-149`): one target per branch at the query position, everything else ignored.
- **Batch grouping**: the dataset yields the three branches of a transition consecutively and the runner takes batches of `3 * batch_size`, so each batch holds whole transitions, and stays aligned across the evaluation chunks because the stream is never restarted.
- **Evaluation regrouping** (`variants/step_fork/evaluation.py:73-105`): predictions are read at each row's own last position and consumed three at a time in `[S]`, `[C]`, `[F]` order, the same order the examples were built in.
- **No answer leak**: none of the three targets appears in any branch input.
- **Exact match** requires digit, carry and flag at every step.
- **Config**: all 29 fields reach a consumer, with the `p_terminal` exception.

### What I ran

- `./yessir.sh test`: 360 passed.
- Scratch check of teacher-forced exact match against `rollout_step_fork`, problem by problem, on a partially trained tiny model (bases 2 and 3, lengths 1 to 3, 150 problems, 8 checkpoints, exact match between 11% and 19%): **0 disagreements in 1,200 comparisons**.

### What I did not get to

- The equivalence check covers `token_type_pe`, CPU and short sequences only.
- No full-size run. `.agents/addition.md` reports that the prototype's `step_fork` reached only about 50% in-distribution; I did not look into why, and nothing I read explains it as a code fault.
- I did not test that removing `train_batch_size=3 * batch_size` would be caught; the existing review says no test would catch it, and I did not verify that.

## Part B: check of `review-code-addition-step_fork.md`

I read it in full.

**Correct, and I confirmed independently**

- Finding 1 (teacher-forced secondary metrics; exact match unaffected): correct. Its argument that the branches do not condition on one another is the right reason this variant has no within-step effect.
- Finding 2 (loss is a mean over `3 * batch_size` targets here and a sum over three heads in `cls`): correct. With AdamW and no weight decay the factor of 3 does not change the updates in any way that matters; only the logged loss values differ.
- Finding 3 (silent no-ops; direct dataset use): correct. I confirmed `symbol_set="decimal"` with base 12 and `p_prior=1.0` is accepted.
- Finding 4 (`eval_permuted_canonical` default; split-seed collision): correct. I reproduced the shared seed 2003.
- Section 2.3's statement that the three step variants see the same `(problem, k)` stream for a shared seed is consistent with the code: all three use `iter_problem_steps` with the same seed.

**Where I would qualify**

- The header calls the verdict "zero headline-moving implementation bugs" and lists `step_accuracy` as a "headline" metric in finding 1's title. Both cannot be read literally at once: the review's own finding 1 shows the second number in `eval.json` moves. The body is right; the summary line overstates.
- Finding 4 says the colliding OOD splits have correlated "byte prefixes, operand lengths, and carry-chain coins". I measured 1 of 200 problems sharing an alphabet prefix. Negligible.
- As in the other reviews, exact-match equivalence is asserted from the argument, and the missing empirical comparison is listed as a test gap. My scratch check now supports it for one embedding mode.

**Missing**

- My finding 2: the harness supplies the column index under the default mode.
- The observation that `halt_accuracy` and `digit_accuracy` are comparable between `step_fork` and `cls` but not with `step_ar`. The `step_ar` review makes this point; this one does not, although it matters for the comparison `step_fork` exists to support.
