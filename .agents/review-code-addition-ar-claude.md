# Code review: `addition` / `ar`

- **Persona**: `.agents/reviewer-code.md` (does the code do what the spec and the claim say?).
- **Code reviewed**: commit `bc2244b` (the working tree at `89bdcb2` differs only by files under `.agents/`).
- **Runs**: none. `results/addition/` does not exist and all 13 `studies/addition/*.json` are `draft` with `runs: []`, so there is no reported number to check. This review is about what a future `ar` number would and would not mean.
- **Order of work**: I read the spec and code and ran my own checks first, then read `.agents/review-code-addition-ar.md` and checked it (Part B).

## Verdict

I found no bug that would make `exact_match` wrong. Three things would make a future `ar` claim mean less than it appears to: the harness supplies the column index under the default embedding mode (finding 1), every metric other than `exact_match` is conditional on the reference trace (finding 2), and a premature `[EOS]` is filed under the wrong error category (finding 3).

## Part A: my review

### Findings

**1. Under `token_type_pe` (the default), the harness gives the model the column index of every trace token. Medium; affects what the headline means, not its value.**

- Where: `experiments/addition/variants/ar/inputs.py:21-43` (`_trace_slot_metadata`), used by training (`inputs.py:65-87`), evaluation (`evaluation.py:65-81`) and rollout (`generation.py:43-51`).
- What happens: each trace token is tagged with `position_id = k` (or `k + 1` for the carry) and a token type, computed from its slot number. In rollout the harness, not the model, does this for every generated token. So when the model predicts `o_k`, the token it is reading from already carries "this is column `k`" as a sinusoid that matches `a_k` and `b_k` exactly.
- Why it matters: `spec/variants/ar.md` describes `ar` as generating the trace "with no external state rewriting", and the spec contrasts it with `step_ar`, where a harness tracks the column. Under `token_type_pe` the column counter is still external in `ar`. A comparison of `ar` against `step_ar` under the default mode therefore does not isolate "model keeps its own place" from "harness keeps place". Only `nope` removes this; `token_type_nope` still tells the model each generated token's role.
- Concrete case: Example 1 (`987 + 25`). When predicting `o_2` under `sum_first`, the input token is `c_2` at `(CARRY, 2)`, and `a_2 = '9'` sits at `(OPERAND_A, 2)`. No counting is needed.
- Headline impact: the number is computed correctly. The risk is the sentence written about it. This is consistent with the spec's layout table, so it is a claim-level gap, not a code defect.

**2. Every metric except `exact_match` is conditional on the reference trace. Medium; moves `step_accuracy`, which is written to `eval.json` next to `exact_match`.**

- Where: `variants/ar/evaluation.py:65-81` builds each input from `reference_add`; `rollout_ar` is never called from `run.py` or `train_and_evaluate.py`.
- The persona asks for autoregressive evaluation. The spec (`task.md` section 3) deliberately specifies teacher forcing and argues that `exact_match` is unaffected. I tested that argument instead of accepting it: see "What I ran". It held.
- `step_accuracy`, `digit_accuracy`, `carry_accuracy` and `halt_accuracy` do not have that property. After the first error the model is handed the correct carries and digits again. Under `carry_first`, `o_k` is predicted from the reference `c_{k+1}` even when the model's own carry was wrong.
- Concrete case: `999 + 1` in base 10, model fails only the first carry. Free-running output is wrong in every later column; teacher-forced `step_accuracy` is 0.75.
- Headline impact: none on `exact_match`. `step_accuracy` should be described as "per-column accuracy given a correct history", never as progress through a rollout.

**3. A premature `[EOS]` is classified as `out_of_alphabet`, a missing `[EOS]` as `syntax_malformation`. Low to medium; moves the error breakdown only.**

- Where: `variants/ar/evaluation.py:122` sets `syntax_ok = True` when `use_query_tokens=False`; `experiments/addition/evaluation.py:132-152` checks values before `halt_ok`; `evaluation.py:87-88` returns `OUT_OF_ALPHABET` for any non-alphabet token.
- Concrete case (I ran it): one-column problem, model emits `[EOS]` in the `o_0` slot gives `out_of_alphabet`; model emits everything correctly but no final `[EOS]` gives `syntax_malformation`.
- Consequence: "halts one column early" and "halts one column late" land in different categories, and with `use_query_tokens=True` or in any other variant the early halt is `syntax_malformation`. The same failure also costs `ar` one `step_accuracy` column that the other variants do not lose. On `extrap_len`, where stopping near column 20 is the expected failure, this would make `ar` look like it emits garbage symbols.
- The spec text is ambiguous here and a test asserts the current behaviour (`tests/test_ar.py`, "Premature EOS inside column 0"), so this is a definition to settle, not an accident.

**4. Loss is averaged per token, so long problems dominate each batch. Low.**

- Where: `yessir/training.py:151-183`. A 21-column problem contributes about 43 targets, a 1-column problem 3. Not a bug; relevant when comparing `ar` with the step variants at equal `max_steps`, since one `ar` batch also contains about 14 times more column transitions.

### What I read and found correct

- **Reference** (`reference.py`): hand-checked `1+1` and `0+0` in base 2, `999+1` in base 10, `B+B` and `B0+1` in base 12, the spec's base-3 example, and unequal lengths with and without overflow in base 2. I also compared 20,000 random problems (bases 2 to 16, lengths 1 to 12, random alphabets) against Python integers: 0 mismatches, flags always `CONTINUE` then one `FINISHED`, and an extra column exactly when the sum overflows.
- **Sampling** (`inputs.py:403-453`): base, lengths and digits uniform; carry-chain indices stay in bounds and do produce a generate-then-propagate chain; `permuted_canonical` never returns the ascending order.
- **Splits** (`config.py:358-416`, `train_and_evaluate.py:162-182`): at defaults the seeds are 999, 2002, 2011, 2999; OOD bases are removed from the list if they are trained on; `extrap_len` draws both lengths from 21 to 40. I generated the default splits and compared them with the first 200,000 training problems: 0 duplicates in every split.
- **No answer leak**: labels are pre-shifted (`ar/inputs.py:119-120`), attention is causal with right padding (`yessir/blocks.py`, `yessir/batch.py`), and the metadata on an input token describes that token, not the one being predicted.
- **Loss mask**: only the trace and the final `[EOS]` are supervised; prompt and padding are `-100`.
- **Exact match really is exact**: with or without query tokens, a problem passes only if every supervised prediction including the final `[EOS]` is right.
- **Config**: every field of `AdditionArConfig` reaches the dataset, the model or the evaluator. `step_order` and `use_query_tokens` are passed to both training and evaluation.
- **Eval mode and state**: `model.eval()` is called, there is no dropout or cached state, and evaluation does not touch the training RNG.

### What I ran

- `./yessir.sh test`: 360 passed.
- A scratch check (not in the repo) that trains a tiny model on bases 2 and 3, lengths 1 to 3, and at 8 checkpoints compares, problem by problem, teacher-forced exact match against `rollout_ar` on 150 problems. All four `(step_order, use_query_tokens)` settings, exact match between 5% and 60% at the checkpoints: **0 disagreements in 4,800 comparisons**.

### What I did not get to

- The equivalence check used `token_type_pe` only, on CPU, with short sequences. I did not test `nope` or `token_type_nope`, GPU, or lengths near 40.
- No full-size training run, so nothing here says whether `ar` learns the task.
- 200 problems per split is a statistical question for the inference reviewer; I did not assess it.
- `web/`, `yessir/vertex.py` and the TUI beyond the one function named in Part B.

## Part B: check of `review-code-addition-ar.md`

I read it in full.

**Correct, and I confirmed independently**

- Finding 1 (premature `[EOS]` taxonomy): correct; same as my finding 3. Its worked numbers are consistent with the code.
- Finding 2 (teacher-forced secondary metrics): correct, including the point that `halt_accuracy` is read after the reference final carry.
- Finding 3 (`bfloat16` wording, `[F]` in `query-tokens.json`): correct. There is no autocast anywhere in `yessir/` or `experiments/`.
- Finding 4 (TUI float stepping): correct. `yessir/tui.py:142` returns `max(step_f, next_f)`, so a float field cannot be stepped down to 0.
- "29 fields": correct (I counted 29 on `AdditionArConfig`).
- Its test-gap list is accurate; the cited test does assert `out_of_alphabet` for a premature `[EOS]`.

**Where I disagree or would qualify**

- It says the premature-`[EOS]` behaviour "contradicts" the spec. The sibling review of `addressed_ar` says the same behaviour "matches" the spec. The spec supports both readings; the two reviews should not both stand as written.
- It states teacher-forced and free-running exact match are "mathematically equivalent" and lists the lack of an empirical check as a test gap. That gap is now partly closed by my scratch check, but there is still no test in the repo.

**Missing**

- My finding 1: under the default mode the harness supplies the column index to a variant whose stated point is that nothing external tracks state. The review verifies the position ids match the spec table but does not ask what that does to the `ar` versus `step_ar` comparison.
- It says train and eval are "structurally disjoint" for the OOD splits but does not say that the `id` split is an independent sample, not a held-out set. At defaults that is harmless (0 of 200 seen); with `p_prior=1.0` I measured 6 of 200, and in a small symbol universe 107 of 200.
