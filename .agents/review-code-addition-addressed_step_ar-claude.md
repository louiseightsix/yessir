# Code review: `addition` / `addressed_step_ar`

- **Persona**: `.agents/reviewer-code.md`.
- **Code reviewed**: commit `bc2244b` (the working tree at `89bdcb2` differs only by files under `.agents/`).
- **Runs**: none. `results/addition/` does not exist and every addition study is `draft` with `runs: []`.
- **Order of work**: own reading and checks first, then `.agents/review-code-addition-addressed_step_ar.md` (Part B).

## Verdict

No bug found that would make `exact_match` wrong. The main risk is the default configuration: with `token_type_pe` the model does not need the addresses at all (finding 1). The secondary metrics are conditional on reference state (finding 2), and two distribution effects confound the zero-preamble and length-extrapolation comparisons (findings 3 and 4).

## Part A: my review

### Findings

**1. With the default `token_type_pe`, column `k` is given by position id on every entry, so the addresses are redundant. Medium; affects what the headline means.**

- Where: `config.py:111` (default inherited by this variant); `experiments/addition/inputs.py:259-274` puts `position_id = j` on `[IDX]`, on each digit of `addr(j)` and on the operand digit; `variants/addressed_step_ar/inputs.py:79-96` does the same for the carry entry, the output history and the step-`k` hint.
- Concrete case: spec Example 1 at `k = 2`. The prompt ends `[IDX] 2` at position 2; `a_2 = '9'` sits at `(OPERAND_A, 2)`. A model can fetch `a_2` by position and never read an address.
- Why it matters: the spec's reason for this variant is content-based address matching in place of positions. A run at default settings cannot show that. In zero-preamble mode it is stronger: the single-digit address `s_j` sits at `(ADDRESS, j)`, so the position id hands the model the rank of each symbol, which is the thing the mode is supposed to make it infer.
- This follows the spec's layout table, so it is a default to choose deliberately, not a code defect. Any study that calls `addressed_step_ar` "coordinate-free" must set `embedding_mode` to `nope` or `token_type_nope`.

**2. `step_accuracy`, `digit_accuracy`, `carry_accuracy` and `halt_accuracy` are conditional on reference state. Medium; moves `step_accuracy` in `eval.json`.**

- Where: `variants/addressed_step_ar/evaluation.py:60-79`. Every step is built from the reference carry and reference output history, with the reference `(t_0, t_1)` already in place.
- Across columns: an error in column `k` is repaired before column `k + 1`.
- Within a column: the flag is read after the reference carry, and under `carry_first` the digit is read after the reference carry. So `halt_accuracy` here is easier than in `cls` or `step_fork`, where the flag is predicted without seeing any carry, and `digit_accuracy` favours `carry_first`.
- Headline impact: none on `exact_match` (checked, see "What I ran").

**3. Zero-preamble mode ties operand length to the base. Low to medium for the `zero-preamble` study.**

- Where: `inputs.py:431`, `eff_min = max(min_length, base)`.
- I sampled 5,000 zero-preamble problems at defaults: minimum length is 2, 4, 8, 10 for bases 2, 4, 8, 10. An `ood_base12` split would use lengths 12 to 20 only.
- Consequence: in this mode a large base always comes with long operands. A drop on `ood_base12` cannot be separated from a length effect, and turning `include_alphabet` off also changes the difficulty of the `id` split.

**4. `extrap_len` contains address digits and widths never seen in training. Medium; a confound in a headline split.**

- Leading address digits for columns up to 20, and those that appear only for columns 21 to 40: base 2 sees addresses up to 5 digits and meets 6-digit ones; base 4 is fully covered; base 8 meets 3, 4, 5 for the first time; base 10 meets 3 and 4. The digit 2 in bases 8 and 10 appears in training only at column 20, on an overflow step.
- The pooled `extrap_len` number therefore mixes covered and uncovered bases with no per-base breakdown.

**5. `p_terminal` is ignored when `all_steps=True`. Low.** `inputs.py:537-539`. Both values are still written to `config.json`.

### What I read and found correct

- **Reference**: hand cases and 20,000 random problems against Python integers, 0 mismatches (details in my `ar` review).
- **Layout** (`variants/addressed_step_ar/inputs.py:72-116`): `[IDX] addr(k) c_k [SEP]`, addressed output history, `[SEP]`, `[IDX] addr(k)`, then the step tokens. Matches the spec, including Example 2 in zero-preamble mode.
- **Labels** (`inputs.py:160-171`): three targets per step; with query tokens the label pattern is `(t_0, ignore, t_1, ignore, flag)`, which lines up with the input `Q_0 t_0 Q_1 t_1 Q_2`. Query tokens are never targets.
- **Evaluation offsets** (`evaluation.py:84-108`): `(-3, -2, -1)` and `(-5, -3, -1)` relative to each row's own length are the right positions.
- **No answer leak**: the flag never appears in the input, and each value is predicted from a position before it.
- **Rollout** (`generation.py`): starts from `s_0` and an empty history, feeds back its own carry and digit, stops on its own `FLAG_FINISHED`, never calls the reference.
- **Zero-preamble validation**: lengths of at least the base are enforced in the sampler, the builder and the config, for OOD bases too.
- **Config**: all 32 fields reach a consumer (with the `p_terminal` exception above).

### What I ran

- `./yessir.sh test`: 360 passed.
- Scratch check of teacher-forced exact match against `rollout_addressed_step_ar`, problem by problem, on a partially trained tiny model (bases 2 and 3, lengths 1 to 3, 150 problems, 8 checkpoints, all four `(step_order, use_query_tokens)` settings, exact match between 5% and 49%): **0 disagreements in 4,800 comparisons**.

### What I did not get to

- The equivalence check covers `token_type_pe`, `include_alphabet=True`, CPU, short sequences.
- No full-size run. Evaluation builds every step of every problem before batching; I did not measure memory at length 40.

## Part B: check of `review-code-addition-addressed_step_ar.md`

I read it in full.

**Correct, and I confirmed independently**

- Finding 1 (teacher-forced secondary metrics, with the `carry_first` and halting asymmetries): correct. Its observation that single-step `step_accuracy` is unaffected by within-step teacher forcing is right, since a wrong first token already fails the step.
- Finding 2 (default `token_type_pe`): correct and, in my view, the most important point for this variant. Same as my finding 1, including the zero-preamble consequence.
- Finding 3 (length clamping in zero-preamble mode; unseen address digits): correct. My sampled minima and my table of leading digits agree with it.
- Finding 4 (`bfloat16` wording; evaluation materialises all steps; inspection CLI prints addition tokens as hex): correct.

**Errors**

- Section 2.3 is titled "Complete 30-field config trace" but lists 32 names, and `AdditionAddressedStepArConfig` has 32 fields. The trace itself is right; the count is wrong.

**Where I would qualify**

- Finding 2 is rated "Low to Medium". Since it decides whether the variant tests its own premise, I would rate it at least as high as finding 1.
- The claim that exact match equals rollout was "verified empirically across all 8 combinations" but the review does not say on what model. On an untrained or fully trained model the check is trivial. Mine used partially trained models, but only four of those eight combinations.

**Missing**

- Nothing that changes the verdict. It does not note that the `id` split is an independent sample and not a held-out set (harmless at defaults: 0 of 200 found in the first 200,000 training problems).
