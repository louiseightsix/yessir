# Code review: `addition` / `step_ar`

- **Persona**: `.agents/reviewer-code.md`.
- **Code reviewed**: commit `bc2244b` (the working tree at `89bdcb2` differs only by files under `.agents/`).
- **Runs**: none. `results/addition/` does not exist and every addition study is `draft` with `runs: []`.
- **Order of work**: own reading and checks first, then `.agents/review-code-addition-step_ar.md` (Part B).

## Verdict

No bug found that would make `exact_match` wrong. The label and offset arithmetic, which is where I expected an off-by-one, is right in all four `(step_order, use_query_tokens)` settings. The caveats: secondary metrics are conditional on reference tokens both across and within a step (finding 1), which makes `halt_accuracy` and `digit_accuracy` not comparable with `cls` and `step_fork` (finding 2), and the harness supplies the column index (finding 3).

## Part A: my review

### Findings

**1. Evaluation feeds reference tokens across columns and within a column. Medium; moves `step_accuracy` in `eval.json`.**

- Where: `variants/step_ar/evaluation.py:59-77` builds each step from the reference carry, reference history and reference `(t_0, t_1)`; `evaluation.py:98-106` reads all three predictions from that single pass. `rollout_step_ar` is never called by a run.
- The persona asks for the model's own state to be fed forward. The spec chooses teacher forcing and argues `exact_match` is unaffected; I tested that (see "What I ran") and it held.
- Across columns: after a wrong carry, the next column gets the right one. Concrete case: `99999 + 1`, model fails only the first carry; free-running, five of six columns are wrong; teacher-forced `step_accuracy` is 5/6.
- A step counts as correct only if both values are right, so within-step forcing cannot inflate `step_accuracy` itself. It does inflate the two metrics in finding 2.

**2. `halt_accuracy` and `digit_accuracy` are easier here than in `cls` or `step_fork`. Medium for cross-variant and step-order comparisons.**

- Where: `evaluation.py:104-106` reads the flag at the last position, after the reference `t_0` and `t_1`.
- Halting at the last operand column depends on whether the carry out is zero. In `cls` and `step_fork` the model must work that out to predict the flag. Here the flag is predicted after the reference carry has been placed in the input, so a model that gets the carry wrong can still be scored as halting correctly.
- Under `carry_first` the digit is likewise predicted after the reference carry.
- Concrete case: `7 + 5` in base 10 under `carry_first`, model predicts carry `0`. Free-running, the digit and the flag are then predicted from that wrong carry. Teacher-forced, they are predicted from the reference carry `1`, so both can be scored correct although the rollout would have gone wrong.
- Consequence: a `halt_accuracy` gap between `step_ar` and `cls`, or a `digit_accuracy` gap between `carry_first` and `sum_first`, partly reflects the evaluation and not the model.

**3. Under `token_type_pe` (the default) the harness supplies the column index. Low to medium; affects what the headline means.**

- Where: `variants/step_ar/inputs.py:82-88`. The carry token, the step delimiter or query tokens, and `o_k` all get `position_id = k`.
- As with `cls`, the "column indexing" part of the task is done by the harness in this mode. By design, but it belongs next to any claim about alignment. `.agents/addition.md` reports the prototype's `step_ar` stalling under NoPE because of "cold counting"; that difficulty is exactly what the default mode removes.

**4. `p_terminal` is ignored when `all_steps=True`, and `all_steps` batches are correlated. Low.** Same as my `cls` review, finding 4.

### What I read and found correct

- **Reference**: hand cases and 20,000 random problems against Python integers, 0 mismatches (details in my `ar` review).
- **Layout** (`variants/step_ar/inputs.py:69-105`): `c_k [SEP] o_0..o_{k-1}` then `[SEP]` or `Q_0`, then the step tokens. History positions are `0..k-1` (unlike `cls`), delimiter and queries at `k`, `o_k` at `k`, `c_{k+1}` at `k + 1`. Matches the spec table.
- **Labels** (`inputs.py:147-158`): without queries the last three labels are `(t_0, t_1, flag)` against inputs `[SEP] t_0 t_1`. With queries the last five are `(t_0, ignore, t_1, ignore, flag)` against inputs `Q_0 t_0 Q_1 t_1 Q_2`. Each label is one position ahead of the token it names. Query tokens are never targets.
- **Evaluation offsets** (`evaluation.py:82`): `(-3, -2, -1)` and `(-5, -3, -1)` from each row's own length pick exactly those label positions.
- **Order handling**: `pack_step_order` and `unpack_step_order` are inverses, and the same order is passed to training, evaluation and the first-error classifier.
- **Rollout** (`generation.py`): three passes per column, each prompt a prefix of the training sequence; feeds back its own values; stops on its own `FLAG_FINISHED`; never calls the reference.
- **Exact match** requires digit, carry and flag at every step.
- **Config**: all 31 fields reach a consumer, with the `p_terminal` exception.

### What I ran

- `./yessir.sh test`: 360 passed.
- Scratch check of teacher-forced exact match against `rollout_step_ar`, problem by problem, on a partially trained tiny model (bases 2 and 3, lengths 1 to 3, 150 problems, 8 checkpoints, all four settings, exact match between 3% and 47%): **0 disagreements in 4,800 comparisons**.

### What I did not get to

- The equivalence check covers `token_type_pe`, CPU and short sequences only. The NoPE modes, where the spec's argument about the step delimiter matters most, are untested by me.
- No full-size run.

## Part B: check of `review-code-addition-step_ar.md`

I read it in full.

**Correct, and I confirmed independently**

- Finding 1 (teacher forcing at two levels; the `carry_first` effect on `digit_accuracy`; the halting effect relative to `cls` and `step_fork`): correct, and well argued. Same as my findings 1 and 2. It correctly notes that `step_accuracy` is immune to within-step forcing.
- Finding 2 (base 2 pooled into the `permuted` split): correct, and not something I had found. In base 2 every in-alphabet error is counted as `prior_intrusion`, and the split has no per-base breakdown. Its chance figure of 39.7% is right: (1 + 1/3 + 1/7 + 1/9) / 4.
- Finding 3 (`fixed_d_model` adds width back when `d_scratch=0`): correct. I confirmed `nope` with `d_scratch=0` gives 48 scratch dimensions and `d_model = 304`.
- Finding 4 (split-seed bump): the collision is real, and the second half is a good catch: a training seed of 2002, 2011, 2999 or 3999 silently moves one split to different problems, which would break a seed sweep that assumes a fixed evaluation set.
- Finding 5 (rollout treats any non-`FINISHED` token as continue; `bfloat16` wording; hex display): correct.
- The layout table in section 3.2 matches the code.

**Where I would qualify**

- The review says the exact-match equivalence was "confirmed empirically across all 12 combinations" without saying on what model. If the model was untrained or converged, the check shows little. Mine used partially trained models but only one embedding mode.
- Finding 4 describes the colliding splits as having "correlated uniform draws". I measured the overlap for the same configuration: 1 of 200 problems shares an alphabet prefix. The correlation is negligible in practice.

**Missing**

- My finding 3: the harness supplies the column index under the default mode. The review confirms the position ids match the spec but does not say what that removes from the task.
- The batch correlation under `all_steps` (it covers only the `p_terminal` no-op).
