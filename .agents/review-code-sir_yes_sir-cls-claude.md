# Code review: `sir_yes_sir` / `cls`

- **Persona**: `.agents/reviewer-code.md`.
- **Code reviewed**: commit `bc2244b` (the working tree at `89bdcb2` differs only by files under `.agents/`).
- **Runs**: there are no `cls` runs. Six runs of its predecessor `sm_mh` exist under `results/sir_yes_sir/` (from `a13cc88`, `586f44d`, `86e46ea`), all marked `OUTDATED: built against draft spec`. All 8 `studies/sir_yes_sir/*.json` are `draft` with `runs: []`. I reviewed the current code.
- **Order of work**: own reading and checks first, then `.agents/review-code-sir_yes_sir-cls.md` (Part B).

## Verdict

No bug found that would make `exact_match` wrong. The code matches its spec. As with `ar`, a result supports less than the metric names suggest: halting is a constant (finding 1), small problem spaces are not held out (finding 2), and one supervision mode changes how many problems are seen as well as how they are supervised (finding 3).

## Part A: my review

### Findings

**1. Every problem in a run has the same length, so the flag head learns a constant. Medium; `halt_accuracy` and the halting part of `exact_match` carry no information.**

- Where: `config.py:74` (`x_length` is one integer); `variants/cls/run.py:26-43` passes it to both datasets.
- What happens: the flag is `FINISHED` at step `L - 1` and `CONTINUE` before it, in every problem. Under `token_type_pe` the query token for step `k` carries `position_id = k`, so the flag is a function of position alone.
- Evidence (my scratch run, `B = 4`, `L = 3`, `token_type_pe`): `halt_accuracy` was 100% at every checkpoint from step 100 on, while `exact_match` never exceeded 2%.
- Why it matters: the `cls-vs-ar` study asks about "halting reliability" and `sequence-length` about "halting at `k == L - 1`". Neither can be assessed while `L` is fixed within a run.

**2. With a small symbol pool or a fixed alphabet, the evaluation problems have been trained on. Medium for those regimes.**

- Where: `variants/cls/run.py:35-43`. Evaluation uses the same generator with a different seed.
- The problem space is `B^L` with `resample_alphabet=False` (64 at defaults) and 54 for pool 3, `B = 3`, `L = 2`, the configuration the convergence test and three of the outdated runs use. At the default full pool it is 207,068,682,240 and overlap does not arise.
- In those regimes `exact_match` is accuracy on seen problems.

**3. `supervision=all_steps` sees one `L`-th as many problems per batch, all correlated. Medium for the `cls-training-passes` study.**

- Where: `variants/cls/dataset.py:73-80` yields the `L` steps of a problem consecutively; `yessir/training.py:302-308` takes batches in order.
- At `batch_size = 64`, `L = 3`: `single_pass` and `random_step` see 64 independent problems per step, `all_steps` sees about 21, each three times. `single_pass` also supervises `L` positions per example where the other two supervise one.
- Consequence: comparing the three modes at equal `max_steps` and `batch_size` changes three things at once (problems seen, targets per step, within-batch correlation). The study needs to say which of these it holds fixed.

**4. The runner has no split that could show memorisation, wrap-around failure or length generalisation. Medium; a gap before those studies can run.** Same as my `ar` review, finding 3. `evaluate_cls` reports one pooled `step_accuracy` (`variants/cls/evaluation.py:112-129`), and evaluation always uses the training configuration.

**5. `step_accuracy` is conditional on the reference outputs, and ignores a premature halt. Low.** The digit head and the flag head are read independently at every step, so a model that predicts `FINISHED` at step 0 still has all `L` digits scored. In `ar` the same mistake costs one step. `step_accuracy` is therefore not like for like between the two variants on unconverged models. `exact_match` is unaffected.

### What I read and found correct

- **Reference and sampling**: as in my `ar` review.
- **Single-pass labels** (`variants/cls/inputs.py:33-48`): the example holds outputs `o_0..o_{L-2}`, and the `L` label pairs sit on the last `L` positions, so step `k` is read from the second `[SEP]` (`k = 0`) or from `o_{k-1}`. No step's target is in its own input.
- **Single-step labels** (`inputs.py:66-75`): input ends at the query token for step `k`; label is `(o_k, flag_k)`.
- **Same hidden state in both modes**: per-position reading with `lengths=None` and last-position reading with `lengths` use the same norm and linear layers (`yessir/heads.py:98-112`), and under the causal mask the state at a query position does not depend on later tokens.
- **Evaluation** (`variants/cls/evaluation.py:112-129`): predictions are taken at the supervised positions; exact match needs every digit and every flag; evaluation always uses single-pass examples whatever the training supervision was (`run.py:35-43`).
- **Step RNG** (`dataset.py:59`): seeded with `seed + 1`, separate from the problem stream, so `random_step` does not change which problems are drawn.
- **Config**: all 21 fields reach a consumer.
- **Eval mode and state**: `model.eval()` is called; no dropout, no state between batches.

### What I ran

- `./yessir.sh test`: 360 passed.
- Scratch check of teacher-forced exact match against `greedy_generate_cls`, problem by problem, on a partially trained tiny model (`B = 4`, `L = 3`, pool of 8, 150 problems, 8 checkpoints, all three embedding modes): **0 disagreements in 3,600 comparisons**. Exact match ranged from 0% to 12%, so this mostly exercises problems that fail early.

### What I did not get to

- I trained only with `single_pass` in the scratch check; `all_steps` and `random_step` training are covered by my reading and the existing unit tests, not by a run of mine.
- The historical `sm_mh` code beyond confirming it is a different design (three heads, a `[STEP]` token, per-step halting metric).
- No full-size run; GPU; `web/`; `yessir/vertex.py`.

## Part B: check of `review-code-sir_yes_sir-cls.md`

I read the findings and test-coverage sections in full and skimmed the "verified correct" sections.

**Correct, and I confirmed independently**

- Section 1 (provenance): correct. No `cls` runs; six `sm_mh` runs, all `OUTDATED`; the `direct-ghost` run does show `exact_match = 0.0` with `halt_accuracy = 0.9967`.
- Finding 1 (studies the runner cannot serve): correct. It also states the fixed-length halting shortcut, in one sentence, which is my finding 1.
- Finding 2 (`all_steps` confound): correct; same as my finding 3. The `L = 64`, `batch_size = 64` example (one problem per batch) follows from the code.
- Finding 3 (flag head needs exact `CONTINUE` in evaluation but generation continues on any non-`FINISHED` value; premature halt does not reduce `step_accuracy`; extra head parameters): correct. I confirmed the parameter counts, 19,241,472 for `cls` and 18,978,816 for `ar` at defaults.
- Finding 4.2 and 4.3 (`symbol_pool_size` unused when `resample_alphabet=False`; small-universe overlap): correct.
- Finding 5 (doc drift and unreachable validation gaps): correct from the code as far as I checked (`bfloat16` wording, `lr` not checked for NaN).

**Errors**

- Finding 4.1 says that with `fixed_d_model=True` and `d_position=0`, `d_model` becomes 480 under `nope` and **496** under `token_type_nope`. I computed both from the config: both are **480**. The point it makes (passing `d_position=0` shrinks the model despite `fixed_d_model`) is right; the second number is wrong.
- Finding 4.3 gives the default problem space as 207,055,380,480. It is 207,068,682,240. The `ar` review gives a third, also wrong, figure. The conclusion is unaffected.

**Where I would qualify**

- The fixed-length halting shortcut is buried inside finding 1 as a side remark. It affects every run's `halt_accuracy` and deserves its own finding.
- The review says exact-match equivalence was "verified analytically and empirically below" but the only empirical evidence it cites is a scripted model in an existing test. My scratch check on trained models agrees, with the limitation noted above.

**Missing**

- Nothing that changes the verdict.
