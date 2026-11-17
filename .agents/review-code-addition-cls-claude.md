# Code review: `addition` / `cls`

- **Persona**: `.agents/reviewer-code.md`.
- **Code reviewed**: commit `bc2244b` (the working tree at `89bdcb2` differs only by files under `.agents/`).
- **Runs**: none. `results/addition/` does not exist and every addition study is `draft` with `runs: []`.
- **Order of work**: own reading and checks first, then `.agents/review-code-addition-cls.md` (Part B).

## Verdict

No bug found that would make `exact_match` wrong. `cls` is the simplest variant and the code matches its spec. The caveats are about interpretation: secondary metrics are conditional on reference state (finding 1), the harness tells the model which column it is on (finding 2), and two config combinations can quietly produce a misleading run (findings 3 and 4).

## Part A: my review

### Findings

**1. `step_accuracy`, `digit_accuracy`, `carry_accuracy` and `halt_accuracy` are computed with the reference carry and history at every step. Medium; moves `step_accuracy` in `eval.json`.**

- Where: `variants/cls/evaluation.py:52-66`. `rollout_cls` is never called by a run.
- The persona asks that step `k + 1` receive the model's own carry and that evaluation stop on the model's own halt. The spec chooses teacher forcing instead and argues `exact_match` is unaffected. I tested that claim (see "What I ran"); it held.
- For the other metrics it does not hold. Concrete case: `9995 + 5` in base 10, model fails only the first carry. Free-running, every later column is wrong. Teacher-forced, columns 1 to 4 are scored with the correct carry and `step_accuracy` is 0.8. A model that predicts `FLAG_FINISHED` at column 0 still has all later columns scored.
- These numbers mean "single-step accuracy from a correct state". They should not be read as how far a rollout gets.

**2. Under `token_type_pe` (the default) the harness supplies the column index. Low to medium; affects what the headline means.**

- Where: `variants/cls/inputs.py:53`. The carry token gets `position_id = k` and the last history token also sits at `k`.
- The task spec lists "column indexing" as one of four things the model must do. In this mode the harness does the counting and the model only has to match equal position ids. That is the documented design of the step variants, so it is not a defect, but a statement such as "the model aligns columns" needs the embedding mode attached.

**3. With `p_prior=1.0` and the default `eval_permuted_canonical=False`, no split tests in-context reading. Medium for the `prior-override` study.**

- Where: `config.py:158`, `config.py:358-373`.
- The `id` and `extrap_len` splits inherit `p_prior`, so they use only the ascending canonical alphabet. A model that ignores the alphabet section and has memorised digit values would score 100% on both. Only the OOD-base splits, which force random alphabets, would show the problem, and they change the base at the same time.
- I also measured overlap at `p_prior=1.0`: 6 of 200 `id` problems appear in the first 200,000 training problems. Small, but not zero as it is at defaults.

**4. `all_steps=True` gives highly correlated batches and ignores `p_terminal`. Low; relevant to the `step-sampling` study.**

- Where: `experiments/addition/inputs.py:537-539`.
- All steps of a problem are yielded consecutively and batches are taken in order, so a batch of 64 holds about 5 problems at default lengths, against 64 independent problems otherwise. Comparing the two modes at equal `max_steps` changes the number of distinct problems seen by about 14 times.
- `p_terminal` has no effect in this mode but is still recorded in `config.json`.

**5. Reusing an output directory appends to `train.jsonl` and `eval.jsonl`. Low; provenance.** `yessir/metrics.py:40` opens in append mode, while `config.json` and `eval.json` are overwritten. Automatic run names avoid this; an explicit `--out-dir` does not.

### What I read and found correct

- **Reference**: hand cases and 20,000 random problems against Python integers, 0 mismatches (details in my `ar` review).
- **Sampling and splits**: as in my `ar` review; 0 duplicates between the default splits and the first 200,000 training problems.
- **Step sampling** (`inputs.py:498-549`): a separate RNG picks the step, so changing `p_terminal` does not change the problems. At defaults I measured 27.2% terminal steps over 20,000 samples, consistent with 20% forced plus the uniform share.
- **Layout** (`variants/cls/inputs.py:40-55`): `c_k [SEP] o_0 .. o_{k-1}` with history positions `1..k`, as the spec says.
- **No answer leak**: the three targets of step `k` do not appear in the step-`k` input.
- **Head and loss**: three 256-way heads read the last unpadded position (`yessir/heads.py:102-112`); the loss is the sum of the three cross-entropies. Padding cannot reach the read position under the causal mask.
- **Evaluation regrouping** (`variants/cls/evaluation.py:91-107`): the flat prediction list is cut back into problems with the right step counts.
- **Exact match** requires digit, carry and flag at every step; a premature `FLAG_FINISHED` or a missing final one fails the problem.
- **Config**: all 29 fields reach a consumer, with the `p_terminal` exception above.
- **Eval mode and state**: `model.eval()` is called; no dropout, no state between batches.

### What I ran

- `./yessir.sh test`: 360 passed.
- Scratch check of teacher-forced exact match against `rollout_cls`, problem by problem, on a partially trained tiny model (bases 2 and 3, lengths 1 to 3, 150 problems, 8 checkpoints, exact match between 2% and 27%): **0 disagreements in 1,200 comparisons**.

### What I did not get to

- The equivalence check covers `token_type_pe`, CPU and short sequences only.
- No full-size run.
- 200 problems per split is a question for the inference reviewer.

## Part B: check of `review-code-addition-cls.md`

I read it in full.

**Correct, and I confirmed independently**

- Finding 1 (teacher forcing; `rollout_cls` unused by runs): correct. Its induction argument for `exact_match` is sound and my scratch check agrees with it.
- Finding 2 (silent no-ops): correct. I confirmed that `symbol_set="decimal"` with base 12 and `p_prior=1.0` is accepted.
- Finding 4 (`MetricsLogger` appends): correct, and not something I had found before reading it.
- Finding 5 (`eval_permuted_canonical` default; loss three times larger than the LM variants; `web/` reads only some artefacts): correct. The first item is my finding 3 and deserves more than "informational".
- Section A notes that `extrap_len` excludes one-short-one-long pairs and that `id` problems can recur at `p_prior=1.0`. Both correct.

**Errors**

- Finding 3 says that when two OOD splits share a seed, "every problem `i` in `ood_base3` draws the exact first 3 symbols of problem `i` in `ood_base4`'s alphabet", because choosing from a one-element tuple "consumes zero random bits". I reproduced the configuration: both splits do get seed 2003, but only **1 of 200** problems shares the alphabet prefix. The streams diverge after the first problem because the two bases draw different numbers of values. The collision is real; the stated consequence is not.

**Where I would qualify**

- The review says teacher-forced exact match equals rollout "in exact fp32 arithmetic" and lists the missing empirical comparison as a test gap. It did not run one. Mine found no disagreement, in one embedding mode.

**Missing**

- My finding 2: the harness supplies the column index under the default mode. The review checks that position ids match the spec but does not connect it to the "column indexing" sub-operation the task spec says the model performs.
- My finding 4, first half: the batch correlation under `all_steps`. The review covers the `p_terminal` no-op but not the 14-fold change in distinct problems per batch.
