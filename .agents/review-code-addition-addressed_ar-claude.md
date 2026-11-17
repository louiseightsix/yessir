# Code review: `addition` / `addressed_ar`

- **Persona**: `.agents/reviewer-code.md`.
- **Code reviewed**: commit `bc2244b` (the working tree at `89bdcb2` differs only by files under `.agents/`).
- **Runs**: none. `results/addition/` does not exist and every addition study is `draft` with `runs: []`.
- **Order of work**: own reading and checks first, then `.agents/review-code-addition-addressed_ar.md` (Part B).

## Verdict

No bug found that would make `exact_match` wrong. This is the variant where the secondary headline is least trustworthy: `step_accuracy` can read 100% for a model that cannot produce a single correct address on its own (finding 1). Under the default embedding mode the variant also does not test what its spec says it tests (finding 2), and the length-extrapolation split is confounded by address digits never seen in training (finding 3).

## Part A: my review

### Findings

**1. `step_accuracy` ignores wrong addresses and is computed with the reference address in the input. High for `step_accuracy` (written to `eval.json`); none for `exact_match`.**

- Where: `variants/addressed_ar/evaluation.py:71-91` feeds the reference trace; `evaluation.py:135-137` records a wrong `[IDX]` or `addr(k)` only in `syntax_ok`; `experiments/addition/evaluation.py:193-204` counts a step as correct from the digit and carry alone.
- Concrete case (I ran the scorer): one column, address wrong, digit and carry right gives `exact_match = 0.0`, `step_accuracy = 1.0`, `digit_accuracy = 1.0`, `halt_accuracy = 1.0`.
- Why it matters: the point of `addressed_ar`, as opposed to `addressed_step_ar`, is that the model writes `addr(k)` itself. A model whose address counter fails completely beyond column 20 still gets the correct address handed to it before predicting each digit, and a wrong address is not counted against the step. On `extrap_len`, `extrap_len_step_accuracy` would then measure what `addressed_step_ar` measures.
- Fix direction, for discussion only: either require `syntax_ok` for a correct step, or report a separate address accuracy.

**2. Under `token_type_pe` (the default) the address can be copied by position, so address counting is not tested. Medium; affects what the headline means.**

- Where: `variants/addressed_ar/inputs.py:89-115` tags the trace `[IDX]` and each address digit with `position_id = k`; `experiments/addition/inputs.py:259-274` tags the operand entries the same way. `config.py:111` makes `token_type_pe` the default for this variant too.
- What happens: when the model has to emit `addr(k)`, the `[IDX]` it reads carries position `k`, and operand entry `k` carries the same position together with the digits of `addr(k)`. For every column that exists in an operand, the address can be looked up by position and copied. The digits `a_k` and `b_k` can be fetched the same way without reading any address.
- Why it matters: the spec's stated reason for this variant is to replace sinusoidal coordinates with content-based address matching. With the default config both are present, so an in-distribution result does not show that addresses are used, and a failure on `extrap_len` cannot be attributed to addresses, since positions 21 to 41 are also unseen sinusoids.
- In rollout the harness assigns these position ids and token types from the expected address width of each column (`inputs.py:110-115`), so the "no harness" property holds for tokens but not for metadata. If the model emits an address of the wrong width, later metadata is misaligned; that only happens after a problem has already failed.

**3. `extrap_len` introduces address digits and widths that never occur in training. Medium; a confound in the headline split.**

- I computed which leading address digits appear for columns up to 20 and which appear only for columns 21 to 40:

| Base | Leading digits seen in training | Seen only on `extrap_len` |
| :--- | :--- | :--- |
| 2 | 1 (addresses up to 5 digits) | none, but 6-digit addresses are new |
| 4 | 1, 2, 3 | none |
| 8 | 1, 2 | 3, 4, 5 |
| 10 | 1, 2 | 3, 4 |

- In base 10 the leading digit 2 is seen only at column 20, which exists only in the trace on an overflow, never on an operand digit. In base 8 it does appear on operand columns 16 to 19.
- Consequence: an `extrap_len` exact match pooled over bases 2, 4, 8, 10 mixes one base where the address format is fully covered (4) with three where it is not. `.agents/addition.md` already names this confounder; the code has no per-base breakdown to separate it.

**4. Zero-preamble mode changes the length distribution per base. Low.**

- Where: `inputs.py:431`. With `include_alphabet=False`, lengths start at the base. I sampled 5,000 problems: minimum length is 2, 4, 8, 10 for bases 2, 4, 8, 10. A comparison of `include_alphabet` on and off is therefore also a comparison of short against long problems.

**5. Premature `[EOS]` in a value slot is `out_of_alphabet`. Low.** Same mechanism as in `ar`; see my `ar` review, finding 3.

### What I read and found correct

- **Reference**: as in my `ar` review (hand cases plus 20,000 random problems against Python integers, 0 mismatches).
- **Address encoding** (`inputs.py:115-138`): least-significant digit first, in the problem's own alphabet, minimal width. Matches the spec examples.
- **Layout and labels** (`variants/addressed_ar/inputs.py`): trace is `[IDX] addr(k)` then the two values (four tokens with queries); labels are pre-shifted; `[IDX]`, the address, the queries, the values and `[EOS]` are all supervised.
- **Evaluation indexing** (`evaluation.py:120-162`): column offsets use the correct per-column address width, and an assertion checks the offsets add up.
- **Exact match** requires every address token, every `[IDX]`, every value and the final `[EOS]`.
- **Zero-preamble validation**: `config.py:488-505` and `inputs.py:160-166` enforce lengths of at least the base for training and OOD bases, and `train_and_evaluate.py:171` passes `include_alphabet` to the evaluation sampler.
- **Config**: all 30 fields reach a consumer.

### What I ran

- `./yessir.sh test`: 360 passed.
- Scratch check of teacher-forced exact match against `rollout_addressed_ar`, problem by problem, on a partially trained tiny model (bases 2 and 3, lengths 1 to 3, 150 problems, 8 checkpoints, all four `(step_order, use_query_tokens)` settings, exact match between 0% and 57%): **0 disagreements in 4,800 comparisons**.

### What I did not get to

- The equivalence check covers `token_type_pe`, `include_alphabet=True`, CPU and short sequences only. Lengths up to 3 mean almost no multi-digit addresses outside base 2.
- No full-size run; memory use at length 40 in base 2 not measured.
- `parse_addressed_ar_trace` is unused by any run; I read it but did not test it.

## Part B: check of `review-code-addition-addressed_ar.md`

I read it in full.

**Correct, and I confirmed independently**

- Finding 1 (oracle address plus `syntax_ok`-blind step scoring): correct and the most important point for this variant. Same as my finding 1; I reproduced the metric values.
- Finding 2 (rollout metadata follows the expected width; the parser accepts special tokens): correct from the code. It correctly says no recorded artefact depends on it.
- Finding 3 (token-weighted loss; about 14 times more transitions per step than the step variants): correct as a design note.
- Finding 4.1 (`halt_accuracy` treats a non-`[IDX]` token between columns as a halting error, but not before column 0): correct from `evaluation.py:149-151`.
- Finding 5 (split-seed collision for adjacent OOD bases): the collision is real. I reproduced both splits getting seed 2003.

**Where I disagree or would qualify**

- Section 3.2 says the trace is generated with "zero harness injection". True for tokens, not for metadata: position ids and token types of generated tokens come from the harness (my finding 2). That is the gap in this review.
- Finding 4.2 says classifying a premature `[EOS]` as `out_of_alphabet` matches the spec; the `ar` review says it contradicts the spec. One of the two needs changing.
- Finding 5 says the colliding splits "reuse the same sequence" of draws. Only the first problem's alphabet prefix coincides; after that the streams diverge because the two bases consume different amounts of randomness (I measured 1 of 200 problems sharing a prefix). The practical impact is nil either way.

**Missing**

- The default-mode confound (my finding 2). The `addressed_step_ar` review raises it for that variant; this review does not, although it applies more strongly here.
- The per-base picture of unseen address digits (my finding 3). The review mentions the confounder only through its worked example.
