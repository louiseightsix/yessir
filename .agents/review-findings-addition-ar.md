# Code Correctness Audit — `addition` / `ar`

## 1. Scope & Commit Reviewed

- **Commit audited**: `HEAD` = `bf0f1bd451e3598eef50048a912ab34252579806`.
- **Recorded `results/addition/` runs**: Only one run exists under `results/addition/` (`2026-11-17-1bf576e-merry-bream`, whose [config.json](file:///atelier/code/workshop/yessir/results/addition/2026-11-17-1bf576e-merry-bream/config.json#L1-L4) is for `"variant": "cls"` at `1bf576e`). **No historical runs exist yet for `ar`**, and all 13 study definitions in `studies/addition/*.json` are currently in `"status": "draft"` with `"runs": []`.
- **Specifications & studies reviewed**:
  - [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L1-L177) and [ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/ar.md#L1-L67) (cross-checked against [step_ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/step_ar.md#L1-L113) and [addressed_ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/addressed_ar.md#L1-L95)).
  - All 13 study files under `studies/addition/`: [carry-chain-injection.json](file:///atelier/code/workshop/yessir/studies/addition/carry-chain-injection.json#L1-L8), [embedding-modes-length.json](file:///atelier/code/workshop/yessir/studies/addition/embedding-modes-length.json#L1-L8), [grokking-and-overtraining.json](file:///atelier/code/workshop/yessir/studies/addition/grokking-and-overtraining.json#L1-L8), [harness-vs-capacity.json](file:///atelier/code/workshop/yessir/studies/addition/harness-vs-capacity.json#L1-L8), [minimum-model-size.json](file:///atelier/code/workshop/yessir/studies/addition/minimum-model-size.json#L1-L8), [prior-override.json](file:///atelier/code/workshop/yessir/studies/addition/prior-override.json#L1-L8), [query-tokens.json](file:///atelier/code/workshop/yessir/studies/addition/query-tokens.json#L1-L8), [seed-stability.json](file:///atelier/code/workshop/yessir/studies/addition/seed-stability.json#L1-L8), [step-order.json](file:///atelier/code/workshop/yessir/studies/addition/step-order.json#L1-L8), [step-sampling.json](file:///atelier/code/workshop/yessir/studies/addition/step-sampling.json#L1-L8), [training-bases.json](file:///atelier/code/workshop/yessir/studies/addition/training-bases.json#L1-L8), [variants-comparison.json](file:///atelier/code/workshop/yessir/studies/addition/variants-comparison.json#L1-L8), and [zero-preamble.json](file:///atelier/code/workshop/yessir/studies/addition/zero-preamble.json#L1-L8).
- **Experiment & variant code reviewed**:
  - Shared `addition` modules: [reference.py](file:///atelier/code/workshop/yessir/experiments/addition/reference.py#L1-L186), [inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L1-L640), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L1-L581), [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L1-L288), [train_and_evaluate.py](file:///atelier/code/workshop/yessir/experiments/addition/train_and_evaluate.py#L1-L239), and [run.py](file:///atelier/code/workshop/yessir/experiments/addition/run.py#L1-L36).
  - `ar` variant modules: [inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/inputs.py#L1-L72), [dataset.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/dataset.py#L1-L126), [generation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/generation.py#L1-L54), [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L1-L331), and [run.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/run.py#L1-L54).
- **Shared `yessir/` modules & tests reviewed**:
  - [vocab.py](file:///atelier/code/workshop/yessir/yessir/vocab.py#L1-L48), [special_tokens.py](file:///atelier/code/workshop/yessir/yessir/special_tokens.py#L1-L20), [alphabet.py](file:///atelier/code/workshop/yessir/yessir/alphabet.py#L1-L100), [example.py](file:///atelier/code/workshop/yessir/yessir/example.py#L1-L90), [dataset.py](file:///atelier/code/workshop/yessir/yessir/dataset.py#L1-L82), [batch.py](file:///atelier/code/workshop/yessir/yessir/batch.py#L1-L234), [embedding.py](file:///atelier/code/workshop/yessir/yessir/embedding.py#L1-L156), [blocks.py](file:///atelier/code/workshop/yessir/yessir/blocks.py#L1-L317), [heads.py](file:///atelier/code/workshop/yessir/yessir/heads.py#L1-L116), [model.py](file:///atelier/code/workshop/yessir/yessir/model.py#L1-L175), [training.py](file:///atelier/code/workshop/yessir/yessir/training.py#L1-L331), [metrics.py](file:///atelier/code/workshop/yessir/yessir/metrics.py#L1-L43), [defaults.py](file:///atelier/code/workshop/yessir/yessir/defaults.py#L1-L14), [experiment.py](file:///atelier/code/workshop/yessir/yessir/experiment.py#L1-L478), [tui.py](file:///atelier/code/workshop/yessir/yessir/tui.py#L1-L581), [artefacts.py](file:///atelier/code/workshop/yessir/yessir/artefacts.py#L1-L406), and [inspect_data.py](file:///atelier/code/workshop/yessir/yessir/inspect_data.py#L1-L166).
  - Tests in [test_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_ar.py#L1-L713), [test_config.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_config.py#L1-L524), [test_evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_evaluation.py#L1-L434), [test_inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_inputs.py#L1-L431), [test_reference.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_reference.py#L1-L231), [test_run.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_run.py#L1-L593), [test_convergence.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_convergence.py#L1-L444), and `yessir/tests/`.

---

## 2. Findings (Suspected Bugs, Spec Mismatches, Silent Confounds, or Edge-Case Traps)

**Overall verdict**: **Zero headline-moving bugs were found in `ar`.** The reference addition oracle, full-trace sequence and pre-shifted label construction, causal attention and left-padded `KVCache` rollout, `[EOS]` halting and `exact_match` equivalence, config plumbing, and train/eval RNG isolation all hold up under line-by-line inspection and adversarial verification.

However, the audit uncovered **one Medium-severity diagnostic metric confound** affecting the study design in [prior-override.json](file:///atelier/code/workshop/yessir/studies/addition/prior-override.json#L5), **two Low-severity taxonomy / parser edge-case behaviours**, and **three Notes** on cross-run metric comparisons and edge-case config validation.

---

### [Medium] `prior_intrusion` misses end-to-end canonical-order fallback on `permuted` (scoring below the `1 / (B - 1)` chance baseline in `B = 3, 4` under `sum_first`, and swinging 3.8x under `carry_first`)

- **Where**: [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L85-L109), [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L162-L169), and [prior-override.json](file:///atelier/code/workshop/yessir/studies/addition/prior-override.json#L1-L8).
- **What happens**:
  `_classify_value_mismatch` implements the exact formula written in [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L162):
  ```python
  digit_value = alphabet.index(expected_token)
  if pred_token == SymbolSet.HEX[digit_value]:
      return FirstErrorType.PRIOR_INTRUSION
  ```
  Here `digit_value` is `(val_prompt(a_k) + val_prompt(b_k) + val_prompt(c_k)) % B` (or `// B`), i.e. the arithmetic result computed under the **prompt's** permuted alphabet. Checking `pred_token == SymbolSet.HEX[digit_value]` only detects an *output-only* canonical fallback—a hypothetical model that reads `a_k` and `b_k` using the prompt's permuted alphabet to compute `total_prompt`, and then indexes `SymbolSet.HEX` instead of `alphabet` at unembedding.

  A model trained at `p_prior = 1.0` (where every training prompt has ascending `SymbolSet.HEX[:B]`) need never consult the prompt's `alphabet` permutation to read `a_k` or `b_k` either. When such a model falls back to memorised canonical numeral order on `permuted`, it performs standard canonical ASCII addition `reference_add(SymbolSet.HEX[:B], a, b)`, interpreting **both** input operands and outputs in canonical order (`total_canon = val_canon(a_k) + val_canon(b_k) + val_canon(c_k)`). On most columns `total_canon % B != total_prompt % B`, so `_classify_value_mismatch` labels the prediction as `procedural_error` rather than `prior_intrusion`.
- **Concrete verification**:
  Evaluating a pure canonical-fallback model (`reference_add(SymbolSet.HEX[:B], p.a, p.b)`) across 1,000 `permuted` problems per base (`iter_problems(..., permuted_canonical=True, seed=999)`):
  - **Under `step_order="sum_first"`** (`o_0` checked first):
    - `B = 3`: `exact_match = 0.024`, `prior_intrusion = 0.246`, `procedural_error = 0.726` -> `prior_intrusion` share of in-alphabet value errors is **`0.253`**, which is **half the `1 / (B - 1) = 0.500` chance baseline** (because fixed points `alphabet[v] == HEX[v]` where `total_canon == total_prompt (mod 3)` are non-errors at column `0`, selectively depleting `prior_intrusion` relative to random wrong guesses).
    - `B = 4`: `exact_match = 0.020`, `prior_intrusion = 0.287`, `procedural_error = 0.689` -> share **`0.294`**, again **below** the `0.333` chance baseline.
    - `B = 8`: `exact_match = 0.002`, `prior_intrusion = 0.186`, `procedural_error = 0.809` -> share **`0.187`** (vs `0.143` chance).
    - `B = 10`: `exact_match = 0.005`, `prior_intrusion = 0.128`, `procedural_error = 0.866` -> share **`0.129`** (barely above `0.111` chance).
  - **Under `step_order="carry_first"`** (`c_1` checked first):
    - Because outgoing carry `c_1` is binary (`0` or `1`), roughly half of column-`0` pairs overflow under neither alphabet (`c_1_canon == '0' == HEX[0]` and `v_carry_prompt == 0`), which hits `pred_token == SymbolSet.HEX[0]` whenever `alphabet[0] != '0'`. On the **exact same** canonical-fallback predictions in `B = 10`, `prior_intrusion` jumps **3.8x** from `0.128` (`sum_first`) to **`0.487`** (`carry_first`, share `0.490`).
- **Impact on headline metrics & study claims**:
  - **Headline metrics (`permuted_exact_match`, `permuted_step_accuracy`) are completely unaffected**: a canonical-memorising model drops to `0.2%–2.4%` `permuted_exact_match` as expected.
  - **Study claim impact**: Directly affects the diagnostic analysis planned in [prior-override.json](file:///atelier/code/workshop/yessir/studies/addition/prior-override.json#L5) (*"so `prior_intrusion` can be compared against each base's `1 / (B - 1)` chance share of in-alphabet value errors"*). Comparing `prior_intrusion / (prior_intrusion + procedural_error)` against `1 / (B - 1)` under `sum_first` would falsely conclude that a 100% canonical-memorising model has *no* canonical prior preference in `B = 3, 4` (and almost none in `B = 10`), and comparing `sum_first` vs `carry_first` on `prior_intrusion` is confounded by binary carry value concentration.

---

### [Low] Under `use_query_tokens=True`, intra-column syntax precedence reclassifies `t_0` value errors as `syntax_malformation` whenever wrong `t_0` derails `q_1`

- **Where**: [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L134-L179), [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L135-L155), and [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L160-L165).
- **What happens**:
  When `use_query_tokens=True`, each column `k` in `ar` is generated autoregressively in four steps: `(q_0, t_0, q_1, t_1)` (e.g. `[S], o_k, [C], c_{k+1}` under `sum_first`). In `_parse_ar_rollout`, both query positions are folded into a single per-column flag `syntax_ok = (pred_q0 == int(q_0)) and (pred_q1 == int(q_1))` ([evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L136-L137); and `syntax_ok = False` on `len(col_slice) == 3` `[q_0, t_0, EOS]` at [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L168-L170)). `classify_first_error` then checks `not pred.syntax_ok` **before** checking `t_0` ([evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L136-L152)), matching the column-level rule in [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L160).
- **Concrete input**:
  On `alphabet = b"0123"`, `a = b"1"`, `b = b"1"` (`use_query_tokens=True, step_order="sum_first"`, ground truth `[[S], '2', [C], '0', [EOS]]`), if the rollout emits `[[S], '9', [EOS]]` or `[[S], '9', [F], '0', [EOS]]`:
  - Token `0` (`[S]`) is **correct**.
  - Token `1` (`'9'`) is the **earliest causal error in the rollout** (`out_of_alphabet`).
  - Conditioned on that out-of-distribution `'9'`, token `2` fails to emit `[C]`.
  Because `pred_q1 != [C]` sets `syntax_ok = False` for column `0`, `evaluate_ar` classifies the problem's first error as `syntax_malformation` rather than `out_of_alphabet`.
- **Impact**:
  Code matches [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L160). Does **not** affect `exact_match`, `step_accuracy`, `oracle_step_accuracy`, or `halt_accuracy`. Only shifts first-error taxonomy counts toward `syntax_malformation` on `use_query_tokens=True` runs when an incorrect `t_0` disrupts `q_1`.

---

### [Low] `_parse_ar_rollout` raises `IndexError` on non-`[EOS]`-terminated rollouts with a 1-token trailing column slice

- **Where**: [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L128-L179).
- **What happens**:
  In `_parse_ar_rollout`, when `start < len(emitted)` and `emitted[start] != eos`, `col_slice = emitted[start : start + stride]` is checked against `len(col_slice) == stride` (line 134). Otherwise it enters `else:` (lines 163–170):
  ```python
  if len(col_slice) == 2:
      syntax_ok = col_slice[0] == int(q_0)
      pred_o, pred_c = unpack_step_order(eos, eos, step_order)
  else:
      syntax_ok = False
      pred_o, pred_c = unpack_step_order(col_slice[1], eos, step_order)
  ```
  If `len(col_slice) == 1` (e.g. `_parse_ar_rollout([ord("2")], num_steps=1, step_order=StepOrder.SUM_FIRST, use_query_tokens=False)` or `_parse_ar_rollout([0xF8], num_steps=1, step_order=StepOrder.SUM_FIRST, use_query_tokens=True)`), `col_slice[1]` raises `IndexError: list index out of range`.
- **Impact**:
  **Cannot happen in `evaluate_ar`**, because `_rollout_ar_chunk` only stops on `tok == eos` (so a 1-token trailing slice at `start == len(emitted) - 1` always has `emitted[start] == eos` and breaks at line 130) or at `max_tokens = stride * (max(L_a, L_b) + 1) + 1 >= stride * num_steps + 1` (so every `k < num_steps` has `len(col_slice) == stride`). Still worth noting as an unguarded helper assumption if `_parse_ar_rollout` is ever called on externally truncated traces.

---

### [Note] `train.jsonl` `loss` (`nats / target`) and `tokens_seen` are not directly comparable between `ar --use-query-tokens` and `ar --no-use-query-tokens`

- **Where**: [inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/inputs.py#L18-L72), [training.py](file:///atelier/code/workshop/yessir/yessir/training.py#L154-L186), [training.py](file:///atelier/code/workshop/yessir/yessir/training.py#L252), and [query-tokens.json](file:///atelier/code/workshop/yessir/studies/addition/query-tokens.json#L5).
- **Details**:
  In `step_ar`, `[S], [C], [F]` are inserted by the step harness and masked with `IGNORE_INDEX = -100`, keeping `3` supervised targets per column step regardless of `use_query_tokens`. In pure-LM `ar`, the model must emit `[S]` and `[C]` itself during free-running generation, so `build_ar_labelled_example` rightly supervises all `4 * (K + 1) + 1` trace + `[EOS]` tokens (vs `2 * (K + 1) + 1` when `use_query_tokens=False`). Once the deterministic `[S]` and `[C]` positions converge to near-zero cross-entropy, `next_token_cross_entropy` divides total NLL by `~57` targets per problem (`L = 1..20`) instead of `~29`, making logged `train.jsonl` `loss` roughly `~0.51x` as large and `tokens_seen` `~1.97x` as large for identical value/halt predictions. Evaluation metrics in `eval.json` / `eval_splits.json` / `eval.jsonl` are unaffected.

---

### [Note] Minor config / capacity-sweep subtleties (`p_chain` at `max_length=1`, fixed `d_ff=1376` when sweeping `d_scratch`)

- **Where**: [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L95-L106), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L191-L203), [inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L449-L457), [model.py](file:///atelier/code/workshop/yessir/yessir/model.py#L20-L25), and [harness-vs-capacity.json](file:///atelier/code/workshop/yessir/studies/addition/harness-vs-capacity.json#L5).
- **Details**:
  1. `AdditionConfig` eagerly rejects unused `symbol_set` / `symbol_pool_size` when `p_prior == 1.0` with no OOD bases ([config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L191-L203)), and `AdditionStepConfig` rejects unused `d_position` and `p_terminal` ([config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L463-L474)). By contrast, passing a non-default `p_chain` when `max_length == 1` (or `eval_p_chain` when `max_length == 1` and `eval_extrap_max_length=None`) is silently accepted even though `n_shared = min(len_a, len_b) >= 2` ([inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L450)) is never satisfied.
  2. When sweeping `d_scratch` from `256` down to `0` on `ar` (`d_model: 512 -> 256`) for [harness-vs-capacity.json](file:///atelier/code/workshop/yessir/studies/addition/harness-vs-capacity.json#L5) or [minimum-model-size.json](file:///atelier/code/workshop/yessir/studies/addition/minimum-model-size.json#L5), `d_ff` stays at `1376` unless `--d-ff` is also overridden (documented in [model.py](file:///atelier/code/workshop/yessir/yessir/model.py#L23-L25)).

---

### [Note] Unreachable production fallback branch in `_rollout_ar_chunk`

- **Where**: [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L45-L108) and [test_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_ar.py#L83-L147).
- **Details**: `_rollout_ar_chunk` checks `if isinstance(model, Transformer):` at line 45 (`model` is annotated `Transformer`). Every real training/evaluation run executes lines 45–85 (`KVCache` + `collate_left_padded`); lines 87–108 only execute when unit tests pass the duck-typed `_ScriptedEvalArModel`.

---

## 3. What Was Verified and Found Sound

1. **Reference addition oracle ([reference.py](file:///atelier/code/workshop/yessir/experiments/addition/reference.py#L38-L186))**:
   - Verified `LSD -> MSD` indexing, asymmetric operand lengths (`0` substitution for `k >= L_a` or `k >= L_b`), arbitrary 240-byte symbol alphabets (`val(s_i) = i`), `B = 2` up to `B = 240`, single-digit `0 + 0` immediate halt (`K = 0`), MSD leading-zero preservation (`k < max(L_a, L_b) - 1` continues even when `o_k = c_{k+1} = s_0`), full carry ripples `(B^L - 1) + 1`, and final carry overflow (`K = N`, guaranteed to halt at `k = N` because `v_a = v_b = 0, v_c = 1 < B => c_{N+1} = s_0`).
2. **Problem sampling, carry-chain injection, and split disjointness ([inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L352-L503), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L338-L405))**:
   - Verified `sample_problem` base selection, `p_prior` canonical mixture (`SymbolSet.HEX[:B]`), `permuted_canonical` rejection sampling (`alphabet != canonical`, always `('1', '0')` in `B = 2`), and carry-chain injection (`1 <= m < min(L_a, L_b)`, `0 <= i <= min(L_a, L_b) - 1 - m`, column `i` satisfying `v_a + v_b >= B`, and columns `i + 1 .. i + m` satisfying `v_a + v_b == B - 1`).
   - Verified domain-separated evaluation split seeds (`f"{eval_seed}:eval:{split_name}"`) are invariant across training `seed` sweeps (`seed-stability.json`), strictly disjoint from integer training `seed` streams, and produce `0 / 200` overlap with 50,000 sampled default training problems while enforcing `B in {3}` (`ood_base3`), `B in {12}` (`ood_base12`), `L_a, L_b in 21..40` (`extrap_len`), and `B in {4, 8, 10}` non-ascending canonical permutations (`permuted`).
   - Verified the `~14x` column-transition ratio cited in [variants-comparison.json](file:///atelier/code/workshop/yessir/studies/addition/variants-comparison.json#L5): for `L_a, L_b ~ Uniform(1..20)`, `E[max(L_a, L_b)] = 13.825`, plus final carry overflow gives `~14.0` columns supervised per `ar` example.
3. **Full-trace sequence & pre-shifted label construction ([ar/inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/inputs.py#L18-L72), [batch.py](file:///atelier/code/workshop/yessir/yessir/batch.py#L181-L234))**:
   - Verified all four `(step_order, use_query_tokens)` combinations token-for-token against [ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/ar.md#L18-L55): `example.input_ids` has `[BOS] alphabet [SEP] a [SEP] b [SEP]` (length `P = B + L_a + L_b + 4`) followed by `trace` (length `2 * (K + 1)` or `4 * (K + 1)`), with `token_type_ids=None` and `position_ids=None`.
   - Verified pre-shifted `labels` has `(IGNORE_INDEX,) * (P - 1) + (*trace, EOS)` (length `P + len(trace) == len(example)`), supervising index `P - 1` (the third `[SEP]` -> `trace[0]`) through `P + len(trace) - 1` (`trace[-1]` -> `[EOS]`), with right-padding in `collate_labelled` filled with `IGNORE_INDEX = -100` (`input_ids` right-padded with `SpecialToken.PAD`).
4. **No answer/target leakage & exact train/eval equivalence ([embedding.py](file:///atelier/code/workshop/yessir/yessir/embedding.py#L52-L156), [blocks.py](file:///atelier/code/workshop/yessir/yessir/blocks.py#L73-L285), [heads.py](file:///atelier/code/workshop/yessir/yessir/heads.py#L21-L53), [model.py](file:///atelier/code/workshop/yessir/yessir/model.py#L124-L175))**:
   - At train time, `MultiHeadSelfAttention` runs with `is_causal=True`; right-padded `PAD` tokens sit strictly after all real tokens (`>= length`) and cannot be attended to by any supervised position `<= length - 1`.
   - At evaluation time, `collate_left_padded` aligns all prompts' final `[SEP]` at column `-1` and masks leading `PAD` slots (`key_padding_mask == False`). In `build_causal_attention_mask`, real queries only attend to causal keys with `key_padding_mask == True` (while left-`PAD` queries attend to their own diagonal so softmax never produces `NaN`). Because `ar` uses `use_token_type=False, d_position=0` (zero positional or section embeddings) and bias-free per-position `RMSNorm` / `SwiGLU` / `SubspaceLMHead`, left-padded `KVCache` prefill + 1-token decode is mathematically identical (up to floating-point associativity) to unpadded single-example `rollout_ar`.
   - Verified decode budget `max_tokens = (4 if use_query_tokens else 2) * (max(L_a, L_b) + 1) + 1` depends **only** on prompt operand lengths `(L_a, L_b)` (never on oracle `len(steps)`), and `kv_cache` capacity `prefill_len + max(max_tokens_chunk)` never overflows because decode stops as soon as `len(emitted) == max_tokens` without feeding the final token back into the cache.
5. **Metrics, `exact_match` iff `emitted == (*trace, EOS)`, and `oracle_step_accuracy` ([ar/evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L111-L331), [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L112-L288))**:
   - Proved and script-verified across 25 problems x 4 `(step_order, use_query_tokens)` modes (covering every EOS truncation, non-EOS `max_tokens` overrun, post-`K` extra-column overrun when `K = N - 1`, and single-position byte corruptions) that `classify_first_error(...) is None` holds **if and only if** `emitted == [*trace, EOS]`.
   - Verified `_compute_ar_oracle_counts` slices `argmax_ids[b_idx, n - num_targets : n]` (`n - num_targets == P - 1`, starting at the third `[SEP]`), gating each column on `syntax_ok` and `(pred_o == step.sum_digit) and (pred_c == step.carry_out)`—which is logically identical to intra-column greedy rollout from the ground-truth column-`k` prefix because any `t_0` mismatch already falsifies the conjunction.
   - Verified `exact_match + out_of_alphabet + prior_intrusion + procedural_error + syntax_malformation == 1.0`, `address_accuracy is None`, `carry_accuracy` is `None` iff zero columns are carry-active, and `{split}_base{b}` slices are emitted in `eval_splits.json` / `eval.jsonl` for multi-base splits without extra model forward passes.
6. **Config plumbing, CLI/JSON overrides, TUI switching, and training loop state ([config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L523-L533), [ar/run.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/run.py#L15-L54), [train_and_evaluate.py](file:///atelier/code/workshop/yessir/experiments/addition/train_and_evaluate.py#L49-L239), [experiment.py](file:///atelier/code/workshop/yessir/yessir/experiment.py#L179-L196), [tui.py](file:///atelier/code/workshop/yessir/yessir/tui.py#L273-L296))**:
   - Traced all 22 fields of `AdditionArConfig` end-to-end; none is shadowed or ignored. Passing inapplicable step-harness keys (e.g. `embedding_mode`, `p_terminal`, `all_steps`, `include_alphabet`) via `--config` JSON raises `TypeError` immediately via `AdditionArConfig.from_dict`. Switching variants in the TUI (`cls -> ar`) updates untouched `d_scratch` from `208` to `256` (`d_model = 512`) while preserving user-edited fields.
   - Verified `evaluate_ar` runs under `@torch.no_grad()` and sets `model.eval()`, `_train_step_with_tokens` restores `model.train()`, and periodic `eval_every` checkpoints preserve a single continuous `iter(train_dataset)`, `AdamW` state, and PyTorch RNG trajectory.

---

## 4. Test Coverage & Gaps

Targeted test suites executed via `.venv-linux/bin/pytest`:
- `experiments/addition/tests/` (`test_ar.py`, `test_config.py`, `test_evaluation.py`, `test_inputs.py`, `test_reference.py`, `test_run.py`): **83 passed**.
- `yessir/tests/`: **182 passed**.

**What is well-covered for `ar`**:
- All four `(step_order, use_query_tokens)` sequence and pre-shifted label layouts on spec Examples 1 and 2, plus incremental prefix slices matching full teacher-forced examples ([test_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_ar.py#L150-L273)).
- `AdditionArDataset` determinism, `iter_problems` equivalence, and inspect CLI ([test_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_ar.py#L275-L319)).
- `rollout_ar` `[EOS]` stopping, immediate `[EOS]` at index `0`, and `max_tokens` length caps with and without query tokens ([test_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_ar.py#L321-L345)).
- Batch-size invariance (`batch_size=1` vs `batch_size=3`/`7`), exact agreement between batched KV-cached `_rollout_ar_chunk` and single-example uncached `rollout_ar`, and `exact_match` agreement on a partially trained `Transformer` ([test_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_ar.py#L544-L713)), plus full CPU convergence (`>= 98%` EM on reversed binary alphabets `b"01"` and `b"10"`, [test_convergence.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_convergence.py#L245-L300)) and CLI artefact smoke run ([test_run.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_run.py#L308-L364)).

**Coverage gaps identified**:
1. **Mid-column premature `[EOS]` under `use_query_tokens=True` (`_parse_ar_rollout` lines 163–179) has zero test coverage**: [test_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_ar.py#L473-L486) only tests `[EOS]` at index `4` (a column boundary `start = 4`, which breaks at line 130) and never exercises `len(col_slice) == 2` (`[[S], EOS]`), `len(col_slice) == 3` (`[[S], o_0, EOS]`), or `pred_t1 == EOS` (`[[S], o_0, [C], EOS]`).
2. **Overrun rollouts (missing `[EOS]` at `K` when `K = max(L_a, L_b) - 1`, or hitting `max_tokens` without `[EOS]`) are not directly tested in `test_evaluate_ar_*`**: [test_evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_evaluation.py#L332-L347) tests pre-built `StepPrediction` overruns in `score_predictions`, but [test_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_ar.py#L415-L542) does not test `_parse_ar_rollout` when a rollout emits an extra `N + 1`-th column on a non-overflowing problem or hits `max_tokens` without `[EOS]`.
3. **`_ScriptedEvalArModel` in [test_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_ar.py#L106-L108) hardcodes `use_query_tokens=False` (and `SUM_FIRST`) when building `oracle_full_noq`**, and exercises the non-`Transformer` fallback loop in `_rollout_ar_chunk` rather than the `KVCache` path.
4. **End-to-end canonical-fallback behaviour on `permuted` (`reference_add(SymbolSet.HEX[:B], a, b)`) is untested** in [test_evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_evaluation.py#L108-L155), which only tests synthetic single-step output substitutions (`pred_token == SymbolSet.HEX[digit_value]`).

---

## 5. What Was Not Inspected

- Variant-specific implementation files (`inputs.py`, `dataset.py`, `generation.py`, `evaluation.py`, `run.py`) and unit tests for `cls`, `step_fork`, `addressed_step_ar` (and `step_ar` / `addressed_ar` beyond cross-checking their specs and `evaluation.py` against `ar`).
- Frontend TypeScript/Astro rendering code under `web/` and cloud job submission code in [vertex.py](file:///atelier/code/workshop/yessir/yessir/vertex.py).
- Prototype code and runs under `.prototypes/transformadder/`.
