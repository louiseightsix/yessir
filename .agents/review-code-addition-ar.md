# Forensic Code Correctness Review: `addition` / `ar`

- **Experiment / Variant**: `addition` / `ar` (full-trace autoregressive next-token prediction via `SubspaceLMHead`)
- **Audited Commit (`HEAD`)**: `bc2244b9fc121bbbfaf96302e51e7e0082083cac`
- **Historical Run Commits Checked**: None (`results/addition/` does not exist yet at `HEAD`; all 13 study files under `studies/addition/*.json` are in `"status": "draft"` with `"runs": []`)
- **Verdict**: **Sound for headline `exact_match`; one medium spec-vs-code defect in `evaluate_ar`'s premature-`[EOS]` error taxonomy (with a coupled `step_accuracy` asymmetry across `use_query_tokens` and sibling variants), one medium teacher-forcing interpretation caveat on secondary/diagnostic metrics (`step_accuracy`, `digit_accuracy`, `carry_accuracy`, `halt_accuracy`), and two minor spec/TUI defects.**

## Executive Summary

At `bc2244b`, the `addition` / `ar` implementation faithfully executes the task and full-trace sequence specification across `reference.py`, shared `inputs.py`, `variants/ar/{inputs,dataset,generation,evaluation,run}.py`, `config.py`, `train_and_evaluate.py`, and shared `yessir/` modules:

1. **No bug invalidates `exact_match`.** Ground-truth base-`B` `LSD -> MSD` addition (`reference_add`), alphabet/operand/carry-chain sampling (`sample_problem`), train/eval split disjointness (`id`, `ood_base3`, `ood_base12`, `extrap_len`, `permuted`), RNG stream isolation, prompt/trace layout, pre-shifted target labels (`(IGNORE_INDEX,) * (P - 1) + (*trace, EOS)`), causal masking (`is_causal=True`), right-padding invariance, orthogonal subspace embeddings (`SubspaceEmbedding`), and `SubspaceLMHead` all match `spec/task.md` and `spec/variants/ar.md`. Under causal masking and deterministic `argmax`, single-pass teacher-forced `exact_match` (`evaluate_ar`) is mathematically equivalent to free-running rollout `exact_match` (`rollout_ar`) up to `fp32` kernel rounding (`~3e-7` max logit diff across padded batch shapes).
2. **All 29 fields of `AdditionArConfig` are wired end-to-end with zero silent no-ops**, and step-harness / addressed-only parameters (`p_terminal`, `all_steps`, `include_alphabet`) are strictly rejected (`TypeError` / `SystemExit`) rather than silently ignored on `ar`.
3. **Key issues found**:
   - **Finding 1 (Medium — Spec Mismatch & Cross-Mode/Variant Asymmetry)**: In `variants/ar/evaluation.py`, when `use_query_tokens=False` (the default), emitting a premature `[EOS]` at a column boundary `preds[2 * k]` (`0 <= k <= K`) is unpacked into `pred_o` or `pred_c` with `syntax_ok=True`. Because `classify_first_error` checks value mismatches before `halt_ok` and `EOS` (`0xF3`) is outside `alphabet` (`0x00..0xEF`), every premature `[EOS]` is classified as `out_of_alphabet` instead of `syntax_malformation` (contradicting `spec/task.md` §3) and penalises column `k` in `step_accuracy`, whereas `use_query_tokens=True` and all five sibling variants (`cls`, `step_fork`, `step_ar`, `addressed_step_ar`, `addressed_ar`) classify the same premature halt after column `k - 1` as `syntax_malformation` without penalising column `k`'s teacher-forced `step_accuracy`.
   - **Finding 2 (Medium — Teacher-Forced Evaluation Caveat)**: `run_addition_ar` evaluates via single-pass teacher forcing (`evaluate_ar`, as specified in `spec/task.md` §3) rather than free-running `rollout_ar`. While `exact_match` is unaffected, `step_accuracy` (recorded alongside `exact_match` in `eval.json`), `digit_accuracy`, `carry_accuracy`, and `halt_accuracy` feed oracle trace tokens after—and under `carry_first` or `use_query_tokens=True`, even *within*—failing columns, inflating those metrics relative to free-running rollout on failed problems.
   - **Finding 3 (Low — Spec / Study Text Drift)**: `spec/task.md` §3 attributes batch-shape `argmax` flips to `bfloat16` autocast even though the entire codebase runs in `fp32`; `studies/addition/query-tokens.json` mentions inserting `[F]` in `ar` and `addressed_ar`, whereas `ar` and `addressed_ar` only emit and supervise `[S]` and `[C]` (halting via `[EOS]`).
   - **Finding 4 (Low — Interactive TUI Stepping Quirk)**: `_step_value` in `yessir/tui.py` clamps float fields at `max(step_f, next_f)` (`0.05`), preventing left-arrow stepping back to `0.0` on `p_prior`, `p_chain`, and `eval_p_chain`.

---

## Findings (Suspected Bugs, Spec Mismatches, and Correctness Risks)

### Finding 1 (Medium): Premature `[EOS]` under `use_query_tokens=False` is misclassified as `out_of_alphabet` instead of `syntax_malformation` and creates a `step_accuracy` asymmetry across `use_query_tokens` and variants

- **Location**:
  - `variants/ar/` evaluator: [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L115-L137)
  - Shared error classifier: [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L78-L103), [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L132-L154)
  - Specification: [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L143-L159) (`§3`)
- **Spec expectation vs actual behaviour**:
  - [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L158) defines `syntax_malformation` as: *"a non-value prediction at column `k` failed—either a supervised trace header/query/address token (`[IDX]`, `addr(k)`, `[S]`, `[C]`) was ill-formed, or `(o_k, c_{k+1})` were valid and the model emitted a premature or missing halting/step control token (`FLAG_FINISHED` / `FLAG_CONTINUE` / `[EOS]` / `[IDX]`)"*.
  - In `ar` with `use_query_tokens=False` (default), there is no delimiter or query token between column `k - 1` and column `k`: halting prematurely right after completing column `k - 1` (`1 <= k <= K`, or immediately at `k = 0`) means predicting `SpecialToken.EOS` (`0xF3`) at `preds[2 * k]`, which `evaluate_ar` unpacks as column `k`'s first value token (`pred_o` under `sum_first`, `pred_c` under `carry_first`) while hardcoding `syntax_ok = True` ([evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L120-L123)).
  - Although `evaluate_ar` records `no_premature_eos = all(tok != eos for tok in col_slice)` into `halt_ok = False` ([evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L124-L129)), `classify_first_error` checks `value_pairs` ([evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L141-L149)) *before* `if not pred.halt_ok:` ([evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L151)). Because `EOS = 0xF3` is never in `alphabet` (`0x00..0xEF`), `_classify_value_mismatch` ([evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L87-L88)) unconditionally returns `FirstErrorType.OUT_OF_ALPHABET`. Consequently, under `use_query_tokens=False`, premature `[EOS]` can **never** reach `FirstErrorType.SYNTAX_MALFORMATION` (only a missing final `[EOS]` at `preds[-1]` can).
  - By contrast, when the exact same premature `[EOS]` occurs right after valid column `k - 1`:
    - In `ar` with `use_query_tokens=True`, `[EOS]` lands at `preds[4 * k]` (`pred_q0`), setting `syntax_ok = False` ([evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L118-L119)) -> classified as `syntax_malformation`, AND because slot `4 * k` is then teacher-forced with `[S]`/`[C]`, column `k`'s value slots `4 * k + 1` and `4 * k + 3` still predict valid `(o_k, c_{k+1})` and score `1.0` in `step_accuracy`.
    - In `addressed_ar` ([evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/addressed_ar/evaluation.py#L132-L152)), `cls`, `step_fork`, `step_ar`, and `addressed_step_ar`, premature halting after column `k - 1` fails `halt_ok` on column `k - 1` (`after_col_pred == EOS != IDX` or `pred_flag == FLAG_FINISHED != FLAG_CONTINUE`) -> classified as `syntax_malformation`, while column `k`'s teacher-forced `(o_k, c_{k+1})` still score `1.0` in `step_accuracy`.
- **Concrete input / config trigger (verified via Python probe)**:
  On Example 1 (`alphabet = b"0123456789"`, `a = b"789"`, `b = b"52"`, `987 + 25 = 1012`, `K + 1 = 4` columns), suppose a model computes columns `0..2` (`2 1`, `1 1`, `0 1`) accurately, prematurely emits `[EOS]` right after column `2` (forgetting the carry-overflow column `k = 3`), and predicts valid `'1', '0'` when teacher-forced inside column `3`:
  - `evaluate_ar(..., use_query_tokens=False)` on `preds = [*b"211101", EOS, ord("0"), EOS]` returns:
    `exact_match = 0.0`, **`step_accuracy = 0.75`**, `digit_accuracy = 0.75`, `carry_accuracy = 0.75`, `halt_accuracy = 0.0`, **`out_of_alphabet = 1.0`**, **`syntax_malformation = 0.0`**.
  - `evaluate_ar(..., use_query_tokens=True)` on `preds = [*q_trace[:12], EOS, *q_trace[13:], EOS]` returns:
    `exact_match = 0.0`, **`step_accuracy = 1.0`**, `digit_accuracy = 1.0`, `carry_accuracy = 1.0`, `halt_accuracy = 0.0`, **`out_of_alphabet = 0.0`**, **`syntax_malformation = 1.0`**.
- **Plausible impact on headline or diagnostic numbers**:
  - **Headline `exact_match`**: None (`0.0` in all cases).
  - **Secondary headline `step_accuracy` (`eval.json`)**: Moves `step_accuracy` by `-1 / (K + 1)` per premature-`[EOS]` column boundary in `ar (use_query_tokens=False)` relative to `ar (use_query_tokens=True)` and all five sibling variants (relevant when comparing `step_accuracy` in [query-tokens.json](file:///atelier/code/workshop/yessir/studies/addition/query-tokens.json), [variants-comparison.json](file:///atelier/code/workshop/yessir/studies/addition/variants-comparison.json), and `extrap_len` `L = 21..40` in [embedding-modes-length.json](file:///atelier/code/workshop/yessir/studies/addition/embedding-modes-length.json)).
  - **Diagnostic error taxonomy (`eval_splits.json`, `eval.jsonl`)**: Directly shifts premature-halting failures (most notably halting around `k = 20` on `extrap_len` `L = 21..40`, or dropping carry-overflow column `K = N`) from `syntax_malformation` into `out_of_alphabet` whenever `use_query_tokens=False`.

---

### Finding 2 (Medium): `evaluate_ar` scores single-pass teacher-forced predictions rather than free-running `rollout_ar`—exact for `exact_match`, but optimistic for `step_accuracy`, `digit_accuracy`, `carry_accuracy`, and `halt_accuracy`

- **Location**:
  - Teacher-forced evaluator: [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L65-L139)
  - Unused rollout helper: [generation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/generation.py#L14-L67)
  - Variant runner: [run.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/run.py#L34-L48)
  - Spec justification: [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L143-L150), [ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/ar.md#L77-L83)
- **Spec / checklist expectation vs actual behaviour**:
  - `evaluate_ar` implements the exact teacher-forced protocol described in [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L143) and [ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/ar.md#L77) (matching all sibling `addition` variants); `rollout_ar` is never called during `train_and_evaluate`.
  - **Why `exact_match` is unaffected**: Under causal self-attention (`is_causal=True` in [blocks.py](file:///atelier/code/workshop/yessir/yessir/blocks.py#L111)) and per-position norms/FFNs/heads, position `P - 1 + j` in `evaluate_ar` sees `prefix + trace[:j]`, which is token-, type-, and position-identical to step `j` of `rollout_ar` up to the earliest mismatch `j_0`. Because `trace` contains only alphabet bytes (`0x00..0xEF`) and optional `[S]`/`[C]` (`0xF8`/`0xF9`)—never `EOS` (`0xF3`)—and `rollout_ar`'s cap `(4 if use_query_tokens else 2) * (max(L_a, L_b) + 1) + 1` is always `>= len(trace) + 1`, `evaluate_ar` and `rollout_ar` agree on `exact_match` in exact arithmetic.
  - **Where non-`exact_match` metrics diverge from autoregressive rollout** (relevant to the `.agents/reviewer-code.md` outer-loop rollout check):
    1. *Inter-column carry and halt reset*: If column `0` emits a wrong carry `pred_c[0]` (or premature `[EOS]`), `rollout_ar` feeds `pred_c[0]` into column `1` (or halts immediately), whereas `evaluate_ar` feeds the oracle `trace` so columns `1 .. K` still receive ground-truth carries `c_1 .. c_K` and can score `1.0` in `step_accuracy` (`eval.json`), `digit_accuracy`, and `carry_accuracy`.
    2. *Intra-column conditioning under `carry_first` and `use_query_tokens=True`*: Under `StepOrder.CARRY_FIRST`, `o_k` at slot `2 * k + 1` is predicted given the **oracle** `c_{k+1}` at slot `2 * k`, not `pred_c[k]`. Under `use_query_tokens=True`, `t_0` and `t_1` are predicted given oracle `[S]` and `[C]` even if `pred_q0` / `pred_q1` were wrong (explicitly asserted in [test_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_ar.py#L468-L487)).
    3. *Halting decoupled from predicted `c_{K+1}`*: At final column `K` (oracle `c_{K+1} == s_0`), if the model wrongly predicts carry overflow `pred_c[K] = s_1`, free-running `rollout_ar` appends `s_1` and continues into an extra column `K + 1` instead of halting at `K`; teacher-forced `evaluate_ar` feeds oracle `c_{K+1} = s_0` at `trace[-1]` (or `trace[-2]` under `carry_first`), so `preds[-1]` sees `c_{K+1} == s_0` and emits `[EOS]`, scoring `halt_accuracy = 1.0`.
- **Concrete input / config triggers (verified via Python probe)**:
  - *`carry_first` intra-column inflation*: On `987 + 25 = 1012` (`step_order=StepOrder.CARRY_FIRST`), corrupting `pred_c[0]` from `'1'` to `'0'` while keeping conditional-subtraction outputs aligned to the oracle trace gives `exact_match = 0.0`, `step_accuracy = 0.75`, **`digit_accuracy = 1.0`**, `halt_accuracy = 1.0`.
  - *`halt_accuracy` inflation on final carry error*: On `12 + 34 = 46` (`a = b"21"`, `b = b"43"`, `K = 1`), predicting `preds = [ord("6"), ord("0"), ord("4"), ord("1"), EOS]` (wrong carry `'1'` on final column `k = 1`) scores **`halt_accuracy = 1.0`** (`step_accuracy = 0.5`, `exact_match = 0.0`).
- **Plausible impact on headline or diagnostic numbers**:
  - Does **not** move `exact_match`.
  - Inflates `step_accuracy` (in `eval.json`), `digit_accuracy`, `carry_accuracy`, and `halt_accuracy` relative to free-running generation whenever `exact_match < 1.0`—a critical distinction when interpreting partial progress on `extrap_len` (`L = 21..40`), `carry-chain-injection`, or `step-order` ablations.

---

### Finding 3 (Low): Minor documentation / study drift (`bfloat16` autocast note in `spec/task.md`; `[F]` query token in `studies/addition/query-tokens.json`)

- **Location**:
  - [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L145) vs [training.py](file:///atelier/code/workshop/yessir/yessir/training.py#L205-L267) and [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L87-L103)
  - [query-tokens.json](file:///atelier/code/workshop/yessir/studies/addition/query-tokens.json#L5) vs [inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/inputs.py#L54-L62) and [ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/ar.md#L14)
- **Details**:
  1. [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L145) notes `(under bfloat16 autocast, differences in padding and batch shape can flip near-tied argmax decisions)`, whereas no `torch.autocast` or `bfloat16` cast exists anywhere in `yessir/` or `experiments/addition/`; both training and evaluation execute in `fp32` (where padded vs unpadded SDPA/GEMM reduction order still produces `~3e-7` `fp32` ULP differences).
  2. [query-tokens.json](file:///atelier/code/workshop/yessir/studies/addition/query-tokens.json#L5) groups `step_ar`, `ar`, `addressed_step_ar`, and `addressed_ar` as inserting `([S], [C], [F] via use_query_tokens=True)`. As correctly specified in [ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/ar.md#L14) and implemented in [inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/inputs.py#L54) (`q_0, q_1, _ = step_query_tokens(order)`), `ar` omits `[F]` (`AdditionToken.QUERY_FLAG`) because halting uses `[EOS]` rather than `FLAG_CONTINUE`/`FLAG_FINISHED`, and unlike `step_ar` (where `[S]`, `[C]`, `[F]` are masked with `IGNORE_INDEX`), `ar` supervises `[S]` and `[C]` in `next_token_cross_entropy` ([inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/inputs.py#L120)).
- **Impact**: Zero impact on evaluation metrics; worth keeping in mind when comparing `train.jsonl` scalar losses between `use_query_tokens=False` (`2 * (K + 1) + 1` targets) and `use_query_tokens=True` (`4 * (K + 1) + 1` targets, nearly half of which are deterministic `[S]`/`[C]` markers).

---

### Finding 4 (Low): Interactive TUI left-arrow stepping cannot step float fields (`p_prior`, `p_chain`, `eval_p_chain`) down to `0.0`

- **Location**: [tui.py](file:///atelier/code/workshop/yessir/yessir/tui.py#L136-L143), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L105-L110), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L149-L151)
- **Details**: `_step_value` ends float stepping with `return max(step_f, next_f)` (`step_f = 0.05` for `p_prior`, `p_chain`, and `eval_p_chain`). Stepping `p_prior` right (`0.0 -> 0.05`) and then left (`0.05 -> 0.05`) clamps at `0.05` rather than `0.0` (verified via `ConfigEditorState(AdditionArConfig)` probe; inline typing `0` + `Enter` still sets `0.0` cleanly).
- **Impact**: Interactive TUI convenience only; cannot corrupt CLI/JSON runs or recorded configs.

---

## What Was Read and Verified Correct

### 1. Data Generation (`reference.py`, `inputs.py`, `yessir/alphabet.py`)

- **Reference addition oracle** ([reference.py](file:///atelier/code/workshop/yessir/experiments/addition/reference.py#L69-L186)):
  - Hand-checked `LSD -> MSD` arithmetic, carry propagation, unequal operand lengths (`v_a = 0` for `k >= L_a`, `v_b = 0` for `k >= L_b`), and halting (`FLAG_FINISHED` iff `k >= max(L_a, L_b) - 1` and `c_{k+1} == s_0`):
    - *Single-column zero*: `0 + 0` in `B = 2` (`alphabet = b"ab"`, `a = b"a"`, `b = b"a"`) -> `k = 0: total = 0, o_0 = 'a', c_1 = 'a', FLAG_FINISHED` (`1` step).
    - *Single-column overflow*: `1 + 1` in `B = 2` (`b"01"`, `b"1"`, `b"1"`) -> `k = 0: o_0 = '0', c_1 = '1', FLAG_CONTINUE`; `k = 1: total = 1 < B, o_1 = '1', c_2 = '0', FLAG_FINISHED` (`2` steps).
    - *Spec Example 1 (`987 + 25 = 1012`, `B = 10`)* and *Example 2 (`5 + 1 = 6`, `B = 3`, `alphabet = b"q#7"`)*: all `sum_digit`, `carry_in`, `carry_out`, and `flag` values match `spec/task.md` §1.
    - *Smallest and largest bases (`B = 2` and `B = 240`)*: `total <= 2 * B - 1 < 2 * B` ensures `total // B in {0, 1}` for every `B >= 2`, and at overflow column `k = N = max(L_a, L_b)`, `total = 1 < B` guarantees `carry_out == s_0` and termination in at most `N + 1` columns.
    - Invalid unreachable states (`k = 0` with `carry != s_0`, `k = N` with `carry != s_1`, `carry not in (s_0, s_1)`, empty operands, duplicate/special symbols in `alphabet`) raise `ValueError`/`IndexError` eagerly.
- **Problem, alphabet, and carry-chain sampling** ([inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L403-L496), [alphabet.py](file:///atelier/code/workshop/yessir/yessir/alphabet.py#L22-L70)):
  - Base `B` is drawn uniformly from `bases`; `L_a` and `L_b` are drawn independently and uniformly from `min_length .. max_length`.
  - When `permuted_canonical=True`, `sample_problem` redraws shuffled `SymbolSet.HEX[:B]` until `alphabet != canonical` ([inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L418-L425)), ensuring `B = 2` always produces `('1', '0')` rather than returning the unpermuted canonical order `50%` of the time.
  - When `p_chain > 0` and `n_shared = min(L_a, L_b) >= 2`, `m in 1 .. n_shared - 1` and `i in 0 .. n_shared - 1 - m` stay strictly in bounds (`i + m <= n_shared - 1`), setting `a_vals[i] + b_vals[i] >= B` (carry generation) and `a_vals[j] + b_vals[j] == B - 1` across `j = i + 1 .. i + m` (carry propagation).
- **Train/eval disjointness and RNG separation** ([config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L289-L292), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L353-L416), [train_and_evaluate.py](file:///atelier/code/workshop/yessir/experiments/addition/train_and_evaluate.py#L162-L179)):
  - `AdditionConfig.__post_init__` rejects `eval_seed == seed` and strips any `eval_ood_bases` entries already present in `bases`.
  - `_split_seed(offset)` assigns collision-free seeds across `id` (`offset = 0`), `ood_base{b}` (`1000 + b`, with `2 <= b <= 240`), `extrap_len` (`2000`), and `permuted` (`3000`), incrementing by `+1` if `eval_seed + offset == seed`.
  - Structural disjointness holds strictly: `ood_base*` uses bases absent from `bases`; `extrap_len` draws `L_a, L_b` from `(max_length + 1) .. eval_extrap_max_length` (`21..40` vs `1..20`); `permuted` uses non-ascending permutations of `HEX[:B]` (disjoint from `p_prior`'s ascending `HEX[:B]`, and colliding with uniform random 240-byte alphabets only at rate `1 / (240 * 239) = 1.7e-5` in `B = 2` and `<= 7.1e-9` for `B >= 4`).
  - Periodic mid-training evaluation creates fresh local `random.Random(split.seed)` instances in `iter_problems` without touching `train_stream`'s `random.Random(config.seed)` or PyTorch's RNG.

### 2. Tokenisation, Sequence Layout, Target Shifting, and Masking (`variants/ar/inputs.py`, `yessir/batch.py`, `yessir/blocks.py`, `yessir/embedding.py`)

- **Exact layout vs `spec/variants/ar.md` §1–§3** ([inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/inputs.py#L21-L121)):
  - Shared prefix `[BOS] s_0..s_{B-1} [SEP] a_0..a_{L_a-1} [SEP] b_0..b_{L_b-1} [SEP]` has length `P = B + L_a + L_b + 4`, with `TokenType`s `(0, 1*B, 0, 2*L_a, 0, 3*L_b, 0)` and `position_ids` `(0, 0..B-1, 0, 0..L_a-1, 0, 0..L_b-1, 0)`.
  - Trace tokens per column `k` (`0 <= k <= K`):
    - `sum_first, use_query_tokens=False`: `(o_k, c_{k+1})` at `((OUTPUT, k), (CARRY, k + 1))`.
    - `carry_first, use_query_tokens=False`: `(c_{k+1}, o_k)` at `((CARRY, k + 1), (OUTPUT, k))`.
    - `sum_first, use_query_tokens=True`: `([S], o_k, [C], c_{k+1})` at `((SPECIAL, k), (OUTPUT, k), (SPECIAL, k), (CARRY, k + 1))`.
    - `carry_first, use_query_tokens=True`: `([C], c_{k+1}, [S], o_k)` at `((SPECIAL, k), (CARRY, k + 1), (SPECIAL, k), (OUTPUT, k))`.
  - Outgoing carry `c_{k+1}` sits at `(TokenType.CARRY, k + 1)` under all four combinations, aligning with `a_{k+1}` (`OPERAND_A, k + 1`) and `b_{k+1}` (`OPERAND_B, k + 1`) at zero positional offset when attended to by column `k + 1`.
  - Halting uses **only** trailing `SpecialToken.EOS` (`0xF3`) in `labels`; `FLAG_CONTINUE` and `FLAG_FINISHED` never appear in `ar`.
- **Target shifting, causal masking, and zero answer leakage**:
  - `build_ar_labelled_example` ([inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/inputs.py#L119-L121)) constructs `example` of length `P + len(trace)` (excluding `EOS`) and `labels = (IGNORE_INDEX,) * (P - 1) + (*trace, EOS)` of equal length `P + len(trace)`.
  - Positions `0 .. P - 2` are masked with `IGNORE_INDEX = -100`; position `P - 1` (the third `[SEP]`) predicts `trace[0]`; position `P - 1 + j` (`1 <= j < len(trace)`) sees `prefix + trace[:j]` and predicts `trace[j]`; position `P + len(trace) - 1` (`trace[-1]`) predicts `EOS`.
  - `collate_labelled` ([batch.py](file:///atelier/code/workshop/yessir/yessir/batch.py#L101-L154)) right-pads `input_ids` with `SpecialToken.PAD`, `token_type_ids`/`position_ids` with `0`, and `labels` with `IGNORE_INDEX = -100`.
  - Verified empirically via Python probe: corrupting `ex.input_ids[-1]` produces **exact `0.0`** max absolute logit change across all earlier positions `0 .. len(ex) - 2`, and collating a short problem alongside a much longer problem produces `2.98e-7` max `fp32` logit difference across unpadded positions.
- **Train-vs-eval parity across `EmbeddingMode`**:
  - `AdditionArDataset`, `evaluate_ar`, and `rollout_ar` all call `build_ar_example` -> `make_example`, attaching identical `(input_ids, token_type_ids, position_ids)` for `TOKEN_TYPE_PE`, `(input_ids, token_type_ids, None)` for `TOKEN_TYPE_NOPE`, and `(input_ids, None, None)` for `NOPE`.
- **Length extrapolation (`L = 21..40`) positional bounds**:
  - `SubspaceEmbedding` ([embedding.py](file:///atelier/code/workshop/yessir/yessir/embedding.py#L85-L106)) has **no** fixed `max_seq_len` buffer; unit-norm sinusoidal encodings (`_SINUSOIDAL_BASE = 256.0`) are computed dynamically in `float32`. Across the 200 default `extrap_len` problems (`L = 21..40`, `bases = (2, 4, 8, 10)`), sequence lengths reach `169` (`use_query_tokens=False`) / `249` (`use_query_tokens=True`) and `position_ids` reach `40` (theoretical max `41` on `L = 40` carry overflow, or `B - 1` in the alphabet prefix), well within the slowest sinusoidal full wavelength of `~1,137` positions at `d_position = 32`.

### 3. Model, Training, and End-to-End Config Trace (`config.py`, `variants/ar/run.py`, `train_and_evaluate.py`, `yessir/training.py`)

- **Complete 29-field config trace on `AdditionArConfig`**:
  - *Task & sampling*: `bases` (`parsed_bases`), `min_length`, `max_length`, `symbol_set` + `symbol_pool_size` (`symbol_pool`), `p_prior`, `p_chain`, `seed` -> all passed to `AdditionArDataset` in [run.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/run.py#L22-L33) and `config.eval_splits()` / `iter_problems` in [train_and_evaluate.py](file:///atelier/code/workshop/yessir/experiments/addition/train_and_evaluate.py#L120-L179).
  - *Variant flags*: `step_order` and `use_query_tokens` -> passed to both `AdditionArDataset` and `partial(evaluate_ar, ...)` in [run.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/run.py#L29-L39).
  - *Embedding & backbone*: `embedding_mode`, `n_layers`, `n_heads`, `d_ff`, `d_position`, `d_scratch`, `fixed_d_model` -> wired through `use_token_type`, `effective_d_position`, and `effective_d_scratch` (`d_model` held constant at `512` by default when `fixed_d_model=True`) into `Transformer(head_type=HeadType.SUBSPACE_LM, ...)` ([train_and_evaluate.py](file:///atelier/code/workshop/yessir/experiments/addition/train_and_evaluate.py#L107-L116)). *(Researcher note: when sweeping `d_scratch` under `nope` or `token_type_nope` for capacity studies, `fixed_d_model=True` still adds the dropped `16` and/or `d_position` dims into `effective_d_scratch` unless `fixed_d_model=False` is passed, as documented in `spec/task.md` §2.)*
  - *Optimisation & evaluation*: `max_steps`, `batch_size`, `lr`, `log_every`, `eval_every` (`effective_eval_every`), `num_eval_problems`, `eval_batch_size`, `eval_seed`, `eval_p_chain`, `eval_ood_bases`, `eval_extrap_max_length`, `eval_permuted_canonical` -> all active in `train_and_evaluate`. *(Researcher note: in `ar`, each item in `batch_size = 64` is a full `(K + 1)`-column problem rather than a single column step as in `cls`/`step_ar`, so one `ar` step trains on `~14x` more column transitions at `L in 1..20`, and `next_token_cross_entropy` averages across all non-ignored target tokens in the batch.)*
  - *Rejection of inapplicable variant options*: `AdditionArConfig` inherits directly from `AdditionConfig` (not `AdditionStepConfig` or `AdditionAddressedArConfig`). Passing `p_terminal`, `all_steps`, or `include_alphabet` via `AdditionArConfig.from_dict`, CLI flags, or `--config <json>` raises `TypeError` or `SystemExit(2)` immediately (verified via Python probe).

### 4. Evaluation Rollout (`generation.py`), `model.eval()`, and Batch Isolation

- **`rollout_ar`** ([generation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/generation.py#L14-L67)):
  - Decorated with `@torch.no_grad()`, calls `model.eval()`, starts from an empty trace after the third `[SEP]`, reads `logits[0, -1, :].argmax(dim=-1)` one token at a time, and stops immediately upon emitting `SpecialToken.EOS` (including `EOS` in the returned list).
  - Its token cap `tokens_per_col * (max(len(a_tokens), len(b_tokens)) + 1) + 1` uses only the observable prompt operand lengths `len(a)` and `len(b)`—never calling `reference_add` and never leaking whether the true sum overflows (`K = N - 1` vs `K = N`). On non-overflowing problems (`K = N - 1`), a model that fails to emit `EOS` continues generating `tokens_per_col + 1` extra tokens rather than being artificially stopped at the ground-truth length.
  - Note: `generation.py` returns raw `list[Token]` and has no standalone `parse_ar_trace` helper (unlike `addressed_ar/generation.py`); trace unpacking into columns happens only inside teacher-forced `evaluate_ar`.
- **Eval mode & statelessness**: Both `evaluate_ar` and `rollout_ar` call `model.eval()` under `@torch.no_grad()`, `Transformer` contains zero stateful normalisation/dropout/KV-cache buffers across batches, and `train_step` restores `model.train()` on the next training iteration.

### 5. Test Coverage Audit (`experiments/addition/tests/`, `yessir/tests/`)

- **Executed**:
  - `./yessir.sh uv run --no-sync pytest -p no:cacheprovider -m "not convergence" experiments/addition/tests/ yessir/tests/` -> **298 passed** (`19.00s`).
  - `./yessir.sh uv run --no-sync pytest -p no:cacheprovider experiments/addition/tests/test_convergence.py -k test_addition_ar_cpu_convergence` -> **1 passed** (`52.08s`, verifying end-to-end CPU training to `>= 98%` `exact_match` on `evaluate_ar` plus exact `rollout_ar` match on carry-rippling problems across reversed binary alphabets `b"01"` and `b"10"`).
  - `./yessir.sh uv run --no-sync ruff check` and `./yessir.sh uv run --no-sync ty check experiments yessir` -> **All checks passed**.
- **Covered well by tests**:
  - Spec Examples 1 & 2 across all 4 `(step_order, use_query_tokens)` combinations and all 3 `EmbeddingMode`s, incremental prefix equality between `rollout_ar` and `build_ar_labelled_example`, dataset CLI dumping, `rollout_ar` `EOS` stopping and max-length capping, `evaluate_ar` batch-size invariance, within-column error-precedence (`syntax_ok` -> `step_order` values -> `halt_ok`), `prior_intrusion` vs `procedural_error` on permuted canonical alphabets, periodic `eval_every` non-interference with `train.jsonl`, and CLI smoke runs.
- **Specific test gaps**:
  1. **No test checks premature `[EOS]` at a column boundary (`k >= 1`) after valid earlier columns in `ar`**, nor compares `out_of_alphabet` vs `syntax_malformation` between `use_query_tokens=False` and `use_query_tokens=True` ([test_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_ar.py#L489-L504) actually hardcodes `premature[0] = eos` -> `assert m_prem.out_of_alphabet == 1.0`, cementing Finding 1 rather than catching the spec mismatch).
  2. **No test directly asserts equivalence of `exact_match` between `evaluate_ar` and free-running `rollout_ar` across a multi-problem random evaluation split** (including problems where the model fails to emit `[EOS]` on non-overflowing `K = N - 1` inputs).
  3. **No convergence test exercises `ar` under `StepOrder.SUM_FIRST` (the default `step_order`), `use_query_tokens=True`, `TOKEN_TYPE_NOPE`, or `NOPE`** (`test_addition_ar_cpu_convergence` only tests `CARRY_FIRST` with `use_query_tokens=False` and `TOKEN_TYPE_PE`).

---

## What Was Not Checked

- **Live GPU / `cuda` and `mps` execution**: This VM environment was audited on CPU; CUDA/MPS kernel determinism under `torch.use_deterministic_algorithms(True, warn_only=True)` and large-batch `L = 21..40` GPU memory footprint were not benchmarked on hardware accelerators.
- **Full 100k–500k-step multi-base training dynamics**: No recorded runs exist yet under `results/addition/`, and full-scale training sweeps (`bases = "2,4,8,10"`, `L = 1..20`, `100k`–`500k` steps) were not run during this static + unit/probe audit.
- **Astro `web/` frontend rendering and Cloud Build / Vertex AI live job execution**: `web/` TypeScript code and live GCP `gcloud` commands in `yessir/vertex.py` were outside the scope of this `addition/ar` Python variant audit.
