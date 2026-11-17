# Code Review Issues Backlog (`yessir`, `addition`, `sir_yes_sir`)

Consolidated, deduplicated backlog of issues surfaced across the 8 forensic variant code reviews at commit `bc2244b9fc121bbbfaf96302e51e7e0082083cac`:
- [review-code-addition-cls.md](file:///atelier/code/workshop/yessir/.agents/review-code-addition-cls.md)
- [review-code-addition-step_ar.md](file:///atelier/code/workshop/yessir/.agents/review-code-addition-step_ar.md)
- [review-code-addition-step_fork.md](file:///atelier/code/workshop/yessir/.agents/review-code-addition-step_fork.md)
- [review-code-addition-ar.md](file:///atelier/code/workshop/yessir/.agents/review-code-addition-ar.md)
- [review-code-addition-addressed_step_ar.md](file:///atelier/code/workshop/yessir/.agents/review-code-addition-addressed_step_ar.md)
- [review-code-addition-addressed_ar.md](file:///atelier/code/workshop/yessir/.agents/review-code-addition-addressed_ar.md)
- [review-code-sir_yes_sir-ar.md](file:///atelier/code/workshop/yessir/.agents/review-code-sir_yes_sir-ar.md)
- [review-code-sir_yes_sir-cls.md](file:///atelier/code/workshop/yessir/.agents/review-code-sir_yes_sir-cls.md)

**Headline baseline**: Zero bugs were found that corrupt `reference.py`, leak future targets across causal or padding masks, break train/eval coordinate parity, or alter primary `exact_match`. All items below affect error-taxonomy classification, secondary/diagnostic metrics, free-running helpers, study execution, configuration safety, or readability/tests.

---

## Quick Index

| ID | Group / Severity | Scope | Summary |
| :--- | :--- | :--- | :--- |
| **C-1** | High — Correctness & Metric Bugs | `addition/ar` | Premature `[EOS]` under `use_query_tokens=False` is misclassified as `out_of_alphabet` instead of `syntax_malformation` and penalises column `k` in `step_accuracy`. |
| **C-2** | High — Correctness & Metric Bugs | `addition/addressed_ar` | `build_addressed_ar_example` steps trace coordinates modulo oracle `addr(k)` width instead of `[IDX]` boundaries, desynchronising `rollout_addressed_ar` after wrong-width addresses; `parse_addressed_ar_trace` marks control bytes as `is_well_formed=True`. |
| **C-3** | High — Correctness & Metric Bugs | `addition` (`config.py`) | `AdditionConfig._split_seed` `+1` bump collides adjacent `ood_base{b}` / `ood_base{b+1}` splits and shifts one split's problems when `seed == eval_seed + offset`. |
| **C-4** | High — Correctness & Metric Bugs | `yessir` (`metrics.py`) | `MetricsLogger` appends to existing `train.jsonl` and `eval.jsonl` without truncation when `--out-dir` is reused, while `config.json` and `eval.json` overwrite. |
| **C-5** | High — Correctness & Metric Bugs | `yessir` (`tui.py`) | TUI left-arrow float stepping clamps at `step_f` (`0.05`) instead of `0.0`, preventing stepping `p_prior`, `p_chain`, and `eval_p_chain` back to `0.0`. |
| **C-6** | High — Correctness & Metric Bugs | `yessir`, `addition`, `sir_yes_sir` | `lr <= 0` validation accepts `NaN` and `+inf` in `AdditionConfig`, `SirYesSirConfig`, and `build_optimizer`. |
| **E-1** | Medium — Evaluation & Study Validity | `addition/addressed_ar` | Teacher-forced `evaluate_addressed_ar` heals broken model `addr(k)` counters intra-column, and `score_predictions` ignores `syntax_ok` in `step_accuracy` and `digit_accuracy` (can report `100%` `extrap_len_step_accuracy` when address counting completely fails). |
| **E-2** | Medium — Evaluation & Study Validity | `addition/{step_ar,addressed_step_ar,ar}` | Intra-step teacher forcing conditions `o_k` on oracle `c_{k+1}` under `CARRY_FIRST` (inflating `digit_accuracy` vs `SUM_FIRST`) and conditions `flag_k` / `[EOS]` on oracle `c_{k+1}` under both orders (inflating `halt_accuracy` vs `cls`/`step_fork`). |
| **E-3** | Medium — Evaluation & Study Validity | All 8 variants | Production `run.py` calls teacher-forced `evaluate_*` exclusively (`rollout_*` / `greedy_generate_*` are test-only), resetting carries/outputs across steps so `step_accuracy` (`eval.json`) and `carry_accuracy` measure single-step conditional accuracy rather than rollout survival. |
| **E-4** | Medium — Evaluation & Study Validity | `addition` (`permuted` split) | `permuted` pools `B = 2` with higher bases (`bases="2,4,8,10"`); in `B = 2`, `100%` of in-alphabet errors land in `prior_intrusion`, skewing pooled `prior_intrusion` with no per-base breakdown in `eval_splits.json`. |
| **E-5** | Medium — Evaluation & Study Validity | `sir_yes_sir` | Draft studies ask about unseen lengths `L'`, OOD shuffled alphabets on `resample_alphabet=False`, and `wrap_around_accuracy` (`s_{B-1} -> s_0`), none of which `sir_yes_sir` currently evaluates or saves checkpoints for; small-`B` regimes have `100%` verbatim train/eval overlap. |
| **E-6** | Medium — Evaluation & Study Validity | `sir_yes_sir/cls` | `ClsSupervision.ALL_STEPS` yields `L` consecutive items per problem with no shuffle buffer, training on `1 / L` distinct problems and packing correlated steps of the same problem into each minibatch. |
| **E-7** | Medium — Evaluation & Study Validity | `addition`, `sir_yes_sir` | Cross-variant training-budget & loss-scale asymmetries: `cls` sums per-head cross-entropies (`3x` / `2x` `train.jsonl` loss scale vs token-mean `SubspaceLMHead`); `ar` and `addressed_ar` train on `~14x` more column transitions per step than step-wise variants (and weight long `B = 2` traces `~26x–28x` over `L = 1`). |
| **K-1** | Low — Config Traps & Silent No-Ops | `addition/addressed_*` | `AdditionAddressedStepArConfig` and `AdditionAddressedArConfig` default to `EmbeddingMode.TOKEN_TYPE_PE`, injecting sinusoidal `position_ids = j` alongside `[IDX] addr(j)` unless explicitly overridden. |
| **K-2** | Low — Config Traps & Silent No-Ops | `addition`, `sir_yes_sir` | Default evaluation flags hide key study signals: `eval_permuted_canonical=False` omits `permuted` in `p_prior` sweeps; `default_eval_every` (`>= 2500`) skips mid-training checkpoints when `max_steps <= 2500` in `sir_yes_sir`. |
| **K-3** | Low — Config Traps & Silent No-Ops | `addition`, `sir_yes_sir` | Subtle `d_position`, `d_scratch`, and `fixed_d_model` interactions under `NOPE` / `TOKEN_TYPE_NOPE` (`--d-scratch 0` still gets `+32`/`+48` scratch dims when `fixed_d_model=True`; `--d-position 0` shrinks `d_model` when `fixed_d_model=True`; `d_position` is a silent no-op when `fixed_d_model=False`; `d_position=2` has `~6.28` period). |
| **K-4** | Low — Config Traps & Silent No-Ops | `addition`, `sir_yes_sir` | Conditional silent config no-ops (`p_terminal` when `all_steps=True`; `symbol_set`/`symbol_pool_size` when `p_prior=1.0` or `resample_alphabet=False`) record distinct `config.json` files and bypass TUI duplicate-run detection. |
| **K-5** | Low — Config Traps & Silent No-Ops | `addition/addressed_*`, `sir_yes_sir` | Flag interactions that shift distributions: `include_alphabet=False` clamps `eff_min = max(min_length, B)` per base (shifting `id` and `ood_base12` mean lengths and PRNG streams); toggling `resample_alphabet` with `symbol_pool_size=None` changes active symbol-pool size when `B < len(symbol_set)`. |
| **K-6** | Low — Config Traps & Silent No-Ops | `addition/step_fork`, `addition/addressed_*` | Direct callers of `AdditionStepForkDataset` must pass `3 * batch_size` or transitions split across batches; `addressed_*` has no `eval_batch_size` guard on `802`–`911`-token binary `L = 40` sequences. |
| **R-1** | Minor — Readability, Docs & Tests | `yessir` (`inspect_data.py`), `sir_yes_sir` | Dataset inspection CLI ergonomics: `format_token` prints `AdditionToken`s (`[S]`, `[C]`, `[F]`, `[IDX]`) as raw hex `0xf8..0xfb`; `sir_yes_sir` dataset CLIs omit `--symbol-set` and `--symbol-pool-size`. |
| **R-2** | Minor — Readability, Docs & Tests | `addition`, `sir_yes_sir` (`spec/`, `studies/`, `web/`) | Spec, study, and UI drift: both `spec/task.md` files cite `bfloat16` autocast when live code is pure `float32`; `studies/addition/query-tokens.json` mentions `[F]` on `ar` / `addressed_ar`; `web/` does not yet load `eval_splits.json` or `eval.jsonl`. |
| **R-3** | Minor — Readability, Docs & Tests | `addition`, `sir_yes_sir` | Minor halting-definition asymmetries across `evaluate_*` and `rollout_*` / `greedy_generate_*` (non-`FLAG_FINISHED` continuation in rollout vs exact `FLAG_CONTINUE` in eval; `k = 0` vs `k >= 1` boundary in `evaluate_addressed_ar`). |
| **R-4** | Minor — Readability, Docs & Tests | `sir_yes_sir`, `yessir` | Test coverage parity gaps on `sir_yes_sir/ar` (vs `sir_yes_sir/cls`) and minor defensive validation checks (`ParallelClassificationHead` `lengths` range, `_validate_cls_item` mask symmetry). |

---

## 1. High — Correctness & Metric Bugs

Code behaviour that directly contradicts the written specification or produces wrong behaviour on concrete, reachable inputs.

### C-1. `addition/ar`: Premature `[EOS]` under `use_query_tokens=False` is misclassified as `out_of_alphabet` instead of `syntax_malformation` (and penalises `step_accuracy`)
- **Where**: [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L115-L137), [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L132-L154), vs [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L158).
- **Affected variants**: `addition/ar` (`use_query_tokens=False`, the default).
- **Problem**: When `use_query_tokens=False`, there is no header or query token between column `k - 1` and column `k`. Emitting a premature `SpecialToken.EOS` (`0xF3`) right after completing column `k - 1` lands in `preds[2 * k]`, which `evaluate_ar` unpacks into `pred_o` or `pred_c` with `syntax_ok = True` while setting `halt_ok = False`. Because `classify_first_error` checks `value_pairs` before `pred.halt_ok` and `0xF3` is outside `alphabet` (`0x00..0xEF`), every premature `[EOS]` at a column boundary is classified as `FirstErrorType.OUT_OF_ALPHABET` instead of `FirstErrorType.SYNTAX_MALFORMATION` and reduces `step_accuracy` by `1 / (K + 1)`. Under `use_query_tokens=True` (where `[EOS]` lands in `pred_q0`, setting `syntax_ok = False`) and across all 5 sibling variants, the exact same premature halt after column `k - 1` is classified as `SYNTAX_MALFORMATION` and leaves column `k`'s teacher-forced `step_accuracy` unpenalised.
- **Suggested fix**: Decide how boundary premature `[EOS]` at `col_slice[0]` (`preds[2 * k]`) in `evaluate_ar` should be represented—e.g. treating `pred_t0 == eos` (a premature halt right at the column boundary) as a halting/syntax failure rather than an `out_of_alphabet` digit error, or aligning `classify_first_error` / `StepPrediction` across `use_query_tokens=False` and `use_query_tokens=True`—and add a unit test in `test_ar.py` covering premature `[EOS]` under both `use_query_tokens` settings.

### C-2. `addition/addressed_ar`: Coordinate desynchronisation in `build_addressed_ar_example` during `rollout_addressed_ar` and over-permissive `parse_addressed_ar_trace`
- **Where**: [inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/addressed_ar/inputs.py#L86-L117), [generation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/addressed_ar/generation.py#L35-L89), [generation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/addressed_ar/generation.py#L130-L157).
- **Affected variants**: `addition/addressed_ar` (free-running rollout and trace parser).
- **Problem**:
  1. `build_addressed_ar_example` never inspects `trace` for `AdditionToken.IDX` (`0xFB`) or `[S]` / `[C]`; it steps `slot` modulo a hardcoded oracle column width `1 + len(encode_address(k, alphabet_tokens)) + body_len`. During `rollout_addressed_ar` under `TOKEN_TYPE_PE` or `TOKEN_TYPE_NOPE`, if the model emits a wrong-width address (e.g. a 1-digit address instead of a 2-digit address at `k = B`), every subsequent emitted token gets shifted `token_type_ids` and `position_ids` (even tagging `[IDX]` as `TokenType.CARRY`).
  2. `parse_addressed_ar_trace` claims `is_well_formed` is `True` iff segments are well-formed `[IDX] addr(k) ...` entries followed by `[EOS]`, but never checks that `addr`, `sum_digit`, or `carry_out` lie in `SYMBOL_RANGE` (`0x00..0xEF`)—so `[IDX, SEP, PAD, BOS, EOS]` returns `is_well_formed = True`.
- **Suggested fix**: Drive `token_type_ids` and `position_ids` in `build_addressed_ar_example` from actual `[IDX]` segment boundaries in `trace` (matching oracle metadata whenever `trace` is well-formed), and validate in `parse_addressed_ar_trace` that `addr`, `sum_digit`, and `carry_out` contain only valid symbol bytes (not special/control tokens).

### C-3. `addition`: `AdditionConfig._split_seed` `+1` collision bump can collide adjacent `ood_base*` splits or shift one split across a seed sweep
- **Where**: [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L353-L387).
- **Affected variants**: All 6 `addition` variants.
- **Problem**: `_split_seed(offset)` computes `candidate = self.eval_seed + offset` and returns `candidate + 1 if candidate == self.seed else candidate`, using `offset = 1000 + b` for `ood_base{b}`.
  1. When consecutive OOD bases `(b, b + 1)` are configured and `seed == eval_seed + 1000 + b` (e.g. `bases="2,8", eval_ood_bases="3,4", eval_seed=999, seed=2002`), `ood_base3` bumps `2002 -> 2003`, colliding identically with `ood_base4`'s `2003`.
  2. In a multi-seed training sweep with fixed `eval_seed`, if one training `seed` happens to equal `eval_seed + offset`, that single run silently evaluates on a shifted 200-problem set (`candidate + 1`) unlike its sibling seeds.
- **Suggested fix**: Derive deterministic, non-overlapping split seeds via domain-separated strings or tuples (e.g. `f"{self.eval_seed}:eval:{split_name}"`, mirroring `f"{seed}:step"` in [inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L535)), which can never collide with an integer training `seed` or across splits and keeps eval sets strictly invariant across all integer `seed` sweeps.

### C-4. `yessir`: `MetricsLogger` appends to existing `train.jsonl` and `eval.jsonl` without truncation when `--out-dir` is reused
- **Where**: [metrics.py](file:///atelier/code/workshop/yessir/yessir/metrics.py#L24-L42), [train_and_evaluate.py (addition)](file:///atelier/code/workshop/yessir/experiments/addition/train_and_evaluate.py#L94-L104), [train_and_evaluate.py (sir_yes_sir)](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/train_and_evaluate.py#L58-L65).
- **Affected variants**: All 8 variants (`addition` and `sir_yes_sir`).
- **Problem**: `write_config_json`, `write_eval_json`, and `write_eval_splits_json` overwrite their target files, whereas `MetricsLogger` opens `self.path` with `"a"` on every `log()` call without truncating or rejecting an existing file in `__init__`. Re-running with an explicit `--out-dir <same_dir>` concatenates stale and new JSONL lines (producing non-monotonic `step` trajectories) alongside overwritten `config.json` and `eval.json`.
- **Suggested fix**: Truncate (unlink or open `"w"`) `self.path` once in `MetricsLogger.__init__` (or reject non-empty existing `--out-dir` up front).

### C-5. `yessir`: Interactive TUI left-arrow float stepping clamps at `0.05` instead of `0.0`
- **Where**: [tui.py](file:///atelier/code/workshop/yessir/yessir/tui.py#L136-L143).
- **Affected variants**: All variants when edited via `./yessir.sh run` TUI (`p_prior`, `p_chain`, `eval_p_chain`, `p_terminal`, `lr`).
- **Problem**: `_step_value` ends float stepping with `return max(step_f, next_f)` rather than respecting `spec.min_val`. For `0.0`-bounded probabilities (`step_f = 0.05`, `min_val = 0.0`), stepping right (`0.0 -> 0.05`) and then left clamps at `0.05` instead of returning to `0.0`.
- **Suggested fix**: Respect `spec.min_val` (and `spec.max_val` if present) when clamping float fields (`floor_f = float(spec.min_val) if spec.min_val is not None else step_f`), while keeping `lr > 0` strictly positive.

### C-6. `yessir`, `addition`, `sir_yes_sir`: Non-finite `lr` (`NaN`, `+inf`) passes validation
- **Where**: [training.py](file:///atelier/code/workshop/yessir/yessir/training.py#L87-L88), [config.py (addition)](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L269-L270), [config.py (sir_yes_sir)](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/config.py#L163-L164).
- **Affected variants**: All 8 variants.
- **Problem**: `if lr <= 0:` evaluates to `False` for `float("nan")` and `float("inf")` (unlike `not (0.0 <= p <= 1.0)`, which rejects `NaN` on probability fields).
- **Suggested fix**: Check `if not math.isfinite(lr) or lr <= 0:` in `build_optimizer`, `AdditionConfig.__post_init__`, and `SirYesSirConfig.__post_init__`.

---

## 2. Medium — Evaluation Semantics & Study-Validity Gaps

Places where the code implements the current spec or config schema, *or* lacks evaluation support, in ways that distort secondary/diagnostic metrics or prevent planned `studies/` from answering their stated research questions.

### E-1. `addition/addressed_ar`: Teacher-forced `evaluate_addressed_ar` heals broken `addr(k)` counters and `score_predictions` ignores `syntax_ok` in `step_accuracy` and `digit_accuracy`
- **Where**: [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/addressed_ar/evaluation.py#L71-L165), [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L193-L205), [embedding-modes-length.json](file:///atelier/code/workshop/yessir/studies/addition/embedding-modes-length.json), [variants-comparison.json](file:///atelier/code/workshop/yessir/studies/addition/variants-comparison.json).
- **Affected variants**: `addition/addressed_ar` (also touches `addition/ar` when `use_query_tokens=True`).
- **Problem**: `addressed_ar` is specifically meant to test whether the model can increment and emit `[IDX] addr(k)` on its own and retrieve `(a_k, b_k)` using its emitted address. Because `evaluate_addressed_ar` feeds the **oracle** `trace` into `batch.input_ids` and `score_predictions` computes `digit_ok` and `both_ok` (`step_accuracy`) without requiring `pred.syntax_ok`, a model whose address counter completely fails at `k >= 20` on `extrap_len` (`L = 21..40`) still receives the ground-truth `addr(k)` right before predicting `o_k`, `c_{k+1}`, and `[EOS]`—scoring **`extrap_len_step_accuracy = 1.0`, `digit_accuracy = 1.0`, and `halt_accuracy = 1.0`** alongside `exact_match = 0.0`!
- **Decision / fix options**:
  1. Require `pred.syntax_ok` in `both_ok` (so emitting a wrong `[IDX] addr(k)` or wrong query token fails that column in `step_accuracy`), and/or
  2. Add free-running rollout evaluation (or log rollout `step_accuracy` alongside teacher-forced `step_accuracy`) so `addressed_ar` actually tests self-generated address lookup beyond the first error.

### E-2. `addition/{step_ar,addressed_step_ar,ar}`: Intra-step teacher forcing creates a `CARRY_FIRST` vs `SUM_FIRST` asymmetry in `digit_accuracy` and inflates `halt_accuracy`
- **Where**: [evaluation.py (step_ar)](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/evaluation.py#L64-L129), [evaluation.py (addressed_step_ar)](file:///atelier/code/workshop/yessir/experiments/addition/variants/addressed_step_ar/evaluation.py#L64-L130), [evaluation.py (ar)](file:///atelier/code/workshop/yessir/experiments/addition/variants/ar/evaluation.py#L90-L137), [step-order.json](file:///atelier/code/workshop/yessir/studies/addition/step-order.json).
- **Affected variants**: `addition/step_ar`, `addition/addressed_step_ar`, `addition/ar`, `addition/addressed_ar`.
- **Problem**: Within column `k`, `evaluate_*` appends the **oracle** `(t_0, t_1)` before reading `preds_t1` and `preds_flag` (`[EOS]`).
  1. `step_accuracy` and `carry_accuracy` require `both_ok = (pred_t0 == t_0) and (pred_t1 == t_1)`, so they are algebraically invariant to within-step `t_0 -> t_1` teacher forcing.
  2. However, `digit_accuracy` only tests `pred_o == o_k`. Under `SUM_FIRST`, `o_k` is `t_0` (unconditioned on `c_{k+1}`); under `CARRY_FIRST`, `o_k` is `t_1` and conditions on the **oracle** outgoing carry `c_{k+1}` even when `pred_c` (`t_0`) was wrong—biasing `digit_accuracy` in favour of `CARRY_FIRST`.
  3. Similarly, `flag_k` (`step_ar`, `addressed_step_ar`) and final `[EOS]` (`ar`, `addressed_ar`) see the **oracle** `c_{k+1}` in context rather than `pred_c`, inflating `halt_accuracy` relative to `cls` and `step_fork`.
- **Suggested fix**: Either decode the 3 tokens within each column step autoregressively (`t_0 -> t_1 -> flag_k`, only 2–3 forward tokens per step in `step_ar`/`addressed_step_ar`), or explicitly document in `spec/task.md` §3 and `step-order.json` that `digit_accuracy` and `halt_accuracy` in sequential variants condition on oracle `t_0` / `(t_0, t_1)`.

### E-3. All 8 variants: Production `run.py` never calls `rollout_*` / `greedy_generate_*`, so `step_accuracy` in `eval.json` is teacher-forced conditional transition accuracy, not rollout survival
- **Where**: `run.py` and `evaluation.py` across all 6 `addition` variants and both `sir_yes_sir` variants; [task.md (addition)](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L143-L150), [task.md (sir_yes_sir)](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/spec/task.md#L76-L80), vs [reviewer-code.md](file:///atelier/code/workshop/yessir/.agents/reviewer-code.md#L33-L34).
- **Affected variants**: All 8 variants (`addition/*` and `sir_yes_sir/*`).
- **Problem**: While teacher-forced `exact_match` and `FirstErrorType` provably match free-running greedy rollout in exact arithmetic, `step_accuracy` (recorded alongside `exact_match` in `eval.json` and `eval.jsonl`) and `carry_accuracy` reset `c_k` and `o_0..o_{k-1}` to the oracle trajectory on every column `k = 0 .. K` and ignore premature halts.
- **Decision / fix options**: Decide whether to keep single-pass/per-step teacher forcing for speed (in which case rename or clearly document `step_accuracy` / `carry_accuracy` in `spec/task.md` and study descriptions as *oracle-conditioned single-step accuracy*, never rollout prefix accuracy), or run `rollout_*` / `greedy_generate_*` at final evaluation (or on a subset) to record true free-running rollout step accuracy.

### E-4. `addition`: Pooled `B = 2` in `permuted` split inflates `prior_intrusion` with no per-base breakdown
- **Where**: [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L402-L415), [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L90-L102), [prior-override.json](file:///atelier/code/workshop/yessir/studies/addition/prior-override.json).
- **Affected variants**: All 6 `addition` variants.
- **Problem**: As `spec/task.md` §3 notes, in `B = 2` the only wrong in-alphabet symbol on `alphabet = ('1', '0')` is the canonical ASCII digit, so `100%` of in-alphabet value errors in `B = 2` are classified as `prior_intrusion` (`0%` `procedural_error`), raising the uniform-error chance rate on `bases="2,4,8,10"` to `~39.7%` (and higher when errors concentrate on long binary chains). Because `eval_splits()` creates a single pooled `permuted` split across `train_bases`, `B >= 3` cannot be separated in `eval_splits.json`.
- **Suggested fix**: Exclude `B = 2` from the `permuted` split (`bases = tuple(b for b in train_bases if b >= 3)` when non-empty) or emit per-base `permuted_base{b}` splits / metrics in `eval_splits.json`.

### E-5. `sir_yes_sir`: Draft studies require OOD evaluation splits, `wrap_around_accuracy`, and small-universe overlap awareness not currently in `sir_yes_sir`
- **Where**: [config.py](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/config.py#L66-L80), [evaluation.py](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/evaluation.py#L22-L24), [train_and_evaluate.py](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/train_and_evaluate.py#L102-L130), [studies/sir_yes_sir/](file:///atelier/code/workshop/yessir/studies/sir_yes_sir).
- **Affected variants**: `sir_yes_sir/ar`, `sir_yes_sir/cls`.
- **Problem**:
  1. [sequence-length.json](file:///atelier/code/workshop/yessir/studies/sir_yes_sir/sequence-length.json) and [embedding-modes.json](file:///atelier/code/workshop/yessir/studies/sir_yes_sir/embedding-modes.json) ask how models trained at length `L` generalise to unseen lengths `L'`, whereas `SirYesSirConfig` has a single fixed `x_length` shared by train and eval (giving `cls` a constant positional halting shortcut at `k = L - 1`) and saves no `model.pt` weights.
  2. [memorisation-vs-in-context.json](file:///atelier/code/workshop/yessir/studies/sir_yes_sir/memorisation-vs-in-context.json) asks whether `resample_alphabet=False` memorises static transitions rather than attending to the prompt alphabet, but `eval_dataset` also uses `resample_alphabet=False` (no shuffled-alphabet OOD split).
  3. [alphabet-length.json](file:///atelier/code/workshop/yessir/studies/sir_yes_sir/alphabet-length.json) asks about `wrap_around_accuracy` (`x_k == s_{B-1} -> s_0`), which `EvalMetrics` does not compute (at `B = 240, L = 3`, `0%` wrap-around accuracy still yields `99.58%` `step_accuracy` and `98.76%` `exact_match`).
  4. In fixed-alphabet or small-pool regimes (`B = 4, L = 3, resample_alphabet=False` has `64` total problems; `symbol_pool_size=4, B = 4, L = 3` has `1,536`), `eval_seed != seed` still produces `100%` verbatim problem overlap with `64,000` training draws (and `B = 2` cannot test alphabet-order sensitivity because `[a, b]` and `[b, a]` have identical cyclic successors).
- **Suggested fix**: Either add optional OOD eval splits (`eval_extrap_x_length` / length-range training, `eval_permuted` when `resample_alphabet=False`) and `wrap_around_accuracy` to `sir_yes_sir`, or narrow those four draft study JSON questions to match what `sir_yes_sir` is intended to measure as a minimal reference experiment.

### E-6. `sir_yes_sir/cls`: `ClsSupervision.ALL_STEPS` sees `1 / L` distinct problems and packs correlated steps of the same problem into each minibatch
- **Where**: [dataset.py](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/variants/cls/dataset.py#L67-L88), [training.py](file:///atelier/code/workshop/yessir/yessir/training.py#L302-L308), [cls-training-passes.json](file:///atelier/code/workshop/yessir/studies/sir_yes_sir/cls-training-passes.json).
- **Affected variants**: `sir_yes_sir/cls` (note: `addition`'s `all_steps=True` shares the same `1 / (K + 1)` problem-rate property in [step-sampling.json](file:///atelier/code/workshop/yessir/studies/addition/step-sampling.json)).
- **Problem**: `all_steps` yields all `L` step examples of each problem back-to-back into `train()`, which slices `batch_size` items per step with no shuffle buffer. At matched `(batch_size, max_steps)`, `all_steps` sees `1 / L` as many distinct problems as `single_pass` and `random_step`, packs all `L` steps of each problem into the same gradient step (only `1` problem per batch when `L == batch_size`), and splits boundary problems when `batch_size % L != 0`.
- **Suggested fix**: Either scale `train_batch_size = L * batch_size` (or add a shuffle buffer / log `problems_seen`), or explicitly account for the `1 / L` problem-budget ratio in `cls-training-passes.json` and `step-sampling.json`.

### E-7. `addition` & `sir_yes_sir`: Cross-variant training loss scale (`cls` vs `SubspaceLMHead`) and per-step transition budget (`ar`/`addressed_ar` vs step-wise variants)
- **Where**: [training.py](file:///atelier/code/workshop/yessir/yessir/training.py#L98-L184), [variants-comparison.json](file:///atelier/code/workshop/yessir/studies/addition/variants-comparison.json), [cls-vs-ar.json](file:///atelier/code/workshop/yessir/studies/sir_yes_sir/cls-vs-ar.json).
- **Affected variants**: All 8 variants when compared across variant families.
- **Problem**:
  1. `parallel_classification_cross_entropy` **sums** per-head batch means (`3x` loss scale in `addition/cls`, `2x` in `sir_yes_sir/cls`, plus `262,656` / `393,728` untied head parameters with `254` unused classes on the flag head), whereas `next_token_cross_entropy` in `SubspaceLMHead` variants computes the per-token **mean** (`0` head parameters). `train.jsonl` scalar losses cannot be overlaid directly without dividing `cls` loss by `H`.
  2. At `batch_size = 64`, `cls`, `step_ar`, `step_fork` (`3 * 64`), and `addressed_step_ar` train on `64` single column transitions (`192` targets) per step, whereas `ar` and `addressed_ar` train on `64` full problems (`~875` transitions, `~2,000–6,000` targets) per step, and token-mean loss inside `addressed_ar` weights a `B = 2, L = 20` trace `~26x–28x` more heavily than a `B = 10, L = 1` trace.
- **Suggested fix**: Keep or normalise `parallel_classification_cross_entropy` deliberately (documenting `nats/step` vs `nats/token`), and log `tokens_seen` / `problems_seen` (or note the transition-budget ratio in `variants-comparison.json`) so step-count comparisons aren't confounded with data volume.

---

## 3. Low — Configuration Traps & Silent No-Ops

Defaults and flag combinations that can quietly invalidate an ablation or bypass duplicate-run detection.

### K-1. `addition/addressed_*`: Default `embedding_mode = EmbeddingMode.TOKEN_TYPE_PE` injects sinusoidal `position_ids = j` alongside `[IDX] addr(j)`
- **Where**: [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L111), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L508-L526), [embedding-modes-length.json](file:///atelier/code/workshop/yessir/studies/addition/embedding-modes-length.json), [zero-preamble.json](file:///atelier/code/workshop/yessir/studies/addition/zero-preamble.json).
- **Affected variants**: `addition/addressed_step_ar`, `addition/addressed_ar`.
- **Problem**: Both addressed configs inherit `embedding_mode = EmbeddingMode.TOKEN_TYPE_PE` unchanged from `AdditionConfig`. Running `./yessir.sh run addition addressed_step_ar` without `--embedding-mode nope` (or `token_type_nope`) gives the model 1-hop sinusoidal `position_ids = j` (providing direct numerical rank `j` on `s_j` under `--no-include-alphabet`, and introducing unseen sinusoidal positions `21..41` on `extrap_len`).
- **Suggested fix**: Consider defaulting `embedding_mode` to `EmbeddingMode.NOPE` (or `TOKEN_TYPE_NOPE`) on `AdditionAddressedStepArConfig` and `AdditionAddressedArConfig`, or highlight the required `--embedding-mode` flag in `zero-preamble.json` and `variants-comparison.json`.

### K-2. `addition` & `sir_yes_sir`: Default `eval_permuted_canonical=False` and `default_eval_every >= 2500` omit expected study measurements unless overridden
- **Where**: [config.py (addition)](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L158), [defaults.py](file:///atelier/code/workshop/yessir/yessir/defaults.py#L9-L14), [config.py (sir_yes_sir)](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/config.py#L85-L89).
- **Affected variants**: All 8 variants.
- **Problem**:
  1. In `addition`, `eval_permuted_canonical` defaults to `False`. Running a `p_prior > 0` sweep (e.g. `p_prior = 1.0` for `prior-override.json`) without `--eval-permuted-canonical` omits the `permuted` split, leaving `id` at `p_prior = 1.0` where ASCII memorisation scores `100%`. (Could auto-enable when `p_prior > 0` or default to `True`.)
  2. `default_eval_every(max_steps)` returns `max(2500, max_steps // 20)`. Because `SirYesSirConfig.max_steps` defaults to `1000`, `eval_every=None` evaluates only once at step `1000`, recording no mid-training learning curve for convergence-speed studies unless `--eval-every` is passed explicitly.

### K-3. `addition` & `sir_yes_sir`: Subtle `d_position`, `d_scratch`, and `fixed_d_model` interactions under `NOPE` / `TOKEN_TYPE_NOPE` (plus `d_position=2` wavelength)
- **Where**: [config.py (addition)](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L325-L341), [config.py (sir_yes_sir)](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/config.py#L200-L222), [embedding.py](file:///atelier/code/workshop/yessir/yessir/embedding.py#L49).
- **Affected variants**: All 8 variants.
- **Problem**:
  1. With `fixed_d_model=True` (default) and `embedding_mode in {TOKEN_TYPE_NOPE, NOPE}`, passing `--d-scratch 0` for a zero-scratchpad capacity ablation ([harness-vs-capacity.json](file:///atelier/code/workshop/yessir/studies/addition/harness-vs-capacity.json)) still sets `effective_d_scratch = 32` (`token_type_nope`) or `48` (`nope`), and `derived_summary()` does not display `effective_d_scratch`.
  2. Conversely, with `fixed_d_model=True` under `NOPE` / `TOKEN_TYPE_NOPE`, passing `--d-position 0` sets `dropped_position = 0` and unexpectedly shrinks `d_model` from `512` to `480` / `496`; with `fixed_d_model=False`, `d_position` under `NOPE` / `TOKEN_TYPE_NOPE` is a complete silent no-op.
  3. At minimal `d_position = 2`, `_SINUSOIDAL_BASE = 256.0` uses a single frequency `1.0 rad/pos` with period `2 * pi ~ 6.283`, causing near-collisions across distant positions (`pos = 0` vs `pos = 44`).
- **Suggested fix**: Include `effective_d_scratch` and `effective_d_position` in `derived_summary()`, and reject or warn on non-default `d_position` when `embedding_mode != TOKEN_TYPE_PE`.

### K-4. `addition` & `sir_yes_sir`: Conditional silent no-ops bypass TUI duplicate-run detection
- **Where**: [config.py (addition)](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L439-L456), [inputs.py (addition)](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L426-L430) & [#L537-L548](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L537-L548), [inputs.py (sir_yes_sir)](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/inputs.py#L166-L172), [tui.py](file:///atelier/code/workshop/yessir/yessir/tui.py#L291-L302).
- **Affected variants**: Step-wise `addition` variants (`cls`, `step_ar`, `step_fork`, `addressed_step_ar`) and `sir_yes_sir/*`.
- **Problem**:
  1. `p_terminal` is completely ignored when `all_steps=True`, yet any `p_terminal` value in `0.0..1.0` is accepted and serialised to `config.json`, so `matching_runs()` treats `(all_steps=True, p_terminal=0.0)` and `(all_steps=True, p_terminal=0.5)` as distinct configs despite bit-identical training streams.
  2. In `addition`, `symbol_set` and `symbol_pool_size` are ignored when `p_prior=1.0` and `eval_ood_bases=None` (even allowing `symbol_set="decimal", bases="2,12", p_prior=1.0` to draw `'A', 'B'` from `HEX[:12]`).
  3. In `sir_yes_sir`, `symbol_pool_size` (`>= alphabet_size`) is a silent no-op whenever `resample_alphabet=False`.
- **Suggested fix**: Normalise or reject inapplicable combinations in `__post_init__` (e.g. require default/unset `p_terminal` when `all_steps=True`, validate `max_train_base <= len(symbol_set.resolve())` even when `p_prior == 1.0`, and reject custom `symbol_pool_size != alphabet_size` when `resample_alphabet=False`).

### K-5. `addition/addressed_*` & `sir_yes_sir`: Flag toggles that silently shift length or symbol-pool distributions
- **Where**: [inputs.py (addition)](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L431-L436), [config.py (addition)](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L488-L506), [config.py (sir_yes_sir)](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/config.py#L187-L193).
- **Affected variants**: `addition/addressed_step_ar`, `addition/addressed_ar`, `sir_yes_sir/ar`, `sir_yes_sir/cls`.
- **Problem**:
  1. In `addition/addressed_*`, setting `include_alphabet=False` clamps `eff_min = max(min_length, base)` per problem. When `min_length = 1`, `ood_base12` evaluates operand lengths `12..20` (mean `16.0`) instead of `1..20` (mean `10.5`), desynchronises PRNG draws relative to `include_alphabet=True`, and silently collapses to fixed-length `L_a = L_b = B` if `max_length == B`. (Also note for [embedding-modes-length.json](file:///atelier/code/workshop/yessir/studies/addition/embedding-modes-length.json) that `L <= 20` training leaves MSD address symbols `s_2, s_3` in `B = 10, 12` and 6-bit addresses in `B = 2` unseen prior to `extrap_len` `L = 21..40`.)
  2. In `sir_yes_sir`, toggling `resample_alphabet` between `False` and `True` with `symbol_pool_size=None` and `B < len(symbol_set)` (e.g. `symbol_set="decimal", B=4`) changes the active vocabulary from `4` symbols (`'0'..'3'`) to `10` symbols (`'0'..'9'`) unless `symbol_pool_size=B` is set explicitly.
- **Suggested fix**: Document both controlled-comparison requirements (`min_length >= max_base` for matched `zero-preamble` splits; `symbol_pool_size = B` for matched `memorisation-vs-in-context` splits) in the study files or warn when `eff_min == max_length`.

### K-6. `addition/step_fork` & `addition/addressed_*`: Direct `AdditionStepForkDataset` batching contract and `addressed_*` memory scaling on long binary sequences
- **Where**: [dataset.py (step_fork)](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_fork/dataset.py#L69-L89), [run.py (step_fork)](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_fork/run.py#L47), [evaluation.py (addressed_step_ar)](file:///atelier/code/workshop/yessir/experiments/addition/variants/addressed_step_ar/evaluation.py#L58-L95).
- **Affected variants**: `addition/step_fork`, `addition/addressed_step_ar`, `addition/addressed_ar`.
- **Problem**:
  1. `run_addition_step_fork` multiplies `train_batch_size = 3 * config.batch_size`, whereas direct callers of `train(model, AdditionStepForkDataset(...), batch_size=64)` split every 21st transition's `([S], [C], [F])` triple across optimiser steps (`64 % 3 != 0`).
  2. At `B = 2, L = 40`, `addressed_step_ar` reaches `802` tokens and `addressed_ar` reaches `911` tokens; safe at default `eval_batch_size = 64` (`~1.3–1.7 GB` naive `fp32` attention buffer), but unguarded if a user passes a large `eval_batch_size` (e.g. `256`–`512`), and `evaluate_addressed_step_ar` materialises all `~6,200` step `Example`s in `flat_examples` upfront before chunking.

---

## 4. Minor — Readability, CLI Ergonomics, Spec/Study Drift, and Test Gaps

Non-functional polish, documentation alignment, and unit-test symmetry.

### R-1. Dataset inspection CLI readability & missing flags (`inspect_data.py`, `sir_yes_sir` dataset CLIs)
- **Where**: [inspect_data.py](file:///atelier/code/workshop/yessir/yessir/inspect_data.py#L14-L22), [inputs.py (sir_yes_sir)](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/inputs.py#L177-L211), [dataset.py (sir_yes_sir/ar)](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/variants/ar/dataset.py#L60-L68), [dataset.py (sir_yes_sir/cls)](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/variants/cls/dataset.py#L91-L110).
- **Problem**:
  1. `format_token` only resolves `SpecialToken` (`0xF0..0xF7`), so `AdditionToken`s (`[S]=0xF8`, `[C]=0xF9`, `[F]=0xFA`, `[IDX]=0xFB`) print as raw hex `0xf8..0xfb` (indistinguishable from symbol-pool hex bytes) when running `python -m experiments.addition.variants.<variant>.dataset`. Allowing an optional token-formatter callback or `IntEnum` table in `inspect_data.py` would print `[S]`, `[C]`, `[F]`, `[IDX]` cleanly.
  2. `python -m experiments.sir_yes_sir.variants.{ar,cls}.dataset` omits `--symbol-set` and `--symbol-pool-size`, always inspecting `SymbolSet.ALL`.

### R-2. Spec, study, and `web/` documentation/UI drift
- **Where**: [task.md (addition)](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L145), [task.md (sir_yes_sir)](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/spec/task.md#L78), [query-tokens.json](file:///atelier/code/workshop/yessir/studies/addition/query-tokens.json#L5), [data.ts](file:///atelier/code/workshop/yessir/web/src/lib/data.ts#L168-L176).
- **Problem**:
  1. Both `spec/task.md` files parenthetically attribute batch-shape `argmax` tie flips to `bfloat16` autocast, whereas `yessir/` and `experiments/` run strictly in `float32` with no `torch.autocast` (batch-shape differences come solely from `float32` GEMM/SDPA reduction order).
  2. [query-tokens.json](file:///atelier/code/workshop/yessir/studies/addition/query-tokens.json#L5) states that `use_query_tokens=True` inserts `([S], [C], [F])` across `step_ar`, `ar`, `addressed_step_ar`, and `addressed_ar`, whereas `ar` and `addressed_ar` omit `[F]` (halting via `[EOS]`) and supervise `[S]` and `[C]` in `next_token_cross_entropy` (unlike `step_ar` / `addressed_step_ar`, which mask `[S], [C], [F]` with `-100`).
  3. As already tracked in [addition-eval-splits-in-web.md](file:///atelier/code/workshop/yessir/.agents/addition-eval-splits-in-web.md), `web/` does not yet read `eval_splits.json` or `eval.jsonl`.

### R-3. Minor halting-check asymmetries between `evaluate_*` and `rollout_*` / `greedy_generate_*`
- **Where**: [generation.py (step_ar)](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/generation.py#L77), [generation.py (step_fork)](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_fork/generation.py#L64), [generation.py (sir_yes_sir/cls)](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/variants/cls/generation.py#L70-L71), [evaluation.py (addressed_ar)](file:///atelier/code/workshop/yessir/experiments/addition/variants/addressed_ar/evaluation.py#L149-L152).
- **Problem**:
  1. In `rollout_{cls,step_ar,step_fork,addressed_step_ar}` and `greedy_generate_cls`, non-terminal steps continue on any byte `!= FLAG_FINISHED` (and cap at `max(L_a, L_b) + 1` steps without a separate `halted_cleanly` boolean), whereas `evaluate_*` requires exact `pred_flag == FLAG_CONTINUE` (`245`).
  2. In `evaluate_addressed_ar`, a non-`[IDX]`, non-`[EOS]` syntax mistake right before column `k >= 1` (`after_col_pred != IDX`) sets `halt_ok = False` (`halt_accuracy = 0.0`), whereas the identical mistake before column `0` (`preds[0] != IDX`) leaves `halt_accuracy = 1.0` (and in `evaluate_ar`, non-`[S]` query mistakes between columns always leave `halt_accuracy = 1.0`).

### R-4. Unit-test symmetry gaps (`sir_yes_sir/ar` vs `cls`) and minor defensive checks
- **Where**: [test_evaluation_and_generation.py](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/tests/test_evaluation_and_generation.py#L250-L378), [test_sir_yes_sir_dataset.py](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/tests/test_sir_yes_sir_dataset.py#L196-L220), [test_run.py](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/tests/test_run.py#L145-L204), [heads.py](file:///atelier/code/workshop/yessir/yessir/heads.py#L102-L111), [evaluation.py (sir_yes_sir/cls)](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/variants/cls/evaluation.py#L25-L29).
- **Problem**:
  1. `sir_yes_sir/tests/` tests teacher-forced vs free-running `exact_match` equivalence, mixed-length batch invariance, `_validate_*_item` error branches, dataset `EmbeddingMode`/pool bounds, and `eval_every` training non-perturbation for `cls`, but omits the symmetric tests for `ar`.
  2. `ParallelClassificationHead.forward` checks `lengths.shape == (batch_size,)` without checking `(1 <= lengths) & (lengths <= hidden_states.size(1))` (`lengths = 0` would silently index `-1` if called outside `Example`/`collate`); `_validate_cls_item` in `sir_yes_sir/cls` checks `step[0] != IGNORE_INDEX` without asserting `(step[0] == IGNORE_INDEX) == (step[1] == IGNORE_INDEX)`.
