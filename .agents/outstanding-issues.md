# Unified Outstanding Issues (`HEAD` `e577c01`, code at `ede2632`)

Consolidated list of remaining code, study/spec framing, test, and UI items combining [outstanding-issues-gemini.md](file:///atelier/code/workshop/yessir/.agents/outstanding-issues-gemini.md), [outstanding-issues-claude.md](file:///atelier/code/workshop/yessir/.agents/outstanding-issues-claude.md), and [addition-eval-splits-in-web.md](file:///atelier/code/workshop/yessir/.agents/addition-eval-splits-in-web.md).

Nothing below makes `exact_match` wrong. Only **1.1** is a bug in live Python code; **2.1–2.3** are one-sentence study/spec framing clarifications, **3.1** is a minor test tightening, and **3.2** is the separate `web/` browser task.

---

## Quick Index

| ID | Category / Severity | Scope | Summary |
| :--- | :--- | :--- | :--- |
| **1.1** | Medium — Code & Metric Bug | `addition/addressed_ar` | `_parse_addressed_ar_rollout` shadows premature-body-`[EOS]` handling on multi-digit addresses (`k >= B`), inflating `halt_accuracy` to `1.0` at `k == K`, undercounting `address_accuracy` / `digit_accuracy`, and misclassifying `FirstErrorType`. |
| **2.1** | Low — Study / Spec Framing | `addition` (`harness-vs-capacity.json`, `step_ar.md`) | `cls` and `step_ar` default to `token_type_pe` (supplying `position_id = k`), whereas `ar` is a pure LM (`nope`); `ar.md` and `variants-comparison.json` already note `--embedding-mode nope`, while `harness-vs-capacity.json` does not yet. |
| **2.2** | Low — Study / Spec Framing | `sir_yes_sir` (`alphabet-length.json`, `spec/task.md`) | At `B = 2`, `[a, b]` and `[b, a]` define the same swap `a <-> b`, testing unordered 2-symbol lookup (on the 240-byte pool) or static swap memorisation (on a 2-symbol pool) rather than alphabet order (`B >= 3`). |
| **2.3** | Low — Study / Spec Framing | `sir_yes_sir` & `addition` (`cls-vs-ar.json`, `minimum-model-size.json`) | `ParallelClassificationHead` in `cls` adds `+262,656` (`sir_yes_sir`) / `+393,728` (`addition`) untied head parameters (including `254` unused flag classes) vs `0` in `SubspaceLMHead`—worth noting in capacity comparisons. |
| **3.1** | Minor — Test Tightening | `addition/tests`, `sir_yes_sir/tests` | `test_partially_trained_*_exact_match_matches_rollout` asserts aggregate rather than per-problem `exact_match` agreement between batched `evaluate_*` and single-problem `rollout_*` / `greedy_generate_*`. |
| **3.2** | Separate UI Backlog | `web/` | `web/` does not yet read `eval_splits.json`, `eval.jsonl`, `train.jsonl` `tokens_seen`, or `config.json` `started_at` (tracked in full in [addition-eval-splits-in-web.md](file:///atelier/code/workshop/yessir/.agents/addition-eval-splits-in-web.md)). |

---

## 1. Code & Metric Bug

### 1.1 `addition/addressed_ar`: `_parse_addressed_ar_rollout` shadows premature-body-`[EOS]` handling on multi-digit addresses (`k >= B`)

- **Where**: [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/addressed_ar/evaluation.py#L178-L241) (`_parse_addressed_ar_rollout`).
- **Impact**: Does **not** move `exact_match` or `step_accuracy`; **does** move `halt_accuracy`, `address_accuracy`, `digit_accuracy`, and `FirstErrorType` (`out_of_alphabet` vs `syntax_malformation`) whenever an `addressed_ar` rollout halts inside column `k >= B`.
- **What goes wrong**:
  - Line 178 checks `has_full_entry = len(seg_tokens) >= 1 + body_len` (hardcoding a 1-token minimum address width), before the `elif` branch at lines 212–236 that handles a valid `[IDX] + addr(k)` header followed by a premature `[EOS]` inside the column body.
  - Whenever `k >= B` (`len(expected_addr) >= 2`, e.g. `k >= 2` in base 2, `k >= 4` in base 4, `k >= 8` in base 8, `k >= 10` in base 10, or `extrap_len` columns `21..40`), emitting a **valid multi-digit `[IDX] + addr(k)` header** followed by a partial body (or even zero body tokens when `len(expected_addr) >= 3` and `use_query_tokens=False`) can satisfy `len(seg_tokens) >= 1 + body_len` while still having `len(seg_tokens) < len(expected_addr) + body_len`.
  - `has_full_entry` evaluates to `True`, shadowing lines 212–236 and slicing `pred_addr = tuple(seg_tokens[:-body_len])` and `body = seg_tokens[-body_len:]`. This steals trailing address token(s) into `body`, marks `address_ok = False` and `syntax_ok = False`, shifts `FirstErrorType` from `out_of_alphabet` to `syntax_malformation`, **and when `k == last_k` (`k == K`), sets `halt_ok = True` (`halt_accuracy = 1.0`) even though the rollout halted mid-column before emitting `c_{K+1}`**.
- **Concrete triggers (verified in Python on `alphabet = b"01"`)**:
  1. `b"111" + b"1"` (`4` steps `k = 0..3`, `addr(3) = b"11"`, `use_query_tokens=False`): columns `0..2` right, then `[IDX, '1', '1', o_3, EOS]` (missing `c_4`) -> reports `halt_accuracy = 1.0`, `address_accuracy = 0.75`, `digit_accuracy = 0.75`, `syntax_malformation = 1.0` instead of `halt_accuracy = 0.0`, `address_accuracy = 1.0`, `digit_accuracy = 1.0`, `out_of_alphabet = 1.0`.
  2. Same problem with `use_query_tokens=True` emitting `[IDX, '1', '1', [S], o_3, [C], EOS]` (`len(seg_tokens) = 5 >= 1 + 4`) -> reports `halt_accuracy = 1.0`, `address_accuracy = 0.75`, `syntax_malformation = 1.0` instead of `halt_accuracy = 0.0`, `address_accuracy = 1.0`, `out_of_alphabet = 1.0`.
  3. `b"1111" + b"1"` (`5` steps `k = 0..4`, `addr(4) = b"001"`, `use_query_tokens=False`): emitting header-only `[IDX, '0', '0', '1', EOS]` with zero body tokens (`len(seg_tokens) = 3 >= 1 + 2`) still reports `halt_accuracy = 1.0` and `address_accuracy = 0.80`.
- **Suggested fix** *(preserving wrong-width address parsing on non-final segments where the model emitted e.g. `[IDX, '0', o_2, c_3]` and continued with `[IDX, ...]`)*:
  ```python
  premature_body_eos = (
      is_last_seg
      and ended_with_eos
      and starts_with_idx
      and expected_addr is not None
      and len(expected_addr) <= len(seg_tokens) < len(expected_addr) + body_len
      and tuple(seg_tokens[: len(expected_addr)]) == expected_addr
  )
  has_full_entry = (len(seg_tokens) >= 1 + body_len) and not premature_body_eos
  ```

---

## 2. Study & Spec Framing Confounds

### 2.1 `addition`: Note `--embedding-mode nope` alongside default `token_type_pe` in `harness-vs-capacity.json`

- **Where**: [harness-vs-capacity.json](file:///atelier/code/workshop/yessir/studies/addition/harness-vs-capacity.json#L5), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L438) (`AdditionStepConfig.embedding_mode = EmbeddingMode.TOKEN_TYPE_PE`), and optionally [step_ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/step_ar.md#L3-L6). (Already noted in [ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/ar.md#L5) and [variants-comparison.json](file:///atelier/code/workshop/yessir/studies/addition/variants-comparison.json#L5).)
- **Issue**: `cls`, `step_fork`, and `step_ar` default to `TOKEN_TYPE_PE` (attaching `position_id = k` to `c_k`, query tokens, and `o_k`, matching `a_k` and `b_k` at zero offset), whereas `ar` is a pure `input_ids`-only LM (`nope`). In `harness-vs-capacity.json`, comparing `cls` / `step_ar` at default settings against `ar` on minimum `n_layers` and `d_scratch` varies both external state rewriting and 1-hop sinusoidal column alignment at once.
- **Suggested fix**: Keep `TOKEN_TYPE_PE` as the code default on `AdditionStepConfig`, and add a sentence in `harness-vs-capacity.json` (and optionally `step_ar.md`) stating that runs isolating state rewriting from external column alignment should also evaluate `cls` / `step_ar` with `--embedding-mode nope` (mirroring [cls-vs-ar.json](file:///atelier/code/workshop/yessir/studies/sir_yes_sir/cls-vs-ar.json#L5)).

### 2.2 `sir_yes_sir`: Note that `B = 2` tests unordered 2-symbol lookup (or static swap), not alphabet order

- **Where**: [alphabet-length.json](file:///atelier/code/workshop/yessir/studies/sir_yes_sir/alphabet-length.json#L5), [sir_yes_sir/spec/task.md](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/spec/task.md#L30-L38).
- **Issue**: Under cyclic successor on 2 symbols, `[a, b]` and `[b, a]` define the exact same swap (`a -> b`, `b -> a`). At the `B = 2` lower endpoint of `alphabet-length.json`:
  - With a restricted 2-symbol pool (`symbol_pool_size = 2`), a model can memorise static swap transitions in its weights without reading the prompt's alphabet section at all.
  - With the full 240-byte pool (`SymbolSet.ALL`), the model still must attend to the prompt's alphabet section to identify the partner symbol (e.g. when `x = ['q', 'q', 'q']` on `alphabet = ['q', '#']`), testing unordered 2-set membership rather than symbol order. Order-sensitive lookup starts at `B = 3`.
- **Suggested fix**: Add a one-sentence note on `B = 2` vs `B >= 3` in `alphabet-length.json` and `sir_yes_sir/spec/task.md`.

### 2.3 `sir_yes_sir` & `addition`: Note the untied `ParallelClassificationHead` parameter offset (`cls` vs `SubspaceLMHead`)

- **Where**: [sir_yes_sir/variants/cls/run.py](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/variants/cls/run.py#L48-L50) (`num_classes=(VOCAB_SIZE, VOCAB_SIZE)`), [addition/variants/cls/run.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/cls/run.py#L38-L41) (`num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE)`), [cls-vs-ar.json](file:///atelier/code/workshop/yessir/studies/sir_yes_sir/cls-vs-ar.json#L5), [sir_yes_sir/minimum-model-size.json](file:///atelier/code/workshop/yessir/studies/sir_yes_sir/minimum-model-size.json#L5), [addition/minimum-model-size.json](file:///atelier/code/workshop/yessir/studies/addition/minimum-model-size.json#L5).
- **Issue**: At default `d_model = 512`, `ParallelClassificationHead` adds `+262,656` parameters in `sir_yes_sir/cls` (`19,241,472` vs `18,978,816` in `ar`, including `254 * 512 = 130,048` unused weights on the 256-way flag head plus `132,608` active symbol/flag/norm weights) and `+393,728` parameters in `addition/cls` compared to `0` head parameters in `SubspaceLMHead`. Shrinking the flag head to 2 classes would add label-remapping code while still leaving the `+132k` / `+263k` untied symbol/carry head parameters (which matter most in tiny-model sweeps).
- **Suggested fix**: Keep `VOCAB_SIZE` class indexing so raw `SpecialToken` IDs work directly, and note the untied-head parameter offset in `cls-vs-ar.json` and both `minimum-model-size.json` study files.

---

## 3. Test Tightening & Web UI Backlog

### 3.1 Assert per-problem (not just aggregate) rollout agreement in `test_partially_trained_*_exact_match_matches_rollout`

- **Where**:
  - `sir_yes_sir`: [test_evaluation_and_generation.py](file:///atelier/code/workshop/yessir/experiments/sir_yes_sir/tests/test_evaluation_and_generation.py#L784-L886)
  - `addition`: [test_cls.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_cls.py#L415-L471), [test_step_fork.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_step_fork.py#L454-L507), [test_step_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_step_ar.py#L589-L644), [test_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_ar.py#L630-L681), [test_addressed_step_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_addressed_step_ar.py#L814-L880), [test_addressed_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_addressed_ar.py#L948-L1006).
- **Issue**: All 8 partially-trained real-`Transformer` equivalence tests compare the aggregate mean `rollout_matches / N == pytest.approx(metrics.exact_match)` across `30`–`40` problems, where equal numbers of `+1` and `-1` flips could cancel. Checking per-problem rollout outputs (e.g. comparing batched `_rollout_*_chunk` / `evaluate_*(..., num_problems=1)` problem by problem against `rollout_*` / `greedy_generate_*`) makes the invariant exact per problem.

### 3.2 `web/` visibility of `eval_splits.json`, `eval.jsonl`, `tokens_seen`, and `started_at`

- **Where**: [data.ts](file:///atelier/code/workshop/yessir/web/src/lib/data.ts#L168-L216), documented in full in [addition-eval-splits-in-web.md](file:///atelier/code/workshop/yessir/.agents/addition-eval-splits-in-web.md).
- **Summary**: `web/` currently reads only `config.json` (without `started_at`), `eval.json`, `train.jsonl` (without `tokens_seen`), and `README.md`; diagnostic `EvalMetrics` + `{split}_base{b}` slices in `eval_splits.json` and periodic checkpoint trajectories (plus `sir_yes_sir` `wrap_around_accuracy`) in `eval.jsonl` are not yet loaded or rendered in `web/`.
