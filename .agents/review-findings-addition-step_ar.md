# Code Correctness Review: `addition` — `step_ar` Variant

## 1. Scope & Commit Reviewed

- **Commit audited (`HEAD`)**: `06bbf741cc6fedc7f87b79f9549a75332250e4d6` (Python code under `experiments/` and `yessir/` unchanged since `bf0f1bd451e3598eef50048a912ab34252579806`).
- **Historical runs in `results/addition/`**: No recorded runs exist yet for `step_ar`. The sole directory under `results/addition/` (`2026-11-17-1bf576e-merry-bream`, built at `1bf576ea6bb74e567220d115caa8618c5c9873a6`) is a 50-step smoke run for `"variant": "cls"` ([config.json](file:///atelier/code/workshop/yessir/results/addition/2026-11-17-1bf576e-merry-bream/config.json#L1-L41)). All 13 study definitions in `studies/addition/*.json` currently have `"status": "draft"` and `"runs": []`.
- **Specifications reviewed**:
  - [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L1-L177)
  - [step_ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/step_ar.md#L1-L113)
  - Cross-referenced sibling specs for variant comparisons: [cls.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/cls.md#L1-L82), [step_fork.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/step_fork.md#L1-L97), [ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/ar.md#L1-L67), [addressed_step_ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/addressed_step_ar.md#L1-L102).
- **Study definitions reviewed (`studies/addition/*.json`)**:
  - [carry-chain-injection.json](file:///atelier/code/workshop/yessir/studies/addition/carry-chain-injection.json#L1-L8), [embedding-modes-length.json](file:///atelier/code/workshop/yessir/studies/addition/embedding-modes-length.json#L1-L8), [grokking-and-overtraining.json](file:///atelier/code/workshop/yessir/studies/addition/grokking-and-overtraining.json#L1-L8), [harness-vs-capacity.json](file:///atelier/code/workshop/yessir/studies/addition/harness-vs-capacity.json#L1-L8), [minimum-model-size.json](file:///atelier/code/workshop/yessir/studies/addition/minimum-model-size.json#L1-L8), [prior-override.json](file:///atelier/code/workshop/yessir/studies/addition/prior-override.json#L1-L8), [query-tokens.json](file:///atelier/code/workshop/yessir/studies/addition/query-tokens.json#L1-L8), [seed-stability.json](file:///atelier/code/workshop/yessir/studies/addition/seed-stability.json#L1-L8), [step-order.json](file:///atelier/code/workshop/yessir/studies/addition/step-order.json#L1-L8), [step-sampling.json](file:///atelier/code/workshop/yessir/studies/addition/step-sampling.json#L1-L8), [training-bases.json](file:///atelier/code/workshop/yessir/studies/addition/training-bases.json#L1-L8), [variants-comparison.json](file:///atelier/code/workshop/yessir/studies/addition/variants-comparison.json#L1-L8), [zero-preamble.json](file:///atelier/code/workshop/yessir/studies/addition/zero-preamble.json#L1-L8).
- **Experiment & variant code audited**:
  - Shared `addition` modules: [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L1-L581), [reference.py](file:///atelier/code/workshop/yessir/experiments/addition/reference.py#L1-L186), [inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L1-L640), [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/evaluation.py#L1-L288), [train_and_evaluate.py](file:///atelier/code/workshop/yessir/experiments/addition/train_and_evaluate.py#L1-L239), [run.py](file:///atelier/code/workshop/yessir/experiments/addition/run.py#L1-L36).
  - `step_ar` variant modules: [inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/inputs.py#L1-L159), [dataset.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/dataset.py#L1-L156), [generation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/generation.py#L1-L82), [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/evaluation.py#L1-L243), [run.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/run.py#L1-L58).
- **Shared `yessir` modules & tests audited**:
  - [alphabet.py](file:///atelier/code/workshop/yessir/yessir/alphabet.py#L1-L100), [vocab.py](file:///atelier/code/workshop/yessir/yessir/vocab.py#L1-L48), [special_tokens.py](file:///atelier/code/workshop/yessir/yessir/special_tokens.py#L1-L20), [example.py](file:///atelier/code/workshop/yessir/yessir/example.py#L1-L90), [dataset.py](file:///atelier/code/workshop/yessir/yessir/dataset.py#L1-L82), [batch.py](file:///atelier/code/workshop/yessir/yessir/batch.py#L1-L234), [embedding.py](file:///atelier/code/workshop/yessir/yessir/embedding.py#L1-L156), [blocks.py](file:///atelier/code/workshop/yessir/yessir/blocks.py#L1-L317), [heads.py](file:///atelier/code/workshop/yessir/yessir/heads.py#L1-L116), [model.py](file:///atelier/code/workshop/yessir/yessir/model.py#L1-L175), [training.py](file:///atelier/code/workshop/yessir/yessir/training.py#L1-L331), [metrics.py](file:///atelier/code/workshop/yessir/yessir/metrics.py#L1-L43), [experiment.py](file:///atelier/code/workshop/yessir/yessir/experiment.py#L1-L478), [artefacts.py](file:///atelier/code/workshop/yessir/yessir/artefacts.py#L1-L406), [tui.py](file:///atelier/code/workshop/yessir/yessir/tui.py#L1-L581), [inspect_data.py](file:///atelier/code/workshop/yessir/yessir/inspect_data.py#L1-L166).
  - Tests in [test_step_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_step_ar.py#L1-L655), [test_reference.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_reference.py#L1-L231), [test_inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_inputs.py#L1-L431), [test_evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_evaluation.py#L1-L434), [test_config.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_config.py#L1-L524), [test_run.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_run.py#L1-L593), [test_convergence.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_convergence.py#L185-L243), and `yessir/tests/`.

---

## 2. Findings (Suspected Bugs, Spec Mismatches, Silent Confounds, or Edge-Case Traps)

**Headline summary**: Zero High- or Medium-severity bugs were found in `step_ar` or the shared `addition` / `yessir` pipeline. Sequence layouts, `token_type_ids`, `position_ids`, pre-shifted `labels` masking, causal attention without answer or padding leakage, genuinely free-running 3-pass intra-column and outer-loop inter-column rollout, single-step `oracle_step_accuracy`, `FirstErrorType` classification, RNG stream separation, and config plumbing all match [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md) and [step_ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/step_ar.md) under the configurations targeted by `studies/addition/*.json`.

Below are one **Low**-severity config-validation edge-case trap and five **Note**-level subtleties / experimental-design confounds uncovered during the audit.

---

### [Low] `symbol_pool_size` slicing on `SymbolSet.ALL` leaves `eval_permuted_canonical=True` enabled on unseen out-of-pool byte tokens

- **Where**:
  - [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L142), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L180-L215), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L300-L306), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L387-L404)
  - [inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L426-L438)
  - [embedding.py](file:///atelier/code/workshop/yessir/yessir/embedding.py#L131-L133), [blocks.py](file:///atelier/code/workshop/yessir/yessir/blocks.py#L250-L256)
- **Trigger & verification**:
  ```python
  from experiments.addition.config import AdditionStepArConfig
  from experiments.addition.inputs import iter_problems

  cfg = AdditionStepArConfig(bases="2,4", eval_ood_bases=None, symbol_pool_size=4)
  # cfg.symbol_pool == b"\x00\x01\x02\x03" (bytes 0..3)
  # Yet cfg.eval_permuted_canonical is True by default, so `permuted` is active
  # and draws from SymbolSet.HEX[:4] == b"0123" (ASCII bytes 48..51)!
  perm = [s for s in cfg.eval_splits() if s.name == "permuted"][0]
  prob = next(iter_problems(bases=perm.bases, symbols=cfg.symbol_pool, permuted_canonical=True, seed=perm.seed))
  assert prob.alphabet == (48, 50, 51, 49)  # completely disjoint from cfg.symbol_pool!
  ```
- **Mechanism & impact**:
  - `SymbolSet.ALL` is `bytes(range(0x00, 0xF0))`, whereas canonical numerals `SymbolSet.HEX` live at ASCII bytes `0x30..0x39` (`48..57`) and `0x41..0x46` (`65..70`). Slicing `symbol_pool_size <= 48` (or `<= 57` for base `10`, `<= 70` for bases `11..16`) on `symbol_set="all"` excludes some or all of `SymbolSet.HEX[:max_train_base]` from `cfg.symbol_pool`.
  - Because `SubspaceEmbedding` represents tokens as frozen `256`-d one-hot vectors and Layer 0 (`TransformerBlock`) projects `RMSNorm(x)` through bias-free linear matrices (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`), any byte `t` absent from training `input_ids` (when `p_prior == 0.0`) leaves column `t` of Layer 0's input projections and `attn_norm.weight[t]` untouched at random `N(0, 0.02^2)` initialisation (and in `SubspaceLMHead` / Layer `L - 1`, coordinate `t` is never a positive target).
  - `AdditionConfig.__post_init__` checks `max_train_base <= len(SymbolSet.HEX)` when `eval_permuted_canonical=True` ([config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L204-L208)) and rejects unused `symbol_pool_size` when `p_prior == 1.0` ([config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L191-L203)), but does not check `set(SymbolSet.HEX[:max_permuted_base]).issubset(set(self.symbol_pool))` when `p_prior == 0.0` (or `p_prior < 1.0`). Similarly, combining `0 < p_prior < 1` with `symbol_set="all", symbol_pool_size <= 48` places canonical problems (`0x30..0x39`) and random-pool problems (`0x00..0x2F`) in disjoint byte ranges, breaking [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L78)'s premise that the random pool already contains the ASCII numeral bytes.
  - **Headline / study impact**: Does **not** affect default runs (`symbol_set="all", symbol_pool_size=None`) or `symbol_set="decimal"` / `"hex"` runs where `symbol_pool` is a prefix of `SymbolSet.HEX`, so none of the 13 `studies/addition/*.json` designs is invalidated unless a user restricts `symbol_pool_size <= 57` on `symbol_set="all"` without passing `--no-eval-permuted-canonical`.

---

### [Note 1] Capacity confounds (`+48` non-scratch embedding dims and fixed `d_ff = 1376`) when sweeping `d_scratch` in `harness-vs-capacity` and `minimum-model-size`

- **Where**:
  - [harness-vs-capacity.json](file:///atelier/code/workshop/yessir/studies/addition/harness-vs-capacity.json#L5), [minimum-model-size.json](file:///atelier/code/workshop/yessir/studies/addition/minimum-model-size.json#L5)
  - [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L104-L105), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L328-L336), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L438-L445), [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L486-L495)
  - [model.py](file:///atelier/code/workshop/yessir/yessir/model.py#L20-L25)
- **Details**:
  1. [harness-vs-capacity.json](file:///atelier/code/workshop/yessir/studies/addition/harness-vs-capacity.json#L5) notes running `step_ar` with `--embedding-mode nope --no-fixed-d-model` alongside default `token_type_pe` when sweeping `d_scratch` down to `0` against `ar`. At `d_scratch = 0`, `step_ar --embedding-mode nope --no-fixed-d-model` matches `ar` at `d_model = 256`, whereas `step_ar` under default `token_type_pe` with `--d-scratch 0` retains the `16`-d `token_type` and `32`-d `position` slices (`d_model = 304`, giving `1,624,704` vs. `1,318,912` parameters per block at `d_ff = 1376`, or `+23.2%` per layer—plus 48 extra residual dimensions that later layers can overwrite as working memory).
  2. `d_ff` defaults to a constant `1376` in `AdditionConfig` ([config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L104)) rather than scaling as `~8/3 * d_model` when `fixed_d_model=False` shrinks `d_model` (documented in [model.py](file:///atelier/code/workshop/yessir/yessir/model.py#L20-L25), e.g. `d_ff = 1376` is `5.375 * d_model` at `d_model = 256`). Sweeps in `minimum-model-size` or `harness-vs-capacity` should pass `--d-ff` explicitly if constant FFN expansion ratio is intended.

---

### [Note 2] `evaluate_step_ar` does not apply `MAX_EVAL_CHUNK_TOKENS` sub-chunking (unlike `evaluate_addressed_step_ar`)

- **Where**:
  - [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/evaluation.py#L26-L85) vs. [addressed_step_ar/evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/addressed_step_ar/evaluation.py#L28-L60)
- **Details**:
  - `evaluate_addressed_step_ar` wraps evaluation forward passes in `_iter_token_bounded_chunks` (`MAX_EVAL_CHUNK_TOKENS = 65_536`) to cap `chunk_len * max_seq_len`, whereas `_predict_last_token_chunks` and `_predict_oracle_pairs_chunks` in `step_ar/evaluation.py` slice strictly by `batch_size`.
  - At default settings (`eval_batch_size = 64`, `eval_extrap_max_length = 40`, `B <= 12`, `use_query_tokens = True`), maximum sequence length is `1 + 12 + 1 + 40 + 1 + 40 + 1 + 1 + 1 + 40 + 5 = 143` tokens (`64 * 143 = 9,152` padded tokens per forward pass), so memory is tiny. Only extreme custom overrides (such as `eval_batch_size >= 512` combined with `B = 240` and `eval_extrap_max_length >= 200`) would benefit from token-bounded sub-chunking.

---

### [Note 3] Synthetic carry-chain injection (`sample_problem`) biases `b_i > a_i` at the carry-start column and never injects into single-operand tails

- **Where**:
  - [inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L449-L458)
  - [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L79)
- **Details**:
  1. At the carry-generating column `i`, `a_vals[i] = rng.randint(1, base - 1)` followed by `b_vals[i] = rng.randint(base - a_vals[i], base - 1)` assigns probability `1 / ((B - 1) * u)` to each pair `(u, v)` with `u + v >= B` (rather than `2 / (B * (B - 1))` uniform over the upper triangle). In base `10`, `(a_i = 1, b_i = 9)` occurs `9x` as often as `(a_i = 9, b_i = 1)`, giving `E[a_i] = 5.0` vs. `E[b_i] = 6.5` at column `i` (propagating columns `i + 1 .. i + m` are uniform and symmetric over `u + v == B - 1`).
  2. Consistent with [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L79), chains are injected within `n_shared = min(len_a, len_b)` (`i + 1 + m <= n_shared`). When `len_a != len_b`, carry propagation through the single-operand tail `min(len_a, len_b) .. max(len_a, len_b) - 1` (where the present operand must equal `s_{B-1}`) is never synthetically boosted and occurs at its natural `(1 / B)^d` rate.

---

### [Note 4] Standalone `AdditionStepArDataset` allows `all_steps=True` with non-default `p_terminal` without raising

- **Where**:
  - [dataset.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/dataset.py#L47-L57) vs. [config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L471-L474)
- **Details**:
  - `AdditionStepConfig.__post_init__` raises `ValueError` when `all_steps=True` and `p_terminal != 0.2`, preventing any experiment run via `AdditionStepArConfig` / `run_addition_step_ar` from silently ignoring `--p-terminal`.
  - Calling `AdditionStepArDataset(all_steps=True, p_terminal=0.8)` directly (or via `python -m experiments.addition.variants.step_ar.dataset --all-steps --p-terminal 0.8`) only validates `0.0 <= p_terminal <= 1.0` and silently ignores `p_terminal`.

---

### [Note 5] `evaluate_step_ar` and `rollout_step_ar` leave `model.training == False` on return

- **Where**:
  - [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/evaluation.py#L127), [generation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/generation.py#L39)
- **Details**:
  - Both functions call `model.eval()` without restoring the caller's prior `model.training` flag on exit.
  - Inside `train_and_evaluate` ([train_and_evaluate.py](file:///atelier/code/workshop/yessir/experiments/addition/train_and_evaluate.py#L160-L168)), every subsequent training step calls `_train_step_with_tokens` ([training.py](file:///atelier/code/workshop/yessir/yessir/training.py#L212)), which unconditionally calls `model.train()` (and `Transformer` contains no dropout or batch-normalisation layers), so mid-training evaluations have zero effect on subsequent training steps.

---

## 3. What Was Verified and Found Sound

1. **Ground-truth oracle (`reference.py`)**:
   - Verified [reference.py](file:///atelier/code/workshop/yessir/experiments/addition/reference.py#L69-L186) against [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L20-L69) on concrete examples and edge cases: `LSD -> MSD` ordering, unequal operand lengths (`v_a = 0` for `k >= L_a`, `v_b = 0` for `k >= L_b`), MSD zeros (`b"500" + b"2" -> b"700"` in 3 steps), single-column `0 + 0` (`K = 0`, 1 step) vs. `1 + 1` in base `2` (`K = 1`, 2 steps), full carry ripple `(B^L - 1) + 1` overflowing into column `K = N` with `o_K = s_1, c_{K+1} = s_0, flag_K = FLAG_FINISHED`, smallest base (`B = 2`), largest base (`B = 240`), and arbitrary symbol permutations.
2. **Problem & step sampling, disjointness, and RNG isolation (`inputs.py`, `config.py`, `train_and_evaluate.py`)**:
   - Verified `sample_problem` ([inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L411-L462)): `permuted_canonical=True` redraws until `alphabet != canonical` (so base `2` is always `('1', '0')`, never ascending `('0', '1')`).
   - Verified `iter_problem_steps` ([inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/inputs.py#L506-L568)): step-index selection uses a dedicated `random.Random(f"{seed}:step")` decoupled from `iter_problems(seed=seed)`, so varying `p_terminal` or `all_steps` leaves the underlying problem stream bit-for-bit identical; `p_terminal` selects `k = K` with probability `p_terminal` and uniform `0 .. K` otherwise (`p_terminal + (1 - p_terminal) / (K + 1)` total terminal rate).
   - Verified split separation in `eval_splits()` ([config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L338-L406)): domain-separated string seeds (`f"{eval_seed}:eval:{split_name}"`) keep all evaluation splits invariant across training `seed` sweeps and training `p_chain` ablations (`eval_p_chain` is separate from `p_chain`). Empirically verified `0 / 200` overlap between the first `100,000` default training problems and all five evaluation splits (`id`, `ood_base3`, `ood_base12`, `extrap_len`, `permuted`), plus strict structural disjointness on `ood_base*` (unseen `len(alphabet)`), `extrap_len` (`L_a, L_b >= 21 > 20`), and `permuted` vs. `p_prior=1.0` (`alphabet != canonical`).
   - Verified `permuted` base filtering ([config.py](file:///atelier/code/workshop/yessir/experiments/addition/config.py#L387-L392)): excludes `B = 2` whenever any `B >= 3` training base exists (avoiding the `1 / (B - 1) == 1.0` binary degeneracy documented in [task.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/task.md#L169)), falling back to `(2,)` only on pure-binary runs.
   - Verified mid-training evaluation RNG isolation ([train_and_evaluate.py](file:///atelier/code/workshop/yessir/experiments/addition/train_and_evaluate.py#L121-L233)): `train_stream` is a single persistent iterator with private `random.Random` instances, and evaluation runs under `@torch.no_grad()` with fresh per-split `random.Random(split.seed)` instances, leaving training losses, weights, optimiser state, and `tokens_seen` strictly invariant to `eval_every`.
3. **Tokenisation, sequence layout, causal masking, and embedding modes (`step_ar/inputs.py`, `embedding.py`, `batch.py`, `blocks.py`)**:
   - Verified `build_step_ar_example` and `build_step_ar_step_labelled_example` ([step_ar/inputs.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/inputs.py#L30-L159)) token-by-token against [step_ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/step_ar.md#L18-L99) across all `12` combinations of `StepOrder` (`sum_first`, `carry_first`), `use_query_tokens` (`False`, `True`), and `EmbeddingMode` (`token_type_pe`, `token_type_nope`, `nope`):
     - Shared prefix `[BOS] alphabet [SEP] a [SEP] b [SEP]` followed by `c_k` at `(TokenType.CARRY, k)`, `[SEP]` at `(TokenType.SPECIAL, 0)`, and `o_0 .. o_{k-1}` at `(TokenType.OUTPUT, 0 .. k - 1)`.
     - `use_query_tokens=False`: appends `[SEP]` at `(TokenType.SPECIAL, k)`, then `(t_0, t_1)` with `(TokenType.OUTPUT, k)` for `o_k` and `(TokenType.CARRY, k + 1)` for `c_{k+1}`; `labels` has length `P_k + 2` with `(IGNORE_INDEX,) * (P_k - 1) + (t_0, t_1, flag_k)`.
     - `use_query_tokens=True`: replaces trailing `[SEP]` with `Q_0` (`[S]` or `[C]`), interleaved `Q_0 t_0 Q_1 t_1 Q_2` (`Q_i` at `(TokenType.SPECIAL, k)`); `labels` has length `P_k + 4` with `(IGNORE_INDEX,) * (P_k - 1) + (t_0, IGNORE_INDEX, t_1, IGNORE_INDEX, flag_k)`, so harness-inserted query tokens are never supervised.
     - Rollout's 3 incremental step examples (`step_tokens = ()`, `(t_0,)`, `(t_0, t_1)`) are exact prefixes of the teacher-forced training sequence up to each supervised position, with identical `input_ids`, `token_type_ids`, and `position_ids`.
   - Verified zero answer or padding leakage: both training (`collate_labelled` in [batch.py](file:///atelier/code/workshop/yessir/yessir/batch.py#L181-L234)) and `step_ar` evaluation (`collate` in [batch.py](file:///atelier/code/workshop/yessir/yessir/batch.py#L139-L179)) use right-padding with causal SDPA (`is_causal=True` in [blocks.py](file:///atelier/code/workshop/yessir/yessir/blocks.py#L277-L280)) and strictly per-position `SubspaceEmbedding`, `RMSNorm`, `SwiGLUFeedForward`, and `SubspaceLMHead`. Note that `step_ar` does not use `KVCache` or left-padding—it executes 3 right-padded forward passes per active column, matching training forward passes identically.
4. **Config plumbing, loss normalisation, and token accounting (`config.py`, `step_ar/run.py`, `training.py`)**:
   - Traced all 31 fields of `AdditionStepArConfig` from CLI / JSON (`add_config_args`, `build_config_from_args` in [experiment.py](file:///atelier/code/workshop/yessir/yessir/experiment.py#L127-L197)) through `run_addition_step_ar` ([step_ar/run.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/run.py#L15-L50)) and `train_and_evaluate` ([train_and_evaluate.py](file:///atelier/code/workshop/yessir/experiments/addition/train_and_evaluate.py#L49-L239)): none is shadowed or a silent no-op (`d_position != 32` is rejected when `embedding_mode != TOKEN_TYPE_PE`; `p_terminal != 0.2` is rejected when `all_steps=True`; `fixed_d_model=True` adds dropped `16` and/or `32` slices into `effective_d_scratch` to hold `d_model = 512`).
   - Verified loss and `tokens_seen` ([training.py](file:///atelier/code/workshop/yessir/yessir/training.py#L154-L186), [training.py](file:///atelier/code/workshop/yessir/yessir/training.py#L216-L255)): `next_token_cross_entropy` averages strictly over non-`IGNORE_INDEX` labels (`3` targets per column transition under both `use_query_tokens=False` and `True`, i.e. `3 * batch_size = 192` targets per step at `batch_size = 64`), matching `cls` and `step_fork` in both `nats / target` loss normalisation and `tokens_seen` per step.
5. **Autoregressive rollout & metrics (`step_ar/generation.py`, `step_ar/evaluation.py`, `addition/evaluation.py`)**:
   - Verified `rollout_step_ar` ([generation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/generation.py#L16-L82)) and `evaluate_step_ar` ([step_ar/evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/step_ar/evaluation.py#L88-L243)):
     - **Intra-column autoregression**: 3 sequential greedy passes per column; `t_1` conditions on the model's own predicted `t_0` (never oracle `t_0`), and `flag_k` conditions on the model's own `(t_0, t_1)`.
     - **Inter-column feedback & halting**: column `k + 1` receives `carries[i] = pred_c` and `outputs[i].append(pred_o)` unpacked via `step_order`; rollout stops per problem on `pred_flag == FLAG_FINISHED` or the reference-free upper bound `max(len(a), len(b)) + 1` (never `len(oracle_steps)`). Out-of-alphabet or special tokens predicted in `pred_o` or `pred_c` are valid byte IDs (`0..255`) accepted by `build_step_ar_example` without crashing.
     - **`oracle_step_accuracy` equivalence**: verified algebraically and empirically across all `12` `(EmbeddingMode x StepOrder x use_query_tokens)` combinations on a `Transformer` that reusing rollout `(pred_o, pred_c)` on columns reached via the exact ground-truth `(carry_in, output[:k])` prefix plus single-pass teacher-forced `_predict_oracle_pairs_chunks` (`offsets = (-5, -3)` with query tokens, `(-3, -2)` without) on unreached/diverged columns gives exact bit-for-bit agreement (`< 1e-12` diff) with unconditional 2-pass and 1-pass evaluation across all `K + 1` reference columns.
     - **All 11 `EvalMetrics` fields & per-base slices**: verified `exact_match`, `step_accuracy`, `oracle_step_accuracy`, `digit_accuracy`, `carry_accuracy` (`None` when zero carry-active steps), `halt_accuracy` (requires `len(preds) == K + 1` and exact `FLAG_CONTINUE` on `k < K` and `FLAG_FINISHED` on `k == K`, so non-flag tokens or never-halting models on overflow columns still fail `halt_accuracy` and `exact_match`), `address_accuracy` (`None` on `step_ar`), and the 4-way `FirstErrorType` partition of unity (`exact_match + out_of_alphabet + prior_intrusion + procedural_error + syntax_malformation == 1.0`, checking `o_k` vs. `c_{k+1}` in `step_order`), plus `{split}_base{b}` disaggregation in `eval.jsonl` and `eval_splits.json`.

---

## 4. Test Coverage & Gaps

- **Executed test suites**:
  - `.venv-linux/bin/pytest experiments/addition/tests/test_step_ar.py experiments/addition/tests/test_reference.py experiments/addition/tests/test_inputs.py experiments/addition/tests/test_evaluation.py experiments/addition/tests/test_config.py experiments/addition/tests/test_run.py` -> **84 passed**.
  - `.venv-linux/bin/pytest yessir/tests/test_{alphabet,vocab,special_tokens,example,dataset,batch,embedding,blocks,heads,model,training,metrics,experiment,artefacts,tui}.py` -> **171 passed**.
- **What `tests/` thoroughly covers for `step_ar`**:
  - Exact token, `token_type_id`, `position_id`, and `labels` layouts for Examples 1 and 2 of [step_ar.md](file:///atelier/code/workshop/yessir/experiments/addition/spec/variants/step_ar.md) across all 4 `(step_order, use_query_tokens)` combinations and all 3 `EmbeddingMode`s, plus exact prefix agreement between the 3 incremental rollout passes and the teacher-forced sequence ([test_step_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_step_ar.py#L122-L307)).
  - Intra-step conditioning on the model's own wrong `c_1` under `carry_first` (catching intra-step teacher-forcing leaks), inter-step carry/output cascade vs. `oracle_step_accuracy` fallback, `batch_size=1` vs. `batch_size=5`/`7` invariance, out-of-alphabet/special token predictions (`SEP`, `255`) in rollout state, and `max(L_a, L_b) + 1` rollout cap ([test_step_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_step_ar.py#L356-L528)).
  - End-to-end equivalence between `rollout_step_ar` and `evaluate_step_ar` (`exact_match`) on a partially trained `Transformer` across all 3 `EmbeddingMode`s ([test_step_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_step_ar.py#L589-L655)), periodic `eval_every` invariance on `step_ar` ([test_run.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_run.py#L489-L593)), and CPU convergence (`>= 98%` exact match on reversed binary alphabets, [test_convergence.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_convergence.py#L185-L243)).
- **Remaining test gaps**:
  1. **Config validation gap for `symbol_pool_size` vs. `eval_permuted_canonical=True` / `p_prior`** (Finding `[Low]` above): [test_config.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_config.py#L316-L438) tests `symbol_pool_size` bounds and `p_prior == 1.0`, but does not test `symbol_set="all"` with small `symbol_pool_size` (`<= 48`) where `SymbolSet.HEX[:B]` falls outside `cfg.symbol_pool`.
  2. **Non-flag continuation token at `flag_k` in `evaluate_step_ar`**: [test_step_ar.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_step_ar.py#L356-L528) tests `FLAG_CONTINUE` and `FLAG_FINISHED` predictions, and [test_evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_evaluation.py#L81-L105) unit-tests `halt_ok=False` on `StepPrediction`, but no integration test feeds an arbitrary non-flag byte (e.g. `ord('0')` on `k < K`) through `evaluate_step_ar` to assert directly that rollout continues to `k + 1` while `halt_accuracy` and `exact_match` score `0.0` with `SYNTAX_MALFORMATION`.
  3. **CPU convergence with `use_query_tokens=True` or `sum_first` on `step_ar`**: [test_convergence.py](file:///atelier/code/workshop/yessir/experiments/addition/tests/test_convergence.py#L185-L243) runs `step_ar` convergence only under `(carry_first, use_query_tokens=False, token_type_pe)` (marked `@pytest.mark.convergence`, excluded from default CI).

---

## 5. What Was Not Inspected

- **`web/` frontend (Astro / TypeScript)** and **`yessir/vertex.py`** (cloud job submission): out of scope for `step_ar` model/training/evaluation semantics.
- **Full-trace implementation details of `ar`, `addressed_ar`, and `addressed_step_ar` (`variants/{ar,addressed_ar,addressed_step_ar}/`)** beyond the cross-variant interfaces (`config.py`, `evaluation.py`, and `addressed_step_ar/evaluation.py` chunking) needed to audit `step_ar` comparisons.
- **`.prototypes/transformadder/`**: excluded per `AGENTS.md` (live `experiments/addition/` was audited directly against `experiments/addition/spec/`).
