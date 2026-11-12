# The sir-yes-sir experiment

`experiments/sir_yes_sir/` is a minimal, end-to-end smoke-test experiment. We are building it first—before rebuilding `experiments/addition/`—one small, reviewed piece at a time (see "Working with Louis" in `AGENTS.md`).

## Purpose

- **Exercise the full stack**: It touches every part of the shared `yessir/` machinery (vocabulary/alphabets, examples/batching, model, training loop, evaluation, and result recording).
- **Force generic interfaces**: The variants listed below (embedding modes, FFNs, heads, harnesses) are deliberate. Having several of each in use at once is what forces the `yessir/` interfaces to be generic before `addition` depends on them. They count as concrete uses, not abstractions created in advance.
- **Separate infrastructure from science**: The task itself is trivial to learn and interpret, and trains quickly. If it fails to converge in-distribution, the bug is in the code, not the hypothesis.
- **Living reference**: Kept up to date as the canonical example of how an experiment in this repo uses `yessir/`.
- **Learning vehicle**: Introduces each piece of shared machinery in the simplest possible context before applying it to `experiments/addition/`.

## The Task: In-Context Successor (next symbol mod B, no carry)

Given an ordered alphabet of `B ≥ 2` distinct symbols `[s_0, s_1, ..., s_{B-1}]` drawn from the symbol pool (`0x00`–`0xEF`) and an input sequence `x = [x_0, ..., x_{L-1}]` of symbols from that alphabet, at each step `k` in `0 .. L-1` where `x_k = s_i`:

- **Primary output (`o_k`)**: the next symbol in the alphabet, `s_{(i+1) mod B}`.
- **Wrap output (`w_k`)**: `s_1` if `i == B - 1` (wrap-around), else `s_0`.
- **Halting flag**: `FLAG_FINISHED` if `k == L - 1`, else `FLAG_CONTINUE`.

### Concrete examples

**Example 1 — Canonical base-4 alphabet:**
- Alphabet (`B = 4`): `['0', '1', '2', '3']` (so `s_0 = '0'` means no wrap, `s_1 = '1'` means wrap)
- Input (`L = 3`): `['1', '3', '0']`
- Step-by-step outputs:
  - `k = 0`: `x_0 = '1'` -> next symbol `o_0 = '2'`, wrap `w_0 = '0'` (no wrap), flag `FLAG_CONTINUE`
  - `k = 1`: `x_1 = '3'` -> next symbol `o_1 = '0'` (wraps around to start of alphabet), wrap `w_1 = '1'` (wrapped), flag `FLAG_CONTINUE`
  - `k = 2`: `x_2 = '0'` -> next symbol `o_2 = '1'`, wrap `w_2 = '0'` (no wrap), flag `FLAG_FINISHED` (last input position)
- Primary output sequence: `['2', '0', '1']`

**Example 2 — Random 3-symbol alphabet:**
- Alphabet (`B = 3`): `['q', '#', '7']` (here `s_0 = 'q'`, `s_1 = '#'`, `s_2 = '7'`)
- Input (`L = 4`): `['#', '7', 'q', '7']`
- Step-by-step outputs:
  - `k = 0`: `x_0 = '#'` -> `o_0 = '7'`, wrap `w_0 = 'q'` (`s_0`, no wrap), flag `FLAG_CONTINUE`
  - `k = 1`: `x_1 = '7'` -> `o_1 = 'q'` (wraps around), wrap `w_1 = '#'` (`s_1`, wrapped), flag `FLAG_CONTINUE`
  - `k = 2`: `x_2 = 'q'` -> `o_2 = '#'`, wrap `w_2 = 'q'` (`s_0`, no wrap), flag `FLAG_CONTINUE`
  - `k = 3`: `x_3 = '7'` -> `o_3 = 'q'` (wraps around), wrap `w_3 = '#'` (`s_1`, wrapped), flag `FLAG_FINISHED`
- Primary output sequence: `['7', 'q', '#', 'q']`

Why this task:
- Requires attending to the in-context alphabet (testing symbol binding, random vs. canonical vs. permuted alphabets, and unseen alphabet sizes `B`).
- Steps through a variable-length input `x` (testing `token_type_ids` and `position_ids` across all embedding modes and length generalisation).
- Emits two symbol targets plus a halting flag per step (exercising both `TiedIdentityHead` and `MultiTaskHead`, and both state-machine and autoregressive harnesses).
- Each step `k` is purely local (no carry chain between steps), so convergence is fast and unambiguous.

## Shared `yessir/` Machinery to Build and Exercise

1. **Byte vocabulary, alphabets, examples & batching**:
   - `yessir/vocab.py`: 1-byte = 1-token (`0x00`–`0xFF`, `VOCAB_SIZE = 256`), partitioning bytes into `SYMBOL_RANGE` (`0x00`–`0xEF`), `SHARED_SPECIAL_RANGE` (`0xF0`–`0xF7`), and `EXPERIMENT_SPECIAL_RANGE` (`0xF8`–`0xFF`), plus `MAX_TOKEN_TYPES = 16` and `TOKEN_TYPE_RANGE`.
   - `yessir/special_tokens.py`: `SpecialToken(IntEnum)` (`SEP`, `PAD`, `BOS`, `EOS`, `STEP`, `FLAG_CONTINUE`, `FLAG_FINISHED`).
   - `yessir/alphabet.py`: `SymbolSet` (`ALL`, `DECIMAL`, `HEX`) and `create_alphabet(symbols, *, sample_size=None, rng=None)` supporting ordered, shuffled (permuted), and uniformly sampled alphabets.
   - `yessir/example.py`: `Example` container (`input_ids`, optional `token_type_ids`, optional `position_ids`).
   - `yessir/batch.py`: `Batch` container (`input_ids`, `lengths`, optional `token_type_ids`, optional `position_ids`) and right-padding `collate(examples)` (plus `collate_labelled` with `IGNORE_INDEX = -100` in Stage B).
   - `yessir/dataset.py` & `yessir/inspect_data.py`: `LabelledExample` (`example`, `labels`), `IterableDataset` protocol, and shared CLI inspection / JSONL dumping helpers.
2. **Transformer architecture**:
   - `SubspaceEmbedding` (`yessir/embedding.py`): parameter-free orthogonal slices concatenating `256`-d one-hot `input_ids`, optional `16`-d one-hot `token_type_ids` (`use_token_type`), optional unit-norm sinusoidal `position_ids` (`d_position`), and zero-initialised scratchpad working memory (`d_scratch`), giving `d_model = 256 + d_token_type + d_position + d_scratch`.
   - Pre-LN `TransformerBlock` (`yessir/blocks.py`: `RMSNorm` with/without gain, causal `MultiHeadSelfAttention` via SDPA, `SwiGLU` and `GELU` FFNs) and `Transformer` trunk (`yessir/model.py`: `1 / sqrt(2 * n_layers)` residual projection scaling).
   - Both output heads (`yessir/heads.py`): `MultiTaskHead` (parallel linear projections reading at `lengths - 1`) and parameter-free `TiedIdentityHead` (reading the `0..255` symbol subspace across all sequence positions).
3. **Training utilities & loop**:
   - `seed_everything`, `build_optimizer` (AdamW with selective weight decay), `build_scheduler` (linear warm-up + cosine decay).
   - Multitask cross-entropy loss (for `MultiTaskHead`) and masked next-token cross-entropy loss (for `TiedIdentityHead`), gradient accumulation, `bfloat16` autocast, gradient clipping.
4. **Execution harnesses & evaluation**:
   - State-machine drivers (`sm_mh`, read through `MultiTaskHead`; `sm_seq`, read through `TiedIdentityHead`) and autonomous autoregressive (`ar`) rollout driver.
   - Teacher-forced vs. free-running evaluation, exact match, step/token accuracy, halting accuracy, and error taxonomy (prior intrusion, out-of-alphabet, procedural error, syntax malformation).
5. **Logging & artefacts**:
   - `MetricsLogger` (`NaN`-safe `metrics.jsonl`), `config.json`, and `summary.json` written to `results/<experiment>/<run>/`.

## Plan

1. **Write the spec** for `sir_yes_sir` in `experiments/sir_yes_sir/spec/`.
2. **Build piece by piece**: introduce each component of `yessir/` alongside its use in `experiments/sir_yes_sir/`, with tests passing via `./yessir.sh test` and a review pause after each piece.
3. **Run and record**: run the completed smoke-test experiment and record its output in `results/sir_yes_sir/` (following `.agents/results.md`).
4. **Move on to `experiments/addition/`**: once the shared machinery is proven and understood in `sir_yes_sir`, rebuild `addition` on top of it.

## Staged build-out

The order is: first build one path end to end, then add a second variant of each piece. Each stage below is one reviewed piece, with `./yessir.sh test` passing. Every variant in the machinery list above is built before moving on to `addition`.

### Stage A: one path, end to end

1. **Spec** in `experiments/sir_yes_sir/spec/README.md` *(done)*: the task, input layout (`input_ids`, `token_type_ids`, `position_ids`), and exact-match scoring rule (covering `o_k`, `w_k`, and `flag_k`).
2. **Vocabulary, special tokens, and alphabets** *(done)*: `yessir/vocab.py`, `yessir/special_tokens.py`, `yessir/alphabet.py` (`SymbolSet`, `create_alphabet`), and tests in `yessir/tests/`.
3. **Reference rule** *(done)*: `reference_step(alphabet, x, k) -> (o_k, w_k, flag_k)` in `experiments/sir_yes_sir/reference.py` and tests reproducing Examples 1 and 2 in `experiments/sir_yes_sir/tests/test_reference.py`.
4. **`Example`, `Batch` / `collate`, and `sir_yes_sir` input builders** *(done)*: `yessir/example.py`, `yessir/batch.py`, and `experiments/sir_yes_sir/inputs.py` (`TokenType`, `EmbeddingMode`, `build_sm_example`, `build_ar_example`) with unit tests.
5. **Minimal model** *(done)*: `SubspaceEmbedding` (`yessir/embedding.py`), pre-LN block (`RMSNorm` with gain, `SwiGLUFeedForward`, causal `MultiHeadSelfAttention`, `TransformerBlock` in `yessir/blocks.py`), `MultiTaskHead` (`yessir/heads.py`), and `Transformer` (`yessir/model.py`) with unit tests.
6. **Dataset & minimal training** *(done)*: `LabelledExample` and `IterableDataset` (`yessir/dataset.py`, `yessir/inspect_data.py`, `experiments/sir_yes_sir/dataset.py`); `seed_everything`, `build_optimizer` (AdamW), `multitask_cross_entropy`, `train_step`, and `train` (`yessir/training.py`, fp32 on CPU, no scheduler).
7. **`sm_mh` harness and teacher-forced evaluation** *(done)*: `run_sm_mh` (`experiments/sir_yes_sir/harness.py`) and `EvalMetrics` / `evaluate_sm_mh` (`experiments/sir_yes_sir/evaluation.py`: exact match, step accuracy, halting accuracy).
8. **Convergence test** *(done)*: `experiments/sir_yes_sir/tests/test_convergence.py` trains a tiny model for a few hundred steps on CPU and asserts in-distribution exact match above 99%. This is the smoke test that stays live.
9. **Artefacts & CLI runner** *(done)*: `MetricsLogger` (`yessir/metrics.py`), `config.json` & `summary.json` + `<YYYY-MM-DD>-<short_sha>-<adj>-<noun>` run naming (`yessir/artefacts.py`), config & experiment contract (`yessir/experiment.py`, `yessir/tui.py`), and `experiments/sir_yes_sir/config.py` / `run.py`.

### Stage B: second variants (each one forces an interface to become generic)

10. Exercise the remaining `EmbeddingMode`s (`NOPE` and `TOKEN_TYPE_NOPE`, already supported by `SubspaceEmbedding` and `inputs.py`) in training and evaluation.
11. `GELU` FFN, and `RMSNorm` without gain.
12. `TiedIdentityHead`, `collate_labelled`, masked next-token cross-entropy, and `sm_seq` harness.
13. `ar` free-running generation driver: `build_ar_example` is already in `experiments/sir_yes_sir/inputs.py` (passive `TokenType.OUTPUT` section with sequential `position_ids = 0, 1, ...`). The `ar` harness will define `StepOrder` (`OUT_FIRST`: `(o_k, w_k, flag_k)` vs. `WRAP_FIRST`: `(w_k, o_k, flag_k)`), `build_ar_trace` (serialising `reference_step` across `k = 0 .. L - 1` into a flat target trace for training), and trace parsing/rollout until `FLAG_FINISHED`.
14. Training extras: warm-up + cosine scheduler, selective weight decay, gradient accumulation, `bfloat16` autocast, clipping.
15. Evaluation extras: free-running evaluation, plus an error taxonomy defined for this task (or dropped where a category doesn't apply).
16. Out-of-distribution splits: unseen `B`, longer `L`, permuted canonical alphabets (`create_alphabet(SymbolSet.DECIMAL[:B], rng=rng)`). These are measurements, not part of the convergence test.

### Stage C

17. A full run, recorded in `results/sir_yes_sir/` (following `.agents/results.md`).
18. Move on to `experiments/addition/`.
