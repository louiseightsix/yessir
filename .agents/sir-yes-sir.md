# The sir-yes-sir experiment

`experiments/sir_yes_sir/` is a minimal, end-to-end smoke-test experiment. We are building it first—before rebuilding `experiments/addition/`—one small, reviewed piece at a time (see "Working with Louis" in `AGENTS.md`).

## Purpose

- **Exercise the full stack**: It touches every part of the shared `yessir/` machinery (tokenisation/sequences, model, training loop, evaluation, and result recording).
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
- Steps through a variable-length input `x` (testing positional/role embeddings across all modes and length generalisation).
- Emits two symbol targets plus a halting flag per step (exercising both `TiedIdentityHead` and `MultiTaskHead`, and both state-machine and autoregressive harnesses).
- Each step `k` is purely local (no carry chain between steps), so convergence is fast and unambiguous.

## Shared `yessir/` Machinery to Build and Exercise

1. **Byte vocabulary & sequence primitives**:
   - 1-byte = 1-token (`0x00`–`0xFF`), reserving `0xF0`–`0xFF` for special tokens and `0x00`–`0xEF` for numeral/symbol alphabets.
   - Alphabet sampling (`canonical_order`, `canonical_permuted`, `symbols_random`).
   - `Seq` container (`input_ids`, optional `role_ids`, optional `position_ids`) and right-padding batch collators (`collate`, `collate_labelled` with `IGNORE_INDEX = -100`, as in PyTorch's `ignore_index`).
2. **Transformer architecture**:
   - `SubspaceEmbedding` (dims `0..255` fixed one-hot byte identity; dims `256..511` supporting all three modes: NoPE zeros, unit-norm learned Role + NoPE, and unit-norm Role + Sinusoidal PE).
   - Pre-LN `TransformerBlock` (`RMSNorm` with/without gain, causal `MultiHeadSelfAttention` via SDPA, `SwiGLU` and `GELU` FFNs, residual projection scaling).
   - Both output heads: parameter-free `TiedIdentityHead` (sequence channel) and `MultiTaskHead` (weight channel, reading at `lengths - 1`).
3. **Training utilities & loop**:
   - `seed_everything`, `build_optimizer` (AdamW with selective weight decay), `build_scheduler` (linear warm-up + cosine decay).
   - Masked cross-entropy loss (sequence channel) and multi-head cross-entropy loss (weight channel), gradient accumulation, `bfloat16` autocast, gradient clipping.
4. **Execution harnesses & evaluation**:
   - State-machine drivers (`sm_mh`, read through the multi-task head; `sm_seq`, read through the sequence channel; etc.) and autonomous autoregressive (`ar`) rollout driver.
   - Teacher-forced vs. free-running evaluation, exact match, step/token accuracy, halting accuracy, and error taxonomy (prior intrusion, out-of-alphabet, procedural error, syntax malformation).
5. **Logging & artefacts**:
   - `MetricsLogger` (`NaN`-safe `metrics.jsonl`), `config.json`, and `summary.json` written to `results/<experiment>/<run>/`.

## Plan

1. **Write the spec** for `sir_yes_sir` in `experiments/sir_yes_sir/spec/`.
2. **Build piece by piece**: introduce each component of `yessir/` alongside its use in `experiments/sir_yes_sir/`, with tests passing via `./yessir.sh test` and a review pause after each piece.
3. **Run and record**: run the completed smoke-test experiment and record its output in `results/sir_yes_sir/` (following `.agents/results.md`).
4. **Move on to `experiments/addition/`**: once the shared machinery is proven and understood in `sir_yes_sir`, rebuild `addition` on top of it.

## Proposed staged build-out

The order is: first build one path end to end, then add a second variant of each piece. Each stage below is one reviewed piece, with `./yessir.sh test` passing. Every variant in the machinery list above is built before moving on to `addition`.

### Stage A: one path, end to end

1. **Spec skeleton** in `experiments/sir_yes_sir/spec/`: the task, plus the prompt layout (how the alphabet and `x` are laid out, and the role of each position). Louis decides the layout.
2. **Vocabulary and alphabets**: byte constants (symbol pool and special tokens), `symbols_random`, `canonical_order`. Tests cover the ranges, distinct symbols and `B ≥ 2`.
3. **Reference rule**: `step(alphabet, x, k) -> (o_k, w_k, flag)`. Tests reproduce Examples 1 and 2 exactly.
4. **`Seq`, the SM-MH prompt builder and `collate`**. Tests check shapes and padding.
5. **Minimal model**: `SubspaceEmbedding` in one mode (proposed: Role + Sinusoidal, the mode most likely to converge), pre-LN block (`RMSNorm` with gain, `SwiGLU`, causal SDPA attention) and `MultiTaskHead`. Shape tests.
6. **Minimal training**: `seed_everything`, AdamW, multi-head cross-entropy, fp32 on CPU, no scheduler.
7. **`sm_mh` harness and teacher-forced evaluation**: exact match, step accuracy, halting accuracy.
8. **Convergence test**: a pytest that trains a tiny model for a few hundred steps on CPU and asserts in-distribution exact match above 99%. This is the smoke test that stays live.
9. **Artefacts**: `MetricsLogger` (`metrics.jsonl`), `config.json`, `summary.json`.

### Stage B: second variants (each one forces an interface to become generic)

10. The remaining embedding modes: NoPE, and Role + NoPE.
11. `GELU` FFN, and `RMSNorm` without gain.
12. Sequence channel: `TiedIdentityHead`, `collate_labelled`, masked cross-entropy, `sm_seq` harness.
13. `ar` prompt and free-running generation driver.
14. Training extras: warm-up + cosine scheduler, selective weight decay, gradient accumulation, `bfloat16` autocast, clipping.
15. Evaluation extras: free-running evaluation, plus an error taxonomy defined for this task (or dropped where a category doesn't apply).
16. Out-of-distribution splits: unseen `B`, longer `L`, `canonical_permuted`. These are measurements, not part of the convergence test.

### Stage C

17. A full run, recorded in `results/sir_yes_sir/` (following `.agents/results.md`).
18. Move on to `experiments/addition/`.

### Open questions

- What does exact match cover: only the `o_k`, or also `w_k` and the halting flag?
- What is `canonical_order` for `B > 10`?
- What is the order of emissions in `ar`? If `w_k` comes after `o_k`, the model can derive it from `o_k`, so the wrap output becomes a weak check. In `sm_mh`, both heads read the same hidden state, so each has to compute its answer independently.
