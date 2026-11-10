# The sir-yes-sir experiment

`experiments/sir_yes_sir/` is a minimal, end-to-end smoke-test experiment. We are building it first—before rebuilding `experiments/addition/`—one small, reviewed piece at a time (see "Working with Louis" in `AGENTS.md`).

## Purpose

- **Exercise the full stack**: It touches every part of the shared `yessir/` machinery (tokenisation/sequences, model, training loop, evaluation, and result recording).
- **Separate infrastructure from science**: The task itself is trivial to learn and interpret, and trains quickly. If it fails to converge, the bug is in the code, not the hypothesis.
- **Living reference**: Kept up to date as the canonical example of how an experiment in this repo uses `yessir/`.
- **Learning vehicle**: Introduces each piece of shared machinery in the simplest possible context before applying it to `experiments/addition/`.

## The Task: In-Context Successor (next symbol mod B, no carry)

Given an ordered alphabet ("axiom") of `B` distinct symbols `[s_0, s_1, ..., s_{B-1}]` drawn from the symbol pool (`0x00`–`0xEF`) and an input sequence `x = [x_0, ..., x_{L-1}]` of symbols from that axiom, at each step `k` in `0 .. L-1` where `x_k = s_i`:

- **Primary output (`o_k`)**: the next symbol in the axiom, `s_{(i+1) mod B}`.
- **Wrap output (`c_{k+1}`)**: `s_1` if `i == B - 1` (wrap-around), else `s_0`.
- **Halting flag**: `FLAG_FINISHED` if `k == L - 1`, else `FLAG_CONTINUE`.

### Concrete examples

**Example 1 — Canonical base-4 alphabet:**
- Axiom (`B = 4`): `['0', '1', '2', '3']` (so `s_0 = '0'` means no wrap, `s_1 = '1'` means wrap)
- Input (`L = 3`): `['1', '3', '0']`
- Step-by-step outputs:
  - `k = 0`: `x_0 = '1'` -> next symbol `o_0 = '2'`, wrap `c_1 = '0'` (no wrap), flag `FLAG_CONTINUE`
  - `k = 1`: `x_1 = '3'` -> next symbol `o_1 = '0'` (wraps around to start of axiom), wrap `c_2 = '1'` (wrapped), flag `FLAG_CONTINUE`
  - `k = 2`: `x_2 = '0'` -> next symbol `o_2 = '1'`, wrap `c_3 = '0'` (no wrap), flag `FLAG_FINISHED` (last input position)
- Final output sequence: `['2', '0', '1']`

**Example 2 — Random 3-symbol alphabet:**
- Axiom (`B = 3`): `['q', '#', '7']` (here `s_0 = 'q'`, `s_1 = '#'`, `s_2 = '7'`)
- Input (`L = 4`): `['#', '7', 'q', '7']`
- Step-by-step outputs:
  - `k = 0`: `x_0 = '#'` -> `o_0 = '7'`, wrap `c_1 = 'q'` (`s_0`, no wrap), flag `FLAG_CONTINUE`
  - `k = 1`: `x_1 = '7'` -> `o_1 = 'q'` (wraps around), wrap `c_2 = '#'` (`s_1`, wrapped), flag `FLAG_CONTINUE`
  - `k = 2`: `x_2 = 'q'` -> `o_2 = '#'`, wrap `c_3 = 'q'` (`s_0`, no wrap), flag `FLAG_CONTINUE`
  - `k = 3`: `x_3 = '7'` -> `o_3 = 'q'` (wraps around), wrap `c_4 = '#'` (`s_1`, wrapped), flag `FLAG_FINISHED`
- Final output sequence: `['7', 'q', '#', 'q']`

Why this task:
- Requires attending to the in-context axiom (testing symbol binding, random vs. canonical vs. permuted alphabets, and unseen alphabet sizes `B`).
- Steps through a variable-length input `x` (testing positional/role embeddings across all modes and length generalisation).
- Emits two symbol targets plus a halting flag per step (exercising both `TiedIdentityHead` and `MultiHeadOutput`, and both state-machine and autoregressive harnesses).
- Each step `k` is purely local (no carry chain between steps), so convergence is fast and unambiguous.

## Shared `yessir/` Machinery to Build and Exercise

1. **Byte vocabulary & sequence primitives**:
   - 1-byte = 1-token (`0x00`–`0xFF`), reserving `0xF0`–`0xFF` for control/structural tokens and `0x00`–`0xEF` for numeral/symbol alphabets.
   - Alphabet sampling (`canonical_order`, `canonical_permuted`, `symbols_random`).
   - `Seq` container (`tokens`, optional `roles`, optional `sigs`) and right-padding batch collators (`collate`, `collate_labelled` with `IGNORE_LABEL = -100`).
2. **Transformer architecture**:
   - `SubspaceEmbedding` (dims `0..255` fixed one-hot byte identity; dims `256..511` supporting all three modes from the start: NoPE zeros, unit-norm learned Role + NoPE, and unit-norm Role + Sinusoidal PE).
   - Pre-LN `TransformerBlock` (`RMSNorm` with/without gain, causal `MultiHeadSelfAttention` via SDPA, `SwiGLU` and `GELU` FFNs, residual projection scaling).
   - Both output heads from the start: parameter-free `TiedIdentityHead` (sequence channel) and `MultiHeadOutput` (weight channel, reading at `lengths - 1`).
3. **Training utilities & loop**:
   - `seed_everything`, `build_optimizer` (AdamW with selective weight decay), `build_scheduler` (linear warm-up + cosine decay).
   - Masked cross-entropy loss (sequence channel) and multi-head cross-entropy loss (weight channel), gradient accumulation, `bfloat16` autocast, gradient clipping.
4. **Execution harnesses & evaluation**:
   - State-machine drivers (`sm_mh`, `sm_seq`, etc.) and autonomous autoregressive (`ar`) rollout driver.
   - Teacher-forced vs. free-running evaluation, exact match, step/token accuracy, halting accuracy, and error taxonomy (prior intrusion, out-of-alphabet, procedural error, syntax malformation).
5. **Logging & artefacts**:
   - `MetricsLogger` (`NaN`-safe `metrics.jsonl`), `config.json`, and `summary.json` written to `results/<experiment>/<run>/`.

## Plan

1. **Write the spec** for `sir_yes_sir` in `experiments/sir_yes_sir/spec/`.
2. **Build piece by piece**: introduce each component of `yessir/` alongside its use in `experiments/sir_yes_sir/`, with tests passing via `./yessir.sh test` and a review pause after each piece.
3. **Run and record**: run the completed smoke-test experiment and record its output in `results/sir_yes_sir/` (following `.agents/results.md`).
4. **Move on to `experiments/addition/`**: once the shared machinery is proven and understood in `sir_yes_sir`, rebuild `addition` on top of it.
