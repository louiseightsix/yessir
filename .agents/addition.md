# The addition experiment

`experiments/addition/` is being rebuilt from scratch, one small piece at a time, with Louis reviewing each piece before the next (see "Working with Louis" in `AGENTS.md`).

- **Spec**: write it in `experiments/addition/spec/` as the pieces go in. It describes the live experiment only. Don't carry over the prototype's spec wholesale; take ideas from it deliberately and say where they came from.
- **The prototype** (`.prototypes/transformadder/`, Louis's internal notes; never mention it in a README) is the earlier, unreviewed exploration of the same question. Read it for ideas, pitfalls and results to compare against. Never copy its code into `experiments/` or `yessir/` in bulk. Treat its results as leads, not findings.
- **Results** from the live experiment go in `results/addition/` (see `.agents/results.md`).

## Leads, pitfalls, and corrections from `.prototypes/transformadder/`

When designing studies, implementing `experiments/addition/`, or comparing against the prototype's ~40 runs in `.prototypes/transformadder/results/`, keep these fact-checked observations in mind:

1. **Correction to `notes/round2.md` ("Breakthrough 1")**:
   - `notes/round2.md` claimed that `SM-MH-T4` (`cls` under `TOKEN_TYPE_NOPE`) collapsed to `0.0%` exact match without sinusoidal coordinates. That claim is contradicted by the actual run data in `.prototypes/transformadder/results/SM-MH-T4-L6-SwiGLU-Causal-multimodal-seed43252/metrics.jsonl`: `SM-MH-T4` grokked between step 35k (`7%` ID EM) and step 85k (`100.0%` ID EM, `37.0%` base-3 EM, and `34.0%` base-12 EM at step 90k; `99.8%` ID EM at step 100k).
   - Meanwhile, `SM-Fork-T4` (`step_fork` under `TOKEN_TYPE_NOPE`) only reached `48.8%` ID EM at 100k (`50.4%` on `SM-Fork-T6`), showing that 3 independent branched queries through a tied `SubspaceLMHead` (`[S]`, `[C]`, `[F]`) were harder to optimise than either 3 separate linear heads (`cls`) or sequential conditioning (`step_ar`).

2. **Delayed grokking vs. OOD overtraining across training steps**:
   - Coordinate-free and zero-preamble setups (`NOPE` Tier 5 and `include_alphabet=False` Tier 7) often looked flat for 70k–120k steps before grokking: `SM-SeqRev-T5` (seed 5329) went from `0%` ID EM at 25k to `76%` at 100k and `100%` at 425k–500k; `SM-SeqRev-T7` (seed 2903924) had `0%` ID EM at step 120k before grokking to `96%` ID EM at step 480k.
   - Conversely, on runs that converged earlier (`TOKEN_TYPE_NOPE` Tier 4), zero-shot OOD base accuracy (`B = 3`, `B = 12`) often **peaked around 75k–100k steps (when ID EM was ~70–85%) and then decayed** as cross-entropy loss was driven to `0.0000` on `{2, 4, 8, 10}` over 500k steps (e.g. `SM-Seq-T4` seed 43253498 base-12 EM dropped from `28.0%` at 100k to `1.6%` at 500k; `SM-SeqRev-T4` seed 32439 base-3 EM dropped from `47.0%` at 75k to `20.0%` at 500k, and base-12 EM from `35.0%` at 100k to `5.2%` at 500k).
   - Seed variance on OOD bases was also large even among runs at `~100%` ID EM (`SM-Seq-T4` base-3 EM: `80.7%` vs. `61.2%`; `SM-SeqRev-T7` base-3 EM: `77.2%` at 100k vs. `5.6%` at 500k).

3. **Representation-dependent `SUM_FIRST` vs. `CARRY_FIRST` trade-off**:
   - Under `TOKEN_TYPE_NOPE` (Tier 4), both emission orders solved ID addition (`99.8%`–`100%`), and `SUM_FIRST` (`SM-Seq-T4`) achieved higher base-3 OOD EM (`80.7%`) than `CARRY_FIRST` (`SM-SeqRev-T4`, `25.6%`).
   - Under harder representations—pure `NOPE` (Tier 5), explicit base-`B` addressing (`addressed_step_ar` Tier 6), and zero-preamble addressing (`include_alphabet=False` Tier 7)—`SUM_FIRST` stalled at `4.4%`–`28.8%` ID EM at 100k steps, whereas `CARRY_FIRST` reached `86.8%`–`96.4%` ID EM at 100k (`100%` at 500k on T5). Predicting the monotone carry threshold `c_{k+1}` first acts as an in-context scratchpad that reduces `o_k` to conditional subtraction.

4. **Harnessed `step_ar` ("cold counting") vs. full-trace `ar` (continuous 1-hop rollout)**:
   - Under `NOPE` in Round 1 (training bases `{2, 10}`), full-trace `AR-T2` and `AR-T3` reached `100%` ID EM (and `AR-T2` hit `41.0%` base-12 OOD EM, the prototype's best base-12 result), while harnessed `SM-Seq-T2` plateaued at `55.0%` ID EM because each column step `k` in `step_ar` starts a fresh forward pass that must count the `k` previous outputs `o_0 .. o_{k-1}` from scratch.
   - The prototype never tested `AR` on training bases `{2, 4, 8, 10}`, `TOKEN_TYPE_NOPE` (Tier 4), `addressed_ar` (Tier 6), or zero-preamble (Tier 7).

5. **Length extrapolation (`L = 21..40`) across all tiers**:
   - No run in the prototype exceeded `7%` exact match on `L = 21..40`.
   - `TOKEN_TYPE_PE` (Tier 1) kept `~61%–69%` `step_accuracy` on `L = 21..40` (solving columns `0..20` before failing on unseen sinusoidal positions `>= 21`), whereas `addressed_step_ar` (Tiers 6 & 7) dropped to `8%–28%` `step_accuracy` on `L = 21..40`.
   - A key confounder to check in `addressed_step_ar` / `addressed_ar`: when training on `L <= 20` in base `B = 10` or `12`, two-digit column addresses `0..19` only ever have `'0'` or `'1'` as their most-significant digit, so columns `20..39` introduce unseen leading address digits at test time.

6. **Implementation & systems notes (`step_fork` batching, VRAM, and simplifications)**:
   - **Column-step parity in `step_fork`**: A batch of `batch_size` column transitions in `step_fork` must contain `3 * batch_size` sequences (`[S]`, `[C]`, `[F]` for each sampled transition) so that each step trains on `batch_size` transitions and `3 * batch_size` target tokens, matching `cls` and `step_ar` (`notes/round2.md` §4).
   - **VRAM & evaluation chunking**: Combining `step_fork` (`3x` sequences per batch) or long `L = 21..40` evaluation problems with `addressed_*` (`~3x` sequence length from `[IDX] addr(j)` tokens, up to ~900 tokens in base 2) increases token volume per batch by up to `~9x` (`notes/round3.md` §6–§7). Keep evaluation batch/chunk sizes bounded on `addressed_*`.
   - **Deliberate simplifications from the prototype**:
     - The prototype's Tier 2 (5 section-header delimiters `[AXIOM]`, `[A]`, `[B]`, `[CARRY]`, `[OUT]` under `NOPE`) and Tier 5 (dual `|` + `*` origin delimiter) are replaced in the live spec by the shared `[SEP]` delimiter across `EmbeddingMode` (`TOKEN_TYPE_PE`, `TOKEN_TYPE_NOPE`, `NOPE`) plus optional `[S]`, `[C]`, `[F]` query tokens (`use_query_tokens`).
     - In Round 1, `GELU` (`99.9%` ID EM) and `SwiGLU` (`100.0%` ID EM) performed identically on `SM-MH-T1`, so the live `yessir` backbone standardises on `SwiGLU`.

