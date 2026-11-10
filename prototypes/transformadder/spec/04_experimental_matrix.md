# Experimental Matrix: Design Space, Admissibility, Controls and Hypotheses

**Specification Document 04**

---

## 1. Design Space

Every configuration is a point in
$$\mathcal{M} = \langle \mathcal{S}_{\text{agency}}, \mathcal{S}_{\text{topology}}, \mathcal{S}_{\text{channel}}, \mathcal{S}_{\text{repr}}, \mathcal{S}_{\text{dist}}, p_{\text{prior}} \rangle$$

1. **Agency** ([`02`](02_mechanics.md) §4): `A-Harnessed` | `A-Autonomous`.
2. **Topology** ([`02`](02_mechanics.md) §5): `T-Factored` | `T-SumFirst` | `T-CarryFirst`.
3. **Channel** ([`02`](02_mechanics.md) §6): `C-Weight` | `C-Sequence`.
4. **Representation** ([`03`](03_representations.md) §3): `sinusoidal_pe` (T1) | `semantic_nope` (T2) | `generic_nope` (T3) | `role_embed_nope` (T4) | `semantic_origin_nope` (T5) | `addressed_kv` (T6) | `zero_preamble_kv` (T7).
5. **Curriculum** ([`03`](03_representations.md) §4): `D-Standard` | `D-FullSpan`.
6. **Prior dial** ([`03`](03_representations.md) §2.2): $p_{\text{prior}} \in \{0, 0.25, 0.5, 0.75, 0.9\}$.

Secondary levers (depth, FFN, attention mask, training bases) are in §4.

---

## 2. Admissibility

| Combination | Status | Reason |
| :--- | :---: | :--- |
| `C-Weight` × `T-SumFirst` / `T-CarryFirst` | **Invalid** | Parallel linear heads come from one forward pass; one head cannot condition on another's output. |
| `C-Weight` × `A-Autonomous` | **Invalid** | Autoregressive generation emits a single token stream. |
| `T-Factored` × `A-Autonomous` | **Out of scope** | Definable with parallel heads per trace position, but not a standard causal rollout. |
| `sinusoidal_pe` with NoPE layouts | **Invalid** | Continuous PE is a coordinate side channel; NoPE layouts have none. |
| `addressed_kv` without `ExplicitKV` query | **Invalid** | Addressed layouts need the harness to state the query address. |
| `zero_preamble_kv` × `D-Standard` | **Invalid** | Short operands ($L < B$) don't expose every alphabet symbol, so the alphabet is unrecoverable. |

### Six mechanics tuples

| Identifier | Agency | Topology | Channel | Description |
| :--- | :---: | :---: | :---: | :--- |
| `SM-MultiHead` (`SM-MH`) | Harnessed | Factored | Weight | One pass per column; 3 linear heads. |
| `SM-Fork` | Harnessed | Factored | Sequence | 3 parallel trigger queries via tied head. |
| `SM-Seq` | Harnessed | SumFirst | Sequence | Micro-AR $o_k \to c_{k+1} \to \text{FLAG}_k$. |
| `SM-SeqRev` | Harnessed | CarryFirst | Sequence | Micro-AR $c_{k+1} \to o_k \to \text{FLAG}_k$. |
| `AR-SumFirst` | Autonomous | SumFirst | Sequence | Trace `([S] o_k [C] c_{k+1})* [FINISHED]`. |
| `AR-CarryFirst` | Autonomous | CarryFirst | Sequence | Trace `([C] c_{k+1} [S] o_k)* [FINISHED]`. |

---

## 3. Model Matrix

The primary matrix at $p_{\text{prior}} = 0$, 6 layers, SwiGLU, causal attention:

| Model | Mechanics | Layout | Curriculum | Question |
| :--- | :--- | :--- | :---: | :--- |
| `SM-MH-SinPE` | SM-MultiHead | `sinusoidal_pe` | Standard | Baseline: explicit coordinates, linear heads. |
| `SM-Seq-SinPE` | SM-Seq | `sinusoidal_pe` | Standard | Sequential emission under explicit coordinates. |
| `AR-SumFirst-Sem` | AR-SumFirst | `semantic_nope` | Standard | Autonomous scratchpad, semantic delimiters. |
| `AR-CarryFirst-Sem` | AR-CarryFirst | `semantic_nope` | Standard | Carry-first advantage in autonomous traces. |
| `SM-MH-Semantic` | SM-MultiHead | `semantic_nope` | Standard | Linear heads with semantic headers, NoPE. |
| `SM-MH-Generic` | SM-MultiHead | `generic_nope` | Standard | Linear heads with a single generic delimiter. |
| `SM-MH-RoleNoPE` | SM-MultiHead | `role_embed_nope` | Standard | Linear heads with learned roles, no coordinates. |
| `SM-Seq-RoleNoPE` | SM-Seq | `role_embed_nope` | Standard | Sum-first tied head with role subspace. |
| `SM-SeqRev-RoleNoPE` | SM-SeqRev | `role_embed_nope` | Standard | Carry-first tied head with role subspace. |
| `SM-MH-SemOrigin` | SM-MultiHead | `semantic_origin_nope` | Standard | Linear heads with origin anchors. |
| `SM-Seq-SemOrigin` | SM-Seq | `semantic_origin_nope` | Standard | Sum-first counting from origin anchors. |
| `SM-SeqRev-SemOrigin` | SM-SeqRev | `semantic_origin_nope` | Standard | Carry-first counting from origin anchors. |
| `SM-Fork-SemOrigin` | SM-Fork | `semantic_origin_nope` | Standard | Order-free trigger queries on anchored prompts. |
| `SM-Seq-Addressed` | SM-Seq | `addressed_kv` | Standard | Sum-first with explicit numeral addresses. |
| `SM-SeqRev-Addressed` | SM-SeqRev | `addressed_kv` | Standard | Carry-first with explicit numeral addresses. |
| `SM-Fork-Addressed` | SM-Fork | `addressed_kv` | Standard | Order-free queries with explicit addresses. |
| `SM-Seq-ZeroPreamble` | SM-Seq | `zero_preamble_kv` | FullSpan | Sum-first, alphabet inferred (no axiom). |
| `SM-SeqRev-ZeroPreamble` | SM-SeqRev | `zero_preamble_kv` | FullSpan | Carry-first, alphabet inferred. |
| `SM-Fork-ZeroPreamble` | SM-Fork | `zero_preamble_kv` | FullSpan | Order-free queries, alphabet inferred. |

### 3.1 Prior-dial sweep
The $p_{\text{prior}}$ sweep runs on three configurations rather than all 19: `SM-Seq-Addressed`, `SM-SeqRev-Addressed` and `AR-CarryFirst-Sem`.

---

## 4. Secondary Levers

### 4.1 Depth ($N_{\text{layers}}$)
One column transition decomposes into: (0) structural indexing, i.e. finding column $k$ and each token's offset (free in T1, computed by attention under NoPE); (1) column alignment, retrieving $a_k$, $b_k$, $c_k$ or detecting absence; (2) de-symbolisation, mapping symbols to values via the axiom; (3) base determination, $B = |\mathcal{A}|$; (4) arithmetic in the FFN; (5) re-symbolisation, copying output bytes from the axiom; (6) halting logic. A plausible (unproven) minimum is 3 layers with coordinates supplied, plus at least one more under NoPE. Default is 6; 2 and 3 are compression tests.

### 4.2 FFN non-linearity
**SwiGLU** (default) $(\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}})W_{\text{down}}$: multiplicative gating suits the carry threshold and conditional subtraction. **GELU** MLP is the parameter-matched baseline ($d_{\text{ffn}}$ 1376 vs. 2048).

### 4.3 Attention mask
**Strict causal** (default, all tiers). **Prefix-LM** (bidirectional over the static prompt prefix, causal after): T1 only. It is invalid under NoPE, where a bidirectional prefix becomes an unordered bag and the axiom's order is lost.

### 4.4 Training bases
Bimodal $\{2, 10\}$, multimodal $\{2, 4, 8, 10\}$, single-base $\{10\}$ ([`03`](03_representations.md) §4.3).

---

## 5. Negative Controls and Ablations

| ID | Type | Tests | Baseline | Intervention | Expected / diagnostic |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NC-1** | Control | Need for per-instance random axioms | Random axioms from $\Sigma$ | Fixed canonical symbols in training (`-NC1Canon`) | ~100% on canonical, ~0% on random axioms |
| **NC-2** | Control | Need for multiple bases | $\{2,4,8,10\}$ | $\{10\}$ only | Base-3 / Base-12 EM; recitation of base-10 arithmetic |
| **NC-3** | Control | Causal dependence on the axiom | $(\mathcal{A}, a, b)$ | Swap $s_0 \leftrightarrow s_1$ in $\mathcal{A}$, keep operand bytes | Model should output the new correct sum $OUT'$, not $OUT$ |
| **NC-4** | Control | T1 reliance on coordinates | Full 512-d input | Zero the 256-d positional subspace at test time | In-domain EM should collapse |
| **ABL-1** | Ablation | Circuit depth | $N = 6$ | $N \in \{2, 3\}$ | Minimum $N$ reaching 99% in-domain and OOD |
| **ABL-2** | Ablation | FFN engine | SwiGLU | GELU | Steps to convergence; OOD base transfer |
| **ABL-3** | Ablation | Prefix visibility | Causal | Prefix-LM (T1 only) | Length-extrapolation EM |
| **ABL-4** | Ablation | Readout channel | `SM-MH` heads | Tied copy head (`SM-Seq`) | EM, parameters, extrapolation |

---

## 6. Confounds

* **Binary-switch**: with two training bases, the model can classify the axiom as "short" or "long" and switch between two memorised routines, never computing $B$. It then fails on unseen bases by reciting a training base's arithmetic. Multimodal training raises the cost of such switching; the recitation-failure criterion ([`01`](01_research_goal.md) §5.2) checks the *form* of errors.
* **Positional OOD**: failure at longer lengths caused by positional inputs never seen in training, not by the algorithm. Learned position tables overflow; sinusoidal coordinates in T1 shift into untrained phase ranges at $L = 21..40$ and may saturate the FFN. NoPE layouts avoid this.
* **Untrained embedding**: a byte never seen in training has untrained input and output weights, so failures on it say nothing about binding. Avoided by sampling from all of $\Sigma$.
* **Delimiter ambiguity**: in T3 one delimiter marks every boundary, so miscounting delimiters means confusing axiom, operands and output. Contrast with T2's distinct headers.
* **Probe scaffolding**: training on probe queries ($p_{\text{probe}} > 0$) may itself help binding; see [`05`](05_evaluation_and_metrics.md) §4.1.

---

## 7. Hypotheses

* **H1 (positional phase drift)**: `sinusoidal_pe` models reach $> 99\%$ in-domain at $L \le 20$ but $\approx 0\%$ at $L = 21..40$; unseen sinusoidal phases break column alignment.
* **H2 (linear heads collapse without coordinates)**: under NoPE, `C-Weight` models generalise poorly or fail, while `C-Sequence` models keep robust retrieval; one tail vector has no coordinate basis to split across several static heads.
* **H3 (carry-first advantage)**: Carry-First beats Sum-First on out-of-distribution generalisation, in both `SM-SeqRev` vs. `SM-Seq` and `AR-CarryFirst` vs. `AR-SumFirst`.
* **H4 (channel, not factorisation)**: `SM-MH` and `SM-Fork` share the same factorisation. Where `SM-MH` fails (e.g. `role_embed_nope`, `semantic_origin_nope`) and `SM-Fork` succeeds, the failure is due to the readout channel.
* **H5 (counting vs. addressing)**: delimiter counting (`semantic_origin_nope`) needs long training and shows delayed convergence (grokking); explicit addressing (`addressed_kv`) converges quickly and extrapolates to $L = 40$ by exact matching.
* **H6 (zero-preamble induction)**: with `D-FullSpan`, a model can infer $\mathcal{A}$ from operand addresses alone (`zero_preamble_kv`).
* **H7 (prior override curve)**: as $p_{\text{prior}}$ rises from 0 to 0.9, accuracy on `canonical_permuted` falls along a monotone sigmoid. Among failures, prior intrusion exceeds the chance rate $1/(B-1)$.
* **H8 (binding vs. execution)**: probe accuracy exceeds per-step and per-digit accuracy on the same instances, so out-of-distribution failures come from execution (alignment, carry propagation), not from failing to read the axiom.
