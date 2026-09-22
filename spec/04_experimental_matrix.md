# The Experimental Design Tensor, Admissibility, and Hypotheses

**Specification Document 04**  
**TransformAdder Project: Foundational Framework**

---

## 1. The Multi-Axis Design Space Tensor

Every model configuration in the research framework is formally indexed within a 7-dimensional computational design tensor:
$$\mathcal{M} = \langle \mathcal{S}_{\text{family}}, \mathcal{S}_{\text{agency}}, \mathcal{S}_{\text{topology}}, \mathcal{S}_{\text{channel}}, \mathcal{S}_{\text{repr}}, \mathcal{S}_{\text{dist}}, p_{\text{prior}} \rangle$$

where:
1. **$\mathcal{S}_{\text{family}}$ (Rule Family)**: The class of procedural transformation rule. Reserved as an extensible axis; currently instantiated as:
   $$\mathcal{S}_{\text{family}} \in \{\texttt{addition}\} \quad (\text{future: } \texttt{cellular\_automata}, \texttt{grid\_translation}, \texttt{arc\_primitive})$$
2. **$\mathcal{S}_{\text{agency}}$ (Supervisory Agency)**: $\texttt{A-Harnessed} \mid \texttt{A-Autonomous}$.
3. **$\mathcal{S}_{\text{topology}}$ (Intra-Step Causal Topology)**: $\texttt{T-Factored} \mid \texttt{T-SumFirst} \mid \texttt{T-CarryFirst}$.
4. **$\mathcal{S}_{\text{channel}}$ (Readout Decoding Channel)**: $\texttt{C-Weight} \mid \texttt{C-Sequence}$.
5. **$\mathcal{S}_{\text{repr}}$ (Representation Layout)**: $\texttt{sinusoidal\_pe} \mid \texttt{semantic\_nope} \mid \texttt{generic\_nope} \mid \texttt{role\_embed\_nope} \mid \texttt{semantic\_origin\_nope} \mid \texttt{addressed\_kv} \mid \texttt{zero\_preamble\_kv}$.
6. **$\mathcal{S}_{\text{dist}}$ (Problem Curriculum)**: $\texttt{D-Standard} \mid \texttt{D-FullSpan}$.
7. **$p_{\text{prior}}$ (Training Prior Dial)**: $p_{\text{prior}} \in \{0.0, 0.25, 0.5, 0.75, 0.9\}$, specifying the training exposure to canonical numerical order.

---

## 2. The Admissibility Matrix (Dead-Cell Elimination)

Not all Cartesian product combinations of these axes represent valid or physically realisable models. The formal **Admissibility Rules** eliminate invalid cells:

| Constraint Rule | Admissibility | Formal Rationale |
| :--- | :---: | :--- |
| `C-Weight` $\times$ `T-SumFirst` | **INVALID** | A multi-head linear layer projects multiple outputs from a single forward pass; it cannot causally condition one head's prediction upon another head's output. |
| `C-Weight` $\times$ `T-CarryFirst`| **INVALID** | Same as above: causal chaining requires sequential token emission. |
| `C-Weight` $\times$ `A-Autonomous`| **INVALID** | Autoregressive generation emits a 1D token sequence; multi-head parallel emission is structurally incompatible with standard causal trace generation. |
| `T-Factored` $\times$ `A-Autonomous`| **OUT OF SCOPE** | Emitting parallel heads at each trace position under teacher forcing is theoretically definable, but lies outside standard causal autoregressive rollouts. |
| `sinusoidal_pe` with any `NoPE` layout | **INADMISSIBLE** | Continuous 2D PE requires coordinate side-channel injection incompatible with coordinate-free representations. |
| `addressed_kv` without `ExplicitKV` query | **INADMISSIBLE** | Discrete numeral address representations require explicit address queries in the harness prompt. |
| `zero_preamble_kv` $\times$ `D-Standard` | **INADMISSIBLE** | Under standard length distributions, short operands ($L < B$) fail to display all alphabet symbols, rendering zero-preamble induction mathematically unsolvable. |

### The Six Structurally Sound Mechanics Tuples

Filtering by the admissibility rules yields exactly **six valid mechanics configurations**:

| Identifier | Agency | Topology | Channel | Operational Role |
| :--- | :---: | :---: | :---: | :--- |
| **`SM-MultiHead`** | `A-Harnessed` | `T-Factored` | `C-Weight` | Single pass per step; 3 parallel linear projection heads. |
| **`SM-Fork`** | `A-Harnessed` | `T-Factored` | `C-Sequence` | 3 parallel micro-queries (`[S]`, `[C]`, `[F]`) via tied copy head. |
| **`SM-Seq`** | `A-Harnessed` | `T-SumFirst` | `C-Sequence` | Micro-AR: Emits $o_k \to c_{k+1} \to \text{FLAG}_k$ sequentially. |
| **`SM-SeqRev`** | `A-Harnessed` | `T-CarryFirst`| `C-Sequence` | Micro-AR: Emits $c_{k+1} \to o_k \to \text{FLAG}_k$ sequentially. |
| **`AR-SumFirst`** | `A-Autonomous` | `T-SumFirst` | `C-Sequence` | Full autonomous trace: `([S] o_k [C] c_{k+1})* [FINISHED]`. |
| **`AR-CarryFirst`**| `A-Autonomous` | `T-CarryFirst`| `C-Sequence` | Full autonomous trace: `([C] c_{k+1} [S] o_k)* [FINISHED]`. |

---

## 3. Primary Canonical Experimental Model Matrix

The primary experimental suite crosses the 6 admissible mechanics tuples against the 7 canonical representation layouts under the neutral prior baseline (**$p_{\text{prior}} = 0.0$**, matching the active fleet training regime):

| Model Matrix Identifier | Mechanics Tuple | Representation Layout | Curriculum | Scientific Objective |
| :--- | :--- | :--- | :---: | :--- |
| **`SM-MH-SinPE`** | `SM-MultiHead` | `sinusoidal_pe` | `D-Standard` | Baseline: 2D continuous coordinates under multi-head projections. |
| **`SM-Seq-SinPE`** | `SM-Seq` | `sinusoidal_pe` | `D-Standard` | Tests sequential token emission under continuous positional encoding. |
| **`AR-SumFirst-Sem`** | `AR-SumFirst` | `semantic_nope` | `D-Standard` | Canonical autonomous scratchpad with semantic delimiters under NoPE. |
| **`AR-CarryFirst-Sem`**| `AR-CarryFirst` | `semantic_nope` | `D-Standard` | Tests autonomous carry-first directional advantage in scratchpads. |
| **`SM-MH-Semantic`** | `SM-MultiHead` | `semantic_nope` | `D-Standard` | Tests multi-head linear decoders with dedicated semantic headers in NoPE. |
| **`SM-MH-Generic`** | `SM-MultiHead` | `generic_nope` | `D-Standard` | Tests linear decoders under uniform delimiter punctuation. |
| **`SM-MH-RoleNoPE`** | `SM-MultiHead` | `role_embed_nope` | `D-Standard` | Tests linear decoders with learned roles but zero positional coordinates. |
| **`SM-Seq-RoleNoPE`** | `SM-Seq` | `role_embed_nope` | `D-Standard` | Tests sum-first tied copy decoders with role subspaces in NoPE. |
| **`SM-SeqRev-RoleNoPE`**| `SM-SeqRev` | `role_embed_nope` | `D-Standard` | Tests carry-first tied copy decoders with role subspaces in NoPE. |
| **`SM-MH-SemOrigin`** | `SM-MultiHead` | `semantic_origin_nope` | `D-Standard` | Tests linear decoders with semantic headers + origin anchor in NoPE. |
| **`SM-Seq-SemOrigin`** | `SM-Seq` | `semantic_origin_nope` | `D-Standard` | Tests sum-first counter circuits with origin anchors in NoPE. |
| **`SM-SeqRev-SemOrigin`**| `SM-SeqRev` | `semantic_origin_nope`| `D-Standard` | Tests carry-first counter circuits with origin anchors in NoPE. |
| **`SM-Fork-SemOrigin`**| `SM-Fork` | `semantic_origin_nope` | `D-Standard` | Tests order-free sequence-space query branching on anchored prompts. |
| **`SM-Seq-Addressed`** | `SM-Seq` | `addressed_kv` | `D-Standard` | Tests sum-first micro-AR under discrete numeral key-value addressing. |
| **`SM-SeqRev-Addressed`**| `SM-SeqRev`| `addressed_kv` | `D-Standard` | Tests carry-first micro-AR under discrete numeral key-value addressing. |
| **`SM-Fork-Addressed`** | `SM-Fork` | `addressed_kv` | `D-Standard` | Tests order-free branching queries under discrete key-value addressing. |
| **`SM-Seq-ZeroPreamble`**| `SM-Seq` | `zero_preamble_kv` | `D-FullSpan` | Tests sum-first zero-preamble in-context alphabet deduction. |
| **`SM-SeqRev-ZeroPreamble`**|`SM-SeqRev`| `zero_preamble_kv`| `D-FullSpan` | Tests carry-first zero-preamble in-context alphabet deduction. |
| **`SM-Fork-ZeroPreamble`**| `SM-Fork` | `zero_preamble_kv` | `D-FullSpan` | Tests branching query zero-preamble in-context alphabet deduction. |

### 3.1 Designated Subset for the Prior Dial Sweep ($p_{\text{prior}}$)
Rather than multiplying all 19 configurations across the full prior sweep, the $p_{\text{prior}} \in \{0.0, 0.25, 0.5, 0.75, 0.9\}$ training sweep is evaluated on a focused subset of three leading architectures representing both supervisory regimes:
1. **`SM-Seq-Addressed`**: Harnessed Sum-First under discrete numeral addressing.
2. **`SM-SeqRev-Addressed`**: Harnessed Carry-First under discrete numeral addressing.
3. **`AR-CarryFirst-Sem`**: Autonomous Carry-First under semantic delimiters.

---

## 4. Formal Theoretical Hypotheses

The experimental program is designed to confirm or falsify eight central theoretical hypotheses:

### Hypothesis H1: Continuous Positional Phase Drift
* **Statement**: Models trained with continuous 2D sinusoidal positional encodings (`sinusoidal_pe`) will achieve near-perfect in-domain accuracy ($> 99.0\%$) on training lengths ($L \le 20$), but will suffer **complete failure ($0.0\%$ EM)** when evaluated on double lengths ($L = 21..40$).
* **Mechanism**: Continuous PE introduces out-of-distribution frequency phases into the feed-forward layers at unseen positions, destroying column alignment.

### Hypothesis H2: Static Linear Head Collapse in Coordinate-Free Spaces
* **Statement**: In coordinate-free representations (`R-Pos-NoPE`), static multi-head linear decoders (`C-Weight`) will fail to generalise or collapse entirely, whereas sequence-space tied copy decoders (`C-Sequence`) will maintain robust in-context retrieval.
* **Mechanism**: A single tail hidden vector $\mathbf{h}_{\text{tail}}$ lacks a continuous coordinate basis to partition its subspace across multiple static projection matrices ($W_{\text{sum}}, W_{\text{carry}}, W_{\text{flag}}$). Conversely, sequence query triggers (`[S]`, `[C]`, `[F]`) paired with tied embeddings enable dynamic in-context routing via attention.

### Hypothesis H3: Directional Asymmetry in Causal Chains
* **Statement**: Across both harnessed micro-AR (`SM-SeqRev` vs. `SM-Seq`) and autonomous scratchpads (`AR-CarryFirst` vs. `AR-SumFirst`), **Carry-First causal topology will achieve strictly superior out-of-distribution generalisation over Sum-First**.
* **Mechanism**: Carry-First computes outgoing carry $c_{k+1}$ first via a monotonic threshold ($a_k + b_k + c_k \ge B$), which linear attention heads easily separate. Sum computation then conditions directly on $c_{k+1}$, simplifying to a conditional linear subtraction.

### Hypothesis H4: Factorisation Equivalence & Decoding Readout
* **Statement**: `SM-MultiHead` and `SM-Fork` implement the exact same factored conditional independence model:
  $$P(o_k, c_{k+1}, \text{FLAG}_k \mid s_k) = P(o_k \mid s_k) \cdot P(c_{k+1} \mid s_k) \cdot P(\text{FLAG}_k \mid s_k)$$
  In representation spaces where static linear heads collapse (`role_embed_nope`, `semantic_origin_nope`), `SM-Fork` will succeed, proving that the failure of `SM-MultiHead` is an artifact of the decoding channel rather than the factored topology.

### Hypothesis H5: Sample Complexity of Relative Counting vs. Key-Value Addressing
* **Statement**: Models relying on relative delimiter counting (`semantic_origin_nope`) will require extensive training and exhibit delayed convergence (grokking) to form global counting circuits. In contrast, discrete numeral key-value addressing (`addressed_kv`) will converge rapidly and achieve length-invariant extrapolation ($L=40$) via exact token matching.

### Hypothesis H6: Zero-Preamble In-Context Induction
* **Statement**: Provided the full-span alphabet curriculum ($L \ge B$) is satisfied, a transformer can learn to infer the base alphabet $\mathcal{A}$ zero-shot directly from operand column addresses without an explicit axiom preamble (`zero_preamble_kv`).

### Hypothesis H7: Prior Suppression & The Rule-Prior Override Curve
* **Statement**: As the training prior strength $p_{\text{prior}}$ increases from $0.0$ to $0.9$, model accuracy on the `canonical_permuted` evaluation condition will degrade along a monotonic sigmoid curve. On failed instances under `canonical_permuted`, the **Prior Intrusion Rate** will significantly exceed the random guessing baseline of $\frac{1}{B-1}$, confirming that errors represent active prior regression rather than stochastic degeneration.

### Hypothesis H8: Decoupled Binding vs. Procedural Execution
* **Statement**: Single-token successor probe accuracy will significantly exceed per-step prediction accuracy ($\text{Acc}_{\text{step}}$) and per-digit output accuracy ($\text{Acc}_{\text{digit}}$) on the exact same problem instances. This proves that out-of-distribution arithmetic failure is driven by procedural calculation, spatial tracking, or carry propagation failure rather than an inability to bind or retrieve the active alphabet rule.
