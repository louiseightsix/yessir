# The 5-Axis Experimental Matrix, Admissibility, and Hypotheses

**Specification Document 04**  
**TransformAdder Project**

---

## 1. The 5-Axis Design Space Tensor

Every model configuration in the TransformAdder project is formally situated within a 5-dimensional design tensor:
$$\mathcal{M} = \langle \mathcal{S}_{\text{agency}}, \mathcal{S}_{\text{topology}}, \mathcal{S}_{\text{channel}}, \mathcal{S}_{\text{repr}}, \mathcal{S}_{\text{dist}} \rangle$$

where:
1. $\mathcal{S}_{\text{agency}} \in \{\texttt{A-Harnessed}, \texttt{A-Autonomous}\}$
2. $\mathcal{S}_{\text{topology}} \in \{\texttt{T-Factored}, \texttt{T-SumFirst}, \texttt{T-CarryFirst}\}$
3. $\mathcal{S}_{\text{channel}} \in \{\texttt{C-Weight}, \texttt{C-Sequence}\}$
4. $\mathcal{S}_{\text{repr}} \in \{\texttt{sinusoidal\_pe}, \texttt{semantic\_nope}, \texttt{generic\_nope}, \texttt{role\_embed\_nope}, \texttt{semantic\_origin\_nope}, \texttt{addressed\_kv}, \texttt{zero\_preamble\_kv}\}$
5. $\mathcal{S}_{\text{dist}} \in \{\texttt{D-Standard}, \texttt{D-FullSpan}\}$

---

## 2. The Admissibility Matrix (Dead-Cell Elimination)

The full Cartesian product yields $2 \times 3 \times 2 \times 7 \times 2 = 168$ theoretical cells. However, mathematical constraints, physical execution models, and structural prerequisites render the vast majority of these cells either impossible or vacuous.

The formal **Admissibility Rules** eliminate invalid cells:

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

Filtering by the admissibility rules reduces the $2 \times 3 \times 2$ mechanics subspace to exactly **six valid tuples**:

| Identifier | Agency | Topology | Channel | Operational Role |
| :--- | :---: | :---: | :---: | :--- |
| **`SM-MultiHead`** | `A-Harnessed` | `T-Factored` | `C-Weight` | Single pass per column; 3 parallel linear projection heads. |
| **`SM-Fork`** | `A-Harnessed` | `T-Factored` | `C-Sequence` | 3 parallel micro-queries (`[S]`, `[C]`, `[F]`) via tied copy head. |
| **`SM-Seq`** | `A-Harnessed` | `T-SumFirst` | `C-Sequence` | Micro-AR: Emits $o_k \to c_{k+1} \to \text{FLAG}_k$ sequentially. |
| **`SM-SeqRev`** | `A-Harnessed` | `T-CarryFirst`| `C-Sequence` | Micro-AR: Emits $c_{k+1} \to o_k \to \text{FLAG}_k$ sequentially. |
| **`AR-SumFirst`** | `A-Autonomous` | `T-SumFirst` | `C-Sequence` | Full autonomous trace: `([S] o_k [C] c_{k+1})* [FINISHED]`. |
| **`AR-CarryFirst`**| `A-Autonomous` | `T-CarryFirst`| `C-Sequence` | Full autonomous trace: `([C] c_{k+1} [S] o_k)* [FINISHED]`. |

---

## 3. Primary Canonical Experimental Model Matrix

The primary experimental suite crosses the 6 admissible mechanics tuples against the 7 canonical representation layouts:

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

---

## 4. Formal Theoretical Hypotheses

The experimental program is designed to confirm or falsify six central theoretical hypotheses:

### Hypothesis H1: Continuous Positional Phase Drift
* **Statement**: Models trained with continuous 2D sinusoidal positional enc (`sinusoidal_pe`) will achieve near-perfect in-domain accuracy ($> 99.0\%$) on training lengths ($L \le 20$), but will suffer **complete failure ($0.0\%$ EM)** when evaluated on double lengths ($L = 21..40$).
* **Mechanism**: Continuous PE introduces out-of-distribution frequency phases into the feed-forward layers at unseen positions, destroying the attention alignment between operand columns.

### Hypothesis H2: Static Linear Head Collapse in Coordinate-Free Spaces
* **Statement**: In coordinate-free representations (`R-Pos-NoPE`), static multi-head linear decoders (`C-Weight`) will fail to generalise or collapse entirely, whereas sequence-space tied copy decoders (`C-Sequence`) will maintain robust in-context retrieval.
* **Mechanism**: A single tail hidden vector $\mathbf{h}_{\text{tail}}$ lacks a continuous coordinate basis to partition its subspace across multiple static projection matrices ($W_{\text{sum}}, W_{\text{carry}}, W_{\text{flag}}$). Conversely, sequence query triggers (`[S]`, `[C]`, `[F]`) paired with tied embeddings allow dynamic in-context routing via attention.

### Hypothesis H3: Directional Asymmetry in Sequential Causal Decoders
* **Statement**: Across both harnessed micro-AR (`SM-SeqRev` vs. `SM-Seq`) and autonomous scratchpads (`AR-CarryFirst` vs. `AR-SumFirst`), **Carry-First causal topology will achieve strictly superior out-of-distribution generalisation over Sum-First**.
* **Mechanism**: Carry-First computes outgoing carry $c_{k+1}$ first via a monotonic threshold ($a_k + b_k + c_k \ge B$), which linear attention heads easily separate. Sum computation then conditions directly on $c_{k+1}$, simplifying to a conditional linear subtraction. Sum-First forces the network to compute non-linear modular residue with zero intermediate scratchpad tokens.

### Hypothesis H4: Factorisation Equivalence & Decoding Readout
* **Statement**: `SM-MultiHead` and `SM-Fork` implement the exact same factored conditional independence model:
  $$P(o_k, c_{k+1}, \text{FLAG}_k \mid s_k) = P(o_k \mid s_k) \cdot P(c_{k+1} \mid s_k) \cdot P(\text{FLAG}_k \mid s_k)$$
  In representation spaces where static linear heads collapse (`role_embed_nope`, `semantic_origin_nope`), `SM-Fork` will succeed, proving that the failure of `SM-MultiHead` is an artifact of the decoding channel rather than the factored topology.

### Hypothesis H5: Sample Complexity of Relative Counting vs. Key-Value Addressing
* **Statement**: Models relying on relative delimiter counting (`semantic_origin_nope`) will require extensive training and exhibit delayed convergence (grokking) to form global counting circuits. In contrast, discrete numeral key-value addressing (`addressed_kv`) will converge rapidly and achieve length-invariant extrapolation ($L=40$) via exact token matching.
* **Mechanism**: Delimiter counting requires self-attention to implement soft numerical integration across tokens in the residual stream. Key-Value addressing transforms column tracking into associative memory lookup.

### Hypothesis H6: Zero-Preamble In-Context Induction
* **Statement**: Provided the full-span alphabet curriculum ($L \ge B$) is satisfied, a transformer can learn to infer the base alphabet $\mathcal{A}$ zero-shot directly from operand column addresses without an explicit axiom preamble (`zero_preamble_kv`).
* **Mechanism**: The single-digit address markers ($0 \dots B-1$) function as an implicit dictionary distributed across the operand tokens.
