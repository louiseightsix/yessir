# Structural Representations, Coordinate Systems, and Problem Curricula

**Specification Document 03**  
**TransformAdder Project**

---

## 1. Overview: The Representational Design Space

While execution mechanics dictate the control flow and joint factorisation, the model's capacity to extrapolate depends critically upon its **internal representation of spatial structure and sequence order**.

Standard transformers rely on absolute positional encodings (sinusoidal or learned), which bind computation to fixed sequence indices and degrade catastrophically when inputs exceed the training length. This specification explores the spectrum from continuous geometric coordinates to discrete symbolic key-value addressing, formalised as:
* **Axis IV: Representation & Inductive Bias ($\mathcal{S}_{\text{repr}}$)**: Sub-dimensions covering coordinates, scoping lexicons, and query addressing.
* **Axis V: Problem Distribution & Curriculum ($\mathcal{S}_{\text{dist}}$)**: Length constraints ensuring mathematical solvability across alphabets.

---

## 2. Axis IV: Structural Representation Sub-Dimensions

```
                               AXIS IV: REPRESENTATION SUB-AXES
                               
  IV-A: Coordinate Mechanism (How is spatial order encoded?)
  ├── R-Pos-Sinusoidal:   Continuous 2D split sinusoidal PE (Role + Significance coordinates).
  └── R-Pos-NoPE:         Zero positional encodings; pure permutation-invariant self-attention.

  IV-B: Scoping & Delimiter Lexicon (How are boundaries demarcated?)
  ├── R-Scope-None:           Zero delimiters; raw juxtaposed digit streams (bare digits).
  ├── R-Scope-RoleEmbed:      Generic delimiter (|) + 64d learned role embedding table.
  ├── R-Scope-Semantic:       Dedicated tokens ([AXIOM], [A], [B], [CARRY], [OUT]).
  ├── R-Scope-Generic:        Single uniform delimiter (|) across all section boundaries.
  ├── R-Scope-SemanticOrigin: Semantic section headers + origin anchor token (*) at section starts.
  ├── R-Scope-Addressed:      Semantic headers + generic entry delimiter (|) + origin token (*)
  │                           separating little-endian numeral address from value.
  └── R-Scope-ZeroPreamble:   Identical to R-Scope-Addressed, but [AXIOM] dictionary is omitted.

  IV-C: Query Addressing Mode (How is the active column identified?)
  ├── R-Query-Implicit:       Prompt ends on bare trigger; model must infer column index k 
  │                           from historical output length |OUT_k| = k.
  └── R-Query-ExplicitKV:     Harness formats query with explicit target numeral address: (k).
```

### 2.1 Sub-Axis IV-A: Coordinate Mechanisms

#### `R-Pos-Sinusoidal` (2D Split Sinusoidal Positional Encoding)
Input embeddings $\mathbf{x}_i \in \mathbb{R}^{d_{\text{model}}}$ ($d_{\text{model}} = 256$) are constructed as the sum of a token embedding, a 64-dimensional learned role embedding, and a 192-dimensional continuous sinusoidal coordinate:
$$\mathbf{x}_i = E[\text{token}_i] + E_{\text{role}}[\text{role}_i] \oplus \text{PE}_{192}(\text{sig}_i)$$
* **Significance Coordinate ($\text{sig}_i$)**: Represents the column index $j \in [0, L)$ within the active section. Wavelengths form a geometric progression from $2\pi$ to $10{,}000 \cdot 2\pi$.
* **Theoretical Vulnerability**: Attention dot products $\mathbf{q}^T \mathbf{k}$ between sinusoidal features depend upon trigonometric phase differences. When sequence lengths extend to unseen regimes ($L > 20$), feed-forward layers encounter activations outside their training support, leading to phase drift and breakdown of column alignment.

#### `R-Pos-NoPE` (No Positional Encoding)
Positional coordinates and role embeddings are entirely omitted:
$$\mathbf{x}_i = E[\text{token}_i] \in \mathbb{R}^{d_{\text{model}}}$$
Self-attention is mathematically permutation-invariant over the input set. To compute sequential operations, the network must rely entirely upon structural delimiters, causal masking, or discrete symbolic tokens.

---

### 2.2 Sub-Axis IV-B: Scoping Lexicons & Delimiter Structures

The prompt sequence must inform the network where sections (axiom, operands, carry, output) begin and end. The codebase formalises six concrete lexical schemes:

1. **`R-Scope-None`**: Operands are directly concatenated with zero punctuation:
   $$[\texttt{BOS}] \circ \mathcal{A} \circ a \circ b$$
   Requires explicit continuous coordinate side-channels (`R-Pos-Sinusoidal`) to distinguish operands.
2. **`R-Scope-RoleEmbed`**: Section boundaries are marked by a single generic delimiter (`|`), complemented by a learned 64-dimensional role embedding (`ROLE_AXIOM`, `ROLE_OPERAND_A`, etc.):
   $$[\texttt{BOS}] \circ \mathcal{A} \circ [|] \circ a \circ [|] \circ b$$
3. **`R-Scope-Semantic`**: Each section is introduced by a distinct semantic token (`[AXIOM]`, `[A]`, `[B]`, `[CARRY]`, `[OUT]`):
   $$[\texttt{BOS}] \circ [\texttt{AXIOM}] \circ \mathcal{A} \circ [\texttt{A}] \circ a \circ [\texttt{B}] \circ b$$
4. **`R-Scope-Generic`**: A single uniform delimiter token (`|`) marks every boundary:
   $$[\texttt{BOS}] \circ \mathcal{A} \circ [|] \circ a \circ [|] \circ b \circ [|] \circ c_k \circ [|] \circ OUT_k$$
   Forces the network to disambiguate sections solely by ordinal counting of `|` tokens.
5. **`R-Scope-SemanticOrigin`**: Combines dedicated semantic section headers with an explicit origin anchor marker `*` (`DELIM_ORIGIN`) placed at the start of each section:
   $$[\texttt{BOS}] \circ [\texttt{AXIOM}] \circ [*] \circ \mathcal{A} \circ [\texttt{A}] \circ [*] \circ a \circ [\texttt{B}] \circ [*] \circ b$$
   Provides attention heads with an invariant reference anchor for relative distance measurement.
6. **`R-Scope-Addressed`**: Every digit entry is formatted as an explicit Key-Value tuple:
   $$[|] \circ \text{Addr}(j) \circ [*] \circ \text{Val}_j$$
   where $\text{Addr}(j)$ is the numeral representation of index $j$ encoded in the base alphabet $\mathcal{A}$. Column alignment becomes an associative memory lookup: "find the entry whose address matches query $k$".
7. **`R-Scope-ZeroPreamble`**: Identical to `R-Scope-Addressed`, but the entire `[AXIOM]` dictionary header is deleted from the sequence:
   $$[\texttt{BOS}] \circ [\texttt{A}] \circ \text{FormattedEntries}(a) \circ [\texttt{B}] \circ \text{FormattedEntries}(b)$$
   Forces the model to deduce the alphabet succession relation $\mathcal{A}$ in-context from operand entries.

---

### 2.3 Sub-Axis IV-C: Query Addressing Modes

In harnessed execution, the prompt must specify which column $k$ the model is expected to evaluate:

* **`R-Query-Implicit`**: The prompt terminates simply on the query trigger:
  $$\dots \circ c_k \circ OUT_k \circ [\texttt{S}]$$
  The model is not told the index $k$; it must infer $k$ by counting the number of historical output digits $|OUT_k| = k$.
* **`R-Query-ExplicitKV`**: The harness formats the query with an explicit symbolic numeral address matching the key-value schema:
  $$\dots \circ [|] \circ \text{Addr}(k) \circ [*] \circ [\texttt{S}]$$
  The attention layer can perform exact token-level key matching between the query address and operand entries.

---

## 3. The Seven Canonical Representation Layouts

The admissible intersections of sub-axes IV-A, IV-B, and IV-C define seven canonical representation layouts implemented in the codebase:

| Canonical Identifier | Coordinate (IV-A) | Scoping Lexicon (IV-B) | Query Mode (IV-C) | Prompt Prefix Grammar |
| :--- | :--- | :--- | :--- | :--- |
| **`sinusoidal_pe`** | `Sinusoidal` | `R-Scope-None` | `Implicit` | `[BOS] A a b` *(with 2D split sinusoidal PE)* |
| **`semantic_nope`** | `NoPE` | `R-Scope-Semantic` | `Implicit` | `[BOS] [AXIOM] A [A] a [B] b` |
| **`generic_nope`** | `NoPE` | `R-Scope-Generic` | `Implicit` | `[BOS] A \| a \| b` |
| **`role_embed_nope`**| `NoPE` | `R-Scope-RoleEmbed` | `Implicit` | `[BOS] A \| a \| b` *(with 64d learned role table)* |
| **`semantic_origin_nope`**| `NoPE` | `R-Scope-SemanticOrigin`| `Implicit` | `[BOS] [AXIOM] * A [A] * a [B] * b` |
| **`addressed_kv`** | `NoPE` | `R-Scope-Addressed` | `ExplicitKV` | `[BOS] [AXIOM] A [A] (\| j * a_j)* [B] (\| j * b_j)*` |
| **`zero_preamble_kv`**| `NoPE` | `R-Scope-ZeroPreamble` | `ExplicitKV` | `[BOS] [A] (\| j * a_j)* [B] (\| j * b_j)*` |

---

## 4. Axis V: Problem Distribution & Curriculum ($\mathcal{S}_{\text{dist}}$)

To isolate representational capabilities from training distribution artifacts, the data generation protocol formalises two distinct curricula:

### 4.1 `D-Standard` (Uniform Variable Length)
* **Sampling Distribution**: For every problem instance, operand lengths are sampled uniformly:
  $$L_a, L_b \sim \mathcal{U}(L_{\min}, L_{\max})$$
  with $L_{\min} = 1$ and $L_{\max} = 20$. Operands may consist of a single digit, and operand lengths may differ ($|a| \ne |b|$).
* **Scope**: Standard across all representation layouts with explicit axiom preambles.

### 4.2 `D-FullSpan` (Full-Span Alphabet Guarantee)
* **Sampling Distribution**: Operand lengths are lower-bounded by the radix $B$:
  $$L_{\text{lower}} = \max(L_{\min}, B)$$
  $$L_a, L_b \sim \mathcal{U}(L_{\text{lower}}, \max(L_{\max}, L_{\text{lower}}))$$
* **Theoretical Necessity**: Under zero-preamble induction (`zero_preamble_kv`), the model reads the alphabet $\mathcal{A}$ off the single-digit numeral addresses $0 \dots B-1$ that label operand columns. If an operand length were shorter than $B$, high-value digits would never appear in single-digit addresses, rendering the problem mathematically unsolvable in-context.
* **Empirical Consequence**: For bases $B \ge L_{\max}$ (e.g. Base 12 when $L_{\max}=20$, or when testing extrapolation), this distribution deterministically collapses to a fixed-length problem of exactly $L = B$ columns rather than a variable-length range. Acknowledging this curriculum shift on its own axis prevents confounding curriculum effects with representational advantages.
