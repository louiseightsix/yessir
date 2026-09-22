# Structural Representations, Prior Stratification, and Curricula

**Specification Document 03**  
**TransformAdder Project: Foundational Framework**

---

## 1. Overview: Representing Rules, Data, and Priors

To execute a parameterized rule $\mathcal{P}_{\Theta}$, the sequence representation must solve three distinct challenges:
1. **Binding Representation**: How the parameter binding $\Theta_{\text{rule}}$ is articulated in tokens (explicit dictionary vs. zero-preamble in-context deduction).
2. **Spatial Alignment**: How input elements, state registers, and output history are indexed without absolute coordinate drift (continuous PE vs. discrete key-value addressing).
3. **Prior Control**: How the bound alphabet interacts with the model's learned semantic priors (novel symbols vs. canonical orders vs. permuted canonical orders).

This specification formalises:
* **Axis IV: Representation & Inductive Bias ($\mathcal{S}_{\text{repr}}$)**.
* **Axis V: Problem Distribution & Curriculum ($\mathcal{S}_{\text{dist}}$)**.
* **Prior Parameterization ($\mathcal{S}_{\text{prior}}$)**: Controlled prior strength and alphabet stratification.

---

## 2. Controlled Prior Stratification & The Prior Dial

A central limitation of naive synthetic benchmarks is that random symbol sampling creates an uncontrolled, noisy prior. If training uses purely random byte alphabets, the model forms no consistent order prior; conversely, if training uses only standard ASCII digits (`0`..`9`), the model memorises static arithmetic tables.

To treat prior suppression as a rigorous experimental variable, we stratify alphabet conditions into three distinct regimes and parameterise training prior strength.

```
                           THE THREE ALPHABET REGIMES
                           
  1. symbols_novel         Arbitrary non-ASCII bytes from Sigma_novel (0x00..0xEF \ ASCII).
                           Model possesses ZERO pre-existing order prior.
                           
  2. canonical_order       Standard ASCII digit glyphs ('0'..'9') in standard numerical order.
                           Rule binding directly aligns with training frequency prior.
                           
  3. canonical_permuted    Standard ASCII digit glyphs ('0'..'9') with permuted successor order 
                           (e.g. '9' -> '0', '2' -> '9'...). 
                           DIRECT RULE-VERSUS-PRIOR CONFLICT.
```

### 2.1 The Alphabet Stratification Regimes
1. **`symbols_novel` (Pure In-Context Binding)**:
   - Symbols are sampled uniformly from the non-ASCII subset $\Sigma_{\text{novel}} = \Sigma \setminus \{\texttt{'0'}, \dots, \texttt{'9'}\}$.
   - Because the model has no pre-existing ordering bias over arbitrary bytes, this condition measures pure parameter binding and execution capacity in a neutral setting.
2. **`canonical_order` (Prior-Reinforced Baseline)**:
   - Symbols are the standard ASCII numerals in their standard mathematical order ($\mathcal{A} = [\texttt{'0'}, \texttt{'1'}, \dots]$).
   - Serves as the positive control where in-context rule binding and weight priors reinforce each other.
3. **`canonical_permuted` (The Rule-Prior Conflict Test)**:
   - The alphabet consists of the familiar ASCII digit glyphs $\{\texttt{'0'}, \dots, \texttt{'9'}\}$, but their successor order is randomly permuted (e.g. $\mathcal{A} = [\texttt{'4'}, \texttt{'7'}, \texttt{'0'}, \texttt{'9'}, \dots]$).
   - This represents the definitive test of **Prior Suppression**: the in-context definition states that $\texttt{'7'} + \texttt{'4'} = \dots$, while the model's pre-training or training frequency strongly associates $\texttt{'4'} + \texttt{'7'} = \texttt{'11'}$. The model must suppress its internal prior and strictly obey the prompt.

### 2.2 The Training Prior Dial ($p_{\text{prior}} \in [0.0, 1.0]$)
Prior strength is not merely an evaluation condition; it is a parameter of the training distribution:
* **$p_{\text{prior}}$**: The fraction of training instances generated under `canonical_order`. The remaining $1 - p_{\text{prior}}$ instances are generated under `symbols_novel`.
* **The Experimental Spectrum**:
  - $p_{\text{prior}} = 0.0$: Purely neutral training. Model acquires zero order prior over ASCII digits.
  - $p_{\text{prior}} = 0.5$: Balanced training. Model develops a strong default prior for ASCII numerals while simultaneously learning that in-context alphabets can vary.
  - $p_{\text{prior}} = 0.9$: Heavily biased training. Simulates large language models where standard digit arithmetic is deeply entrenched in weights.
* **The Compliance Curve**: Evaluating models across $p_{\text{prior}} \in \{0.0, 0.25, 0.5, 0.75, 0.9\}$ on `canonical_permuted` yields the **Prior-Override Compliance Curve**, mapping how rule adherence degrades as competing weight priors strengthen.

---

## 3. Axis IV: Representation Sub-Axes & Lexicons

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

### 3.1 The Seven Canonical Representation Layouts

The admissible intersections across sub-axes IV-A, IV-B, and IV-C define seven canonical representation layouts:

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

To isolate representational and rule-following capability from dataset artifacts, we define two curricula:

### 4.1 `D-Standard` (Uniform Variable Length)
* **Sampling Distribution**: For every instance, operand lengths are sampled uniformly:
  $$L_a, L_b \sim \mathcal{U}(L_{\min}, L_{\max})$$
  with $L_{\min} = 1$ and $L_{\max} = 20$. Operands may consist of a single digit, and operand lengths may differ ($|a| \ne |b|$).
* **Scope**: Applied to all representation layouts featuring an explicit rule parameter preamble.

### 4.2 `D-FullSpan` (Full-Span Alphabet Solvability Guarantee)
* **Sampling Distribution**: Operand lengths are lower-bounded by the radix $B$:
  $$L_{\text{lower}} = \max(L_{\min}, B)$$
  $$L_a, L_b \sim \mathcal{U}(L_{\text{lower}}, \max(L_{\max}, L_{\text{lower}}))$$
* **Theoretical Necessity**: Under zero-preamble induction (`zero_preamble_kv`), the rule parameters $\Theta_{\text{rule}} = \mathcal{A}$ are not explicitly declared; the model must deduce the alphabet sequence from the single-digit column addresses $0 \dots B-1$ tagging the operand entries. If operands were shorter than $B$, higher digits would never appear in single-digit addresses, rendering in-context rule deduction mathematically unsolvable.
* **Empirical Note**: For bases $B \ge L_{\max}$ (e.g. Base 12 when $L_{\max}=20$, or when testing extrapolation), this distribution deterministically collapses to a fixed-length problem of exactly $L = B$ columns rather than a variable-length range. Isolating this curriculum prevents confounding training length distribution with representational capacity.
