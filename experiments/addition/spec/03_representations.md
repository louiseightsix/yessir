# Representations: Alphabet Regimes, the Prior Dial, Layouts and Curricula

**Specification Document 03**

---

## 1. Overview

How a problem is written into tokens raises three separate questions:
1. **Binding**: how the axiom is stated (an explicit preamble, or no preamble with the alphabet inferred from addressed operands).
2. **Alignment**: how operand digits, carry and output history are located without coordinate drift (continuous positional encoding vs. delimiters vs. explicit numeral addresses).
3. **Prior control**: how the axiom relates to whatever symbol-order prior the model picked up in training.

This document defines the alphabet regimes and prior dial (§2), **Axis IV: Representation** ($\mathcal{S}_{\text{repr}}$, §3), and **Axis V: Curriculum** ($\mathcal{S}_{\text{dist}}$, §4).

---

## 2. Alphabet Regimes and the Prior Dial

If training uses only random byte alphabets, the model has no ordering prior to fight. If it uses only `0`..`9`, it memorises tables. To make *prior suppression* a controlled variable, alphabets are stratified into three regimes and training exposure is dialled with $p_{\text{prior}}$.

```
  1. symbols_random       B distinct bytes sampled uniformly from Sigma. Neutral baseline:
                          no consistent ordering prior.

  2. canonical_order      '0'..'9' then 'A','B' (truncated to B), in conventional order.
                          Axiom and any learned prior agree.

  3. canonical_permuted   The same familiar glyphs in a random order (e.g. '4','7','0','9',...).
                          Axiom and learned prior CONFLICT.
```

### 2.1 Regimes
1. **`symbols_random`**: every byte of $\Sigma$ appears in training, so these tokens are not novel; the regime measures binding where the model has no ordering prior over the symbols. This is the primary training and evaluation regime.
2. **`canonical_order`**: the positive control, where binding and prior reinforce each other. Also a parity check: under the core hypothesis, canonical and random axioms must score the same ([`01`](01_research_goal.md) §5.2).
3. **`canonical_permuted`**: the prior-suppression test. The axiom says `'7'` follows `'4'`, while canonical training (if $p_{\text{prior}} > 0$) says `'4' + '7'` gives sum `'1'` carry `'1'` in Base 10. The model must follow the axiom. Since models are trained from scratch, the only prior that exists is the one induced by training exposure to `canonical_order`.

### 2.2 Unconfounded prior dial ($p_{\text{prior}} \in [0, 1]$)
* **Sampling**: each training instance uses `canonical_order` with probability $p_{\text{prior}}$; otherwise its alphabet is sampled uniformly from the **full** pool $\Sigma$ (which includes `'0'`..`'9'`, `'A'`, `'B'`).
* **Why the full pool**: if the random fraction excluded digit glyphs, a $p_{\text{prior}} = 0$ model would never have seen them, and a `canonical_permuted` test would confound prior suppression with token novelty. Sampling from all of $\Sigma$ means that at $p_{\text{prior}} = 0$ the digit glyphs are familiar but carry no order preference; raising $p_{\text{prior}}$ strengthens only the order prior.
* **Composition nuance**: at $p_{\text{prior}} = 0$ the model sees each digit glyph as often as any byte, but rarely an alphabet made *entirely* of them. Optionally, the random fraction can include permuted canonical alphabets at a small rate to match composition.
* **Compliance curve**: sweeping $p_{\text{prior}} \in \{0, 0.25, 0.5, 0.75, 0.9\}$ and evaluating on `canonical_permuted` gives the prior-override compliance curve (H7, [`04`](04_experimental_matrix.md) §7).

*Legacy note*: the original negative control NC-1 (`--train-axiom canonical`, run suffix `-NC1Canon`) is the extreme $p_{\text{prior}} = 1$ case: training on canonical symbols only.

---

## 3. Axis IV: Representation Layouts ($\mathcal{S}_{\text{repr}}$)

```
  IV-A  Coordinates           How is order encoded?
        R-Pos-Sinusoidal      2D split positional encoding (learned role + sinusoidal significance)
        R-Pos-NoPE            No positional encoding; order only via causal attention

  IV-B  Scoping lexicon       How are section boundaries marked?
        R-Scope-None          No delimiters; juxtaposed digit streams
        R-Scope-Semantic      Dedicated headers [AXIOM] [A] [B] [CARRY] [OUT]
        R-Scope-Generic       One generic delimiter (|) for every boundary
        R-Scope-RoleEmbed     Generic delimiter + 64-d learned role embedding
        R-Scope-SemanticOrigin  Semantic headers + origin anchor (*) at each block start
        R-Scope-Addressed     Semantic headers + per-digit entries (| address * value)
        R-Scope-ZeroPreamble  As Addressed, with the [AXIOM] block omitted

  IV-C  Query addressing      How is the active column identified?
        R-Query-Implicit      Model infers k from |OUT_k|
        R-Query-ExplicitKV    Harness writes the target column address k into the query
```

### 3.1 The seven layouts (tiers)

| Tier | Layout | IV-A | IV-B | IV-C | Prompt prefix |
| :---: | :--- | :--- | :--- | :--- | :--- |
| T1 | `sinusoidal_pe` | Sinusoidal | None | Implicit | `[BOS] A a b` (+ 2D split PE) |
| T2 | `semantic_nope` | NoPE | Semantic | Implicit | `[BOS] [AXIOM] A [A] a [B] b` |
| T3 | `generic_nope` | NoPE | Generic | Implicit | `[BOS] A \| a \| b` |
| T4 | `role_embed_nope` | NoPE | RoleEmbed | Implicit | `[BOS] A \| a \| b` (+ 64-d role table) |
| T5 | `semantic_origin_nope` | NoPE | SemanticOrigin | Implicit | `[BOS] [AXIOM] * A [A] * a [B] * b` |
| T6 | `addressed_kv` | NoPE | Addressed | ExplicitKV | `[BOS] [AXIOM] A [A] (\| j * a_j)* [B] (\| j * b_j)*` |
| T7 | `zero_preamble_kv` | NoPE | ZeroPreamble | ExplicitKV | `[BOS] [A] (\| j * a_j)* [B] (\| j * b_j)*` |

In T6/T7 the address $j$ is itself written as a little-endian numeral in the active alphabet, so matching column $k$ is an exact content match rather than a count.

### 3.2 Tier 1 coordinates
The 512-d input is split into a 256-d symbol subspace (the one-hot byte identity, which the tied head reads back) and a 256-d positional subspace made of a 64-d learned **role** embedding (axiom, operand A, operand B, carry, output, flag, start, control) and a 192-d sinusoidal **significance** coordinate (the digit's offset within its block). Columns in different operands therefore share the same significance coordinate.

---

## 4. Axis V: Curriculum ($\mathcal{S}_{\text{dist}}$)

### 4.1 `D-Standard`
Operand lengths $L_a, L_b \sim \mathcal{U}(1, 20)$ independently; single-digit and unequal-length operands occur. Used by every layout with an axiom preamble.

### 4.2 `D-FullSpan`
Lengths are bounded below by the base: $L_{\text{lower}} = \max(1, B)$, $L_a, L_b \sim \mathcal{U}(L_{\text{lower}}, \max(20, L_{\text{lower}}))$.
* **Why**: under `zero_preamble_kv` the alphabet is never stated; the model must recover it from the column addresses $0, \dots, B-1$, which are written as single digits. If operands were shorter than $B$, some digits would never appear as addresses and the alphabet would be unrecoverable.
* **Note**: for $B \ge 20$ (or in extrapolation bands) this collapses to a fixed length $L = B$. Keeping it a separate curriculum stops length distribution being confounded with representation.

### 4.3 Training bases
* **Bimodal** $\{2, 10\}$: the minimal test of whether two bases are enough.
* **Multimodal** $\{2, 4, 8, 10\}$: reduces the Binary-Switch Confound ([`04`](04_experimental_matrix.md) §6) by making a per-base switch costlier than computing $B$ from the axiom. It cannot rule out an $n$-way switch; Base-3 and Base-12 evaluation is what discriminates.
* **Single-base** $\{10\}$: negative control NC-2.
