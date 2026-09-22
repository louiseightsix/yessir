# Decoupled Experimental Taxonomy: An Orthogonal Formalism for Algorithmic Arithmetic Transformers

**Specification Document 08**  
**TransformAdder Project**  
**Author**: Architectural Redesign Proposal (Revised)  
**Date**: September 21, 2026  
**Status**: Proposal / RFC

---

## 1. Executive Summary & Problem Statement

Throughout the first three experimental campaigns (Rounds 1–3) of the TransformAdder project, model configurations were designated by two compound labels:
1. **Runtime Paradigm**: `SM-MH`, `SM-Seq`, `SM-SeqRev`, `SM-Fork`, `AR`.
2. **Inductive Bias Tier**: `Tier 1` through `Tier 7`.

While this nomenclature facilitated early exploration, empirical findings (the collapse of multi-head linear decoders under NoPE, the intra-column directional asymmetry of `SM-Seq` vs. `SM-SeqRev`, the delayed grokking of delimiter-anchored representations, and memory scaling under query expansion) revealed that these labels conflate multiple independent computational variables.

This specification refactors the design space into **five orthogonal axes** governed by an explicit **admissibility matrix**:
1. **Axis I: Supervisory Agency** (Harnessed State Machine vs. Autonomous Autoregression).
2. **Axis II: Intra-Step Causal Topology** (Factored/Order-Free vs. Causal Chain: Sum-First vs. Carry-First).
3. **Axis III: Decoding Channel** (Weight-Space Multi-Head vs. Sequence-Space Query Trigger).
4. **Axis IV: Representation & Inductive Bias** (Coordinates, Scoping Lexicon, and Preamble).
5. **Axis V: Problem Distribution & Curriculum** (Standard Length vs. Full-Span Alphabet Guarantee).

---

## 2. Deconstruction of Experimental Conflations

### 2.1 Conflation 1: The Binary Nature of Supervisory Agency
In earlier discussions, the runtime harness was described as having multiple "levels" based on whether role embeddings or coordinate markers were passed. In reality, **every State Machine configuration performs the exact same external register management**:
* Python maintains the carry state $c_k$ and the accumulated output sequence $OUT_k$.
* Python constructs a fresh prompt slice for each column transition $k$.
* Python evaluates halting criteria and terminates the loop.

Whether the model receives sinusoidal coordinates, role IDs, or discrete address tokens is strictly a property of the **Representation (Axis IV)**, not the harness. The true Agency axis is **strictly binary**:
* **`A-Harnessed`**: Python acts as the external state register; the model computes single column transitions $\Delta s_k$.
* **`A-Autonomous`**: Zero external intervention after step 0; the neural network maintains all state and control flow in its own autoregressive KV cache.

### 2.2 Conflation 2: Micro-AR vs. Macro-AR (Directional Asymmetry in Full AR)
`SM-Seq` and `SM-SeqRev` are simply micro-autoregressive models bounded by an external prompt reset. The discovery in Round 2 that carry-first emission (`SM-SeqRev`) enables monotonic thresholding followed by conditional subtraction:
$$\text{Pass 1: } c_{k+1} = \mathbf{1}_{a_k + b_k + c_k \ge B}$$
$$\text{Pass 2: } o_k = (a_k + b_k + c_k) - c_{k+1} \cdot B$$
applies identically to full autoregressive models (`AR`). Historically, autoregressive scratchpad addition has hardcoded sum-first emission (`[S] o_k [C] c_{k+1}`). Testing **`AR-CarryFirst`** (`[C] c_{k+1} [S] o_k`) is an immediate, high-leverage empirical consequence of this refactoring.

### 2.3 Conflation 3: `SM-MH` vs. `SM-Fork` (Same Factorisation, Distinct Readout)
Both `SM-MH` and `SM-Fork` enforce strict **conditional independence** between column outputs:
$$P(o_k, c_{k+1}, \text{FLAG}_k \mid s_k) = P(o_k \mid s_k) \cdot P(c_{k+1} \mid s_k) \cdot P(\text{FLAG}_k \mid s_k)$$
They compute the exact same mathematical model. They differ solely in the readout channel:
* **`C-Weight` (`SM-MH`)**: Projects 3 distinct linear heads ($W_{\text{sum}}, W_{\text{carry}}, W_{\text{flag}}$) from a single tail hidden vector $\mathbf{h}_{\text{tail}}$.
* **`C-Sequence` (`SM-Fork`)**: Uses a single tied copy head ($W_{\text{head}} = E^T$) evaluated across 3 distinct prompt query triggers (`[S]`, `[C]`, `[F]`).

### 2.4 Conflation 4: Curriculum Confounding in Zero-Preamble Addressing
Tier 7 is not merely an architectural ablation of the `[AXIOM]` preamble. In the procedural generator (`core/data.py`), Tier 7 forces operand lengths to satisfy $L \ge B$ so that all $B$ symbols appear in the operands, ensuring solvability. Consequently, Tier 7 models are trained on a different length distribution than Tiers 1–6 (e.g., $L \in [10, 20]$ for Base 10, omitting short lengths $1..9$). Isolating this requires a dedicated **Problem Distribution Axis (Axis V)**.

---

## 3. The 5 Orthogonal Experimental Axes

### Axis I: Supervisory Agency ($\mathcal{S}_{\text{agency}}$)
* **`A-Harnessed`**: External state machine. Python maintains $c_k$ and $OUT_k$, advancing column pointers and resetting prompts.
* **`A-Autonomous`**: Full autoregression. Model emits the complete trace autonomously into its KV cache until emitting `[FINISHED]`.

### Axis II: Intra-Step Causal Topology ($\mathcal{S}_{\text{topology}}$)
* **`T-Factored`**: Order-free conditional independence. $o_k$ and $c_{k+1}$ are computed independently conditioned on prior state $s_k$.
* **`T-SumFirst`**: Causal chain. Emits sum digit $o_k$ first; carry computation conditions on the emitted sum digit.
* **`T-CarryFirst`**: Causal chain. Emits carry digit $c_{k+1}$ first; sum computation conditions on the emitted carry digit.

### Axis III: Decoding Readout Channel ($\mathcal{S}_{\text{channel}}$)
* **`C-Weight`**: Multi-head projection ($W_{\text{sum}}, W_{\text{carry}}, W_{\text{flag}}$) from a shared hidden state vector $\mathbf{h}_{\text{tail}}$.
* **`C-Sequence`**: Parameter-free tied copy head ($W_{\text{head}} = E^T$) conditioned on explicit sequence triggers (`[S]`, `[C]`, `[F]`).

### Axis IV: Representation & Inductive Bias ($\mathcal{S}_{\text{repr}}$)

```
                            AXIS IV SUB-COMPONENTS
                            
  IV-A: Positional Encoding (Coordinate System)
  ├── R-Pos-Sinusoidal:   Continuous 2D split sinusoidal PE (Role + Significance).
  └── R-Pos-NoPE:         Zero positional coordinates; attention is permutation-invariant.

  IV-B: Scoping & Delimiter Lexicon (Actual Codebase Tokens)
  ├── R-Scope-RoleEmbed:      Generic delimiter (|) + 64d learned role embedding table.
  ├── R-Scope-Semantic:       Dedicated tokens ([AXIOM], [A], [B], [CARRY], [OUT]).
  ├── R-Scope-Generic:        Single generic delimiter (|) across all section boundaries.
  ├── R-Scope-SemanticOrigin: Semantic section headers + origin anchor token (*) at section starts.
  ├── R-Scope-Addressed:      Semantic headers + generic entry delimiter (|) + origin token (*) 
  │                           separating little-endian numeral address from value.
  └── R-Scope-ZeroPreamble:   Identical to R-Scope-Addressed, but [AXIOM] header is omitted.

  IV-C: Query Addressing Mode
  ├── R-Query-Implicit:       Prompt ends on query trigger; model must infer column k from history.
  └── R-Query-ExplicitKV:     Harness explicitly formats query with target numeral address: [S] (k).
```

### Axis V: Problem Distribution & Curriculum ($\mathcal{S}_{\text{dist}}$)
* **`D-Standard`**: Operand lengths uniformly sampled from $[L_{\min}, L_{\max}]$. Single-digit and short operands permitted ($L \ge 1$).
* **`D-FullSpan`**: Operand lengths constrained to $L \ge B$. Guarantees all $B$ alphabet symbols appear in operands $A$ and $B$, ensuring zero-preamble solvability.

---

## 4. The Admissibility Matrix (Dead-Cell Elimination)

Not all Cartesian product combinations of these axes represent valid or physically realisable models. The table below formalises the **Feasibility Admissibility Rules**:

| Combination | Admissibility | Physical / Mathematical Rationale |
| :--- | :---: | :--- |
| `C-Weight` $\times$ `T-SumFirst` | **INVALID** | A multi-head linear layer projects multiple outputs from a single forward pass; it cannot causally chain one head's prediction into another head's input. |
| `C-Weight` $\times$ `T-CarryFirst`| **INVALID** | Same as above: causal chaining requires sequential token emission. |
| `C-Weight` $\times$ `A-Autonomous`| **INVALID** | Autoregressive generation emits a 1D token stream; multi-head parallel emission is incompatible with standard sequential trace generation. |
| `T-Factored` $\times$ `A-Autonomous`| **INVALID** | Standard autoregressive trace generation is fundamentally a causal chain: $x_t \sim P(x_t \mid x_{<t})$. |
| `R-Pos-Sinusoidal` $\times$ `R-Scope-Semantic` | **INADMISSIBLE** | Tier 1 2D sinusoidal PE requires the role embedding table to provide the role coordinate. |
| `R-Query-ExplicitKV` without `R-Scope-Addressed` | **INADMISSIBLE** | Explicit numeral address queries require numeral address formatting in the prompt. |

### Admissible Agency $\times$ Topology $\times$ Channel Subspace

Of the 12 theoretical combinations of Agency $\times$ Topology $\times$ Channel, exactly **four** are structurally sound:

| Configuration Tuple | Canonical Identifier | Operational Description |
| :--- | :--- | :--- |
| `⟨A-Harnessed, T-Factored, C-Weight⟩` | **`SM-MultiHead`** | Single forward pass per column; 3 linear projection heads. |
| `⟨A-Harnessed, T-Factored, C-Sequence⟩` | **`SM-Fork`** | 3 parallel micro-queries (`[S]`, `[C]`, `[F]`) evaluated via tied copy head. |
| `⟨A-Harnessed, T-SumFirst, C-Sequence⟩` | **`SM-Seq`** | Micro-AR: Emits $o_k$ at `[S]`, then $c_{k+1}$ at `[C]`, then `FLAG` at `[F]`. |
| `⟨A-Harnessed, T-CarryFirst, C-Sequence⟩` | **`SM-SeqRev`** | Micro-AR: Emits $c_{k+1}$ at `[C]`, then $o_k$ at `[S]`, then `FLAG` at `[F]`. |
| `⟨A-Autonomous, T-SumFirst, C-Sequence⟩` | **`AR-SumFirst`** | Full autonomous trace: `([S] o_k [C] c_{k+1})* [FINISHED]`. |
| `⟨A-Autonomous, T-CarryFirst, C-Sequence⟩`| **`AR-CarryFirst`** | Full autonomous trace: `([C] c_{k+1} [S] o_k)* [FINISHED]`. |

---

## 5. Master Taxonomy Mapping: Empirical Record & Active Fleet

The table below maps all historical and active configurations across Rounds 1–3 into the decoupled taxonomy. Empirical observations are strictly segregated from active hypotheses:

| Model Designator | Agency | Topology | Channel | Representation (IV-A / IV-B / IV-C) | Dist | Status | Empirical Observation / Active Telemetry |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **`SM-MH-T1`** | `A-Harnessed` | `T-Factored` | `C-Weight` | `Sinusoidal` / `RoleEmbed` / `Implicit` | `D-Std` | Evaluated | **Observed**: 100.0% In-Domain EM; collapses to 0.0% on double lengths ($L=21..40$) due to FFN phase drift. |
| **`SM-Seq-T1`** | `A-Harnessed` | `T-SumFirst` | `C-Sequence` | `Sinusoidal` / `RoleEmbed` / `Implicit` | `D-Std` | Evaluated | **Observed**: 100.0% In-Domain EM; collapses to 0.0% on double lengths (69.3% per-step acc). |
| **`AR-T2`** | `A-Autonomous` | `T-SumFirst` | `C-Sequence` | `NoPE` / `Semantic` / `Implicit` | `D-Std` | Evaluated | **Observed**: 100.0% In-Domain EM; 41.0% on Base-12; 0.0% on double lengths. |
| **`SM-MH-T2`** | `A-Harnessed` | `T-Factored` | `C-Weight` | `NoPE` / `Semantic` / `Implicit` | `D-Std` | Evaluated | **Observed**: 0.0% EM (54.0% per-step acc); delimiter subspace fragmentation under linear heads. |
| **`SM-MH-T3`** | `A-Harnessed` | `T-Factored` | `C-Weight` | `NoPE` / `Generic` / `Implicit` | `D-Std` | Evaluated | **Observed**: 51.8% In-Domain EM; plateaus due to directional ambiguity of single `\|`. |
| **`SM-MH-T4`** | `A-Harnessed` | `T-Factored` | `C-Weight` | `NoPE` / `RoleEmbed` / `Implicit` | `D-Std` | Evaluated | **Observed**: **Total collapse (0.0% EM, 1.8% step acc)**; linear heads fail without coordinates. |
| **`SM-Seq-T4`** | `A-Harnessed` | `T-SumFirst` | `C-Sequence` | `NoPE` / `RoleEmbed` / `Implicit` | `D-Std` | Evaluated | **Observed**: **56.0% Base-3 record**; tied copy + role subspaces optimal for interpolation. |
| **`SM-SeqRev-T4`**| `A-Harnessed` | `T-CarryFirst`| `C-Sequence` | `NoPE` / `RoleEmbed` / `Implicit` | `D-Std` | Evaluated | **Observed**: **2.4% full 40-digit EM**; carry-first monotonic threshold stabilizes rollouts. |
| **`SM-MH-T5`** | `A-Harnessed` | `T-Factored` | `C-Weight` | `NoPE` / `SemanticOrigin` / `Implicit` | `D-Std` | Evaluated | **Observed**: **Total collapse (0.0% EM, 1.2% step acc)**; reinforces weight-head failure. |
| **`SM-Seq-T5`** | `A-Harnessed` | `T-SumFirst` | `C-Sequence` | `NoPE` / `SemanticOrigin` / `Implicit` | `D-Std` | Evaluated | **Observed**: 51.0% EM at 100k; slow counter circuit convergence. |
| **`SM-SeqRev-T5`**| `A-Harnessed` | `T-CarryFirst`| `C-Sequence` | `NoPE` / `SemanticOrigin` / `Implicit` | `D-Std` | Evaluated | **Observed**: **Grokking at step 75k (46.2% EM, seed 5329)**; global counter emerges. |
| **`SM-Fork-T5`** | `A-Harnessed` | `T-Factored` | `C-Sequence` | `NoPE` / `SemanticOrigin` / `Implicit` | `D-Std` | Unrun | **Hypothesised**: Order-free factorisation without multi-head linear collapse. |
| **`SM-Seq-T6`** | `A-Harnessed` | `T-SumFirst` | `C-Sequence` | `NoPE` / `Addressed` / `ExplicitKV` | `D-Std` | Training | **Active Telemetry**: Step 1,700, Loss ~0.91 on L4 GPU. |
| **`SM-SeqRev-T6`**| `A-Harnessed` | `T-CarryFirst`| `C-Sequence` | `NoPE` / `Addressed` / `ExplicitKV` | `D-Std` | Training | **Active Telemetry**: Step 1,750, Loss ~0.90 on L4 GPU. |
| **`SM-Fork-T6`** | `A-Harnessed` | `T-Factored` | `C-Sequence` | `NoPE` / `Addressed` / `ExplicitKV` | `D-Std` | Training | **Active Telemetry**: Step 50, Loss 3.48; accum=8 stable on L4 GPU. |
| **`SM-Seq-T7`** | `A-Harnessed` | `T-SumFirst` | `C-Sequence` | `NoPE` / `ZeroPreamble` / `ExplicitKV`| `D-Full`| Training | **Active Telemetry**: Step 1,900, Loss ~0.89 on L4 GPU. |
| **`SM-SeqRev-T7`**| `A-Harnessed` | `T-CarryFirst`| `C-Sequence` | `NoPE` / `ZeroPreamble` / `ExplicitKV`| `D-Full`| Training | **Active Telemetry**: Step 1,700, Loss ~1.00 on L4 GPU. |
| **`SM-Fork-T7`** | `A-Harnessed` | `T-Factored` | `C-Sequence` | `NoPE` / `ZeroPreamble` / `ExplicitKV`| `D-Full`| Training | **Active Telemetry**: Step 50, Loss 3.53; accum=8 stable on L4 GPU. |

---

## 6. Immediate Experimental Priorities

### 6.1 The Autoregressive Ordering Frontier (`AR-SumFirst` vs. `AR-CarryFirst`)

In autonomous models (`A-Autonomous`), the ground-truth scratchpad generated by `core/reference.py` has historically emitted sum before carry:
```python
# Canonical Sum-First Trace (core/reference.py)
for s in steps:
    trace.extend([TOKEN_S, s["o"], TOKEN_C, s["c_out"]])
trace.append(FLAG_FINISHED)
```

The decoupled taxonomy highlights this as an arbitrary directional inductive bias. We define **`AR-CarryFirst`**:
```python
# Carry-First Trace
for s in steps:
    trace.extend([TOKEN_C, s["c_out"], TOKEN_S, s["o"]])
trace.append(FLAG_FINISHED)
```
This isolates the effect of intra-step causal topology in an unsupervised, autonomous setting. Implementing this requires minimal code modifications (in `core/reference.py` and `core/sequences.py:trace_coords`).

### 6.2 Testing Multi-Head Decoders under Explicit Symbolic Addressing

In Round 2, `C-Weight` (`SM-MH`) scored 0.0% EM across Tiers 4 and 5 because the single tail vector contained no coordinate to anchor distinct linear projections.

With **`R-Pos-NoPE` + `R-Scope-Addressed` + `R-Query-ExplicitKV`**, the prompt terminates on the explicit numeral address `(k)`. Does the presence of this discrete address token in the attention context provide sufficient conditioning for linear heads ($W_{\text{sum}}, W_{\text{carry}}, W_{\text{flag}}$) to succeed without continuous positional encodings? This represents a clean mechanistic test of whether the collapse of multi-head decoding was caused by coordinate-free representations or by the lack of local conditioning.

---

## 7. Direct Refactoring Strategy (Zero Legacy Debt)

Because this codebase is under sole development, we explicitly reject backward-compatibility layers, deprecated alias mappings, and legacy fallback logic. The codebase will be migrated directly to native orthogonal primitives:

1. **Direct CLI & Configurator Migration**:
   Replace `--paradigm` and `--tier` with direct orthogonal arguments:
   - `--agency`: `harnessed` | `autonomous`
   - `--topology`: `factored` | `sum_first` | `carry_first`
   - `--channel`: `weight` | `sequence`
   - `--representation`: `sinusoidal_pe` | `semantic_nope` | `generic_nope` | `role_embed_nope` | `semantic_origin_nope` | `addressed_kv` | `zero_preamble_kv`
   - `--curriculum`: `standard` | `full_span`
2. **First-Class Implementation of `AR-CarryFirst`**:
   Extend `reference_trace(steps, order="sum_first"|"carry_first")` in `core/reference.py` and parameterize AR data encoding and rollout.
3. **Clean Codebase Pruning**:
   Purge legacy conditionals like `if tier == ...` or `if paradigm == ...` in favor of declarative configuration dataclasses.
