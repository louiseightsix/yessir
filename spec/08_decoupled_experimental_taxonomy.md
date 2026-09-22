# Decoupled Experimental Taxonomy: An Orthogonal 4-Axis Formalism for Algorithmic Arithmetic Transformers

**Specification Document 08**  
**TransformAdder Project**  
**Author**: Architectural Redesign Proposal  
**Date**: September 21, 2026  
**Status**: Proposal / RFC

---

## 1. Executive Summary & Problem Statement

Throughout the first three experimental campaigns (Rounds 1–3) of the TransformAdder project, models were categorised using two nominal designators:
1. **Runtime Paradigm**: `SM-MH`, `SM-Seq`, `SM-SeqRev`, `SM-Fork`, `AR`.
2. **Inductive Bias Tier**: `Tier 1` through `Tier 7`.

While this nomenclature enabled rapid empirical iteration, empirical discoveries (the collapse of multi-head decoders under NoPE, the intra-column directional asymmetry of `SM-Seq` vs. `SM-SeqRev`, the delayed grokking of dual-delimiter counting, and the memory scaling of branched queries) have revealed that these nominal categories **conflate multiple independent, orthogonal computational axes**:

* **Conflation 1 (Harness vs. Model)**: The label "State Machine" hides the fact that the external Python runtime performs critical memory management, column pointer tracking, and prompt re-formatting for the model. In contrast, "AR" demands total autonomous agency from the neural network.
* **Conflation 2 (Causal Dependency vs. Paradigm)**: `SM-Seq` and `SM-SeqRev` are not a distinct species of state machine; they are simply **Micro-Autoregression** bounded by an external column step. Crucially, standard macro-`AR` suffers from the exact same unaddressed directional ordering dilemma.
* **Conflation 3 (Factorisation vs. Head Architecture)**: `SM-MH` and `SM-Fork` implement the **exact same order-free causal factorisation**; they differ only in whether decoding occurs in *weight space* (multiple linear heads) or *sequence space* (multiple prompt query triggers with tied copy projection).
* **Conflation 4 (Positional Bias vs. Semantic Metadata vs. Lexical Dictionary)**: "Tiers" bundled continuous coordinates (PE), discrete delimiter counting, learned subspace role embeddings, explicit numeral coordinate tuples, and the presence or absence of an axiom dictionary into a single linear scale.

This document formally resolves these conflations by replacing the monolithic "Tiers" and "Paradigms" with an **orthogonal 4-dimensional tensor formalism**:
$$\mathcal{M} = \langle \text{Agency}, \text{Topology}, \text{Channel}, \text{Representation} \rangle$$

---

## 2. Deconstruction of Historical Conflations

### 2.1 Conflation 1: The Invisible Work of the Runtime Harness

In the original specification, `AR` and `SM-*` were treated as sibling choices. In reality, they represent radically different allocations of computational responsibility between the Python runtime and the neural network:

```
                            ALLOCATION OF RESPONSIBILITY
                            
   [ Level 3 / Tier 1 SM ] ──► Harness injects Coordinates + Roles + Carry + History
   [ Level 2 / Tier 4 SM ] ──► Harness injects Roles + Carry + History (NoPE)
   [ Level 1 / Tier 5 SM ] ──► Harness injects Carry + History (NoPE, pure tokens)
   [ Level 1 / Tier 6-7 SM] ─► Harness injects Query Coordinate + Carry + History
   [ Level 0 / Full AR ]   ──► Harness does NOTHING after step 0. Model does all.
```

In Level 3 (`SM-MH-T1`), the model does not need to learn arithmetic control flow; it is an algebraic transition function evaluated inside a rigid external harness. In Level 0 (`AR`), the model must simultaneously act as memory controller, instruction pointer, loop termination judge, and arithmetic ALU.

### 2.2 Conflation 2: Micro-AR vs. Macro-AR (The Order Dependency Blindspot)

The discovery of intra-column directional asymmetry in Round 2 proved that emitting sum before carry (`SM-Seq`) forces the model to solve modular arithmetic without intermediate scratchpad tokens, whereas emitting carry before sum (`SM-SeqRev`) permits monotonic thresholding followed by conditional subtraction:

$$\text{Sum-First: } o_k = (a_k + b_k + c_k) \pmod B \implies c_{k+1} = \mathbf{1}_{a_k + b_k + c_k \ge B}$$
$$\text{Carry-First: } c_{k+1} = \mathbf{1}_{a_k + b_k + c_k \ge B} \implies o_k = (a_k + b_k + c_k) - c_{k+1} \cdot B$$

Because `SM-Seq` is merely micro-autoregression, **full autoregressive models (`AR`) possess this exact same structural asymmetry**. Historically, autoregressive addition scratchpads have hardcoded a sum-first or carry-first layout without treating this as a primary scientific variable.

### 2.3 Conflation 3: `SM-MH` and `SM-Fork` are Decoupled Implementations of the Same Factorisation

Both `SM-MH` and `SM-Fork` enforce strict **conditional independence** between the column predictions:
$$P(o_k, c_{k+1}, \text{FLAG}_k \mid s_k) = P(o_k \mid s_k) \cdot P(c_{k+1} \mid s_k) \cdot P(\text{FLAG}_k \mid s_k)$$

Neither sum nor carry conditions on the other. They are the same mathematical model, differing solely in the decoder implementation:
* `SM-MH` multiplexes outputs across **weight projections** ($W_{\text{sum}}, W_{\text{carry}}, W_{\text{flag}}$) at a single token position $\mathbf{h}_{\text{tail}}$.
* `SM-Fork` multiplexes outputs across **prompt query keys** (`[S]`, `[C]`, `[F]`) evaluated by a single tied head ($W_{\text{head}} = E^T$).

---

## 3. The 4 Orthogonal Experimental Axes

To eliminate conflation and ensure all future ablations are mathematically well-defined, every experiment configuration is uniquely specified as a point in the product space $\mathcal{S}_{\text{agency}} \times \mathcal{S}_{\text{topology}} \times \mathcal{S}_{\text{channel}} \times \mathcal{S}_{\text{repr}}$.

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                       THE 4-AXIS DESIGN TENSOR                         │
   └────────────────────────────────────────────────────────────────────────┘
          │
          ├─► AXIS I:   SUPERVISORY HARNESS AGENCY  (Levels 0 to 3)
          │
          ├─► AXIS II:  INTRA-STEP CAUSAL TOPOLOGY  (Factored vs. Chain)
          │
          ├─► AXIS III: DECODING CHANNEL            (Weight vs. Sequence)
          │
          └─► AXIS IV:  REPRESENTATION & BIAS       (Coordinates, Delimiters, KV)
```

---

### Axis I: Supervisory Harness Agency ($\mathcal{S}_{\text{agency}}$)

This axis specifies the degree of external state management and metadata scaffolding supplied by the runtime harness at inference time.

| Level | Name | External State Management | Column Pointer Mechanism | Side-Channel Embeddings |
| :--- | :--- | :--- | :--- | :--- |
| **`A3`** | **Dual Oracle** | External register updates carry $c_k$ and outputs $OUT_k$; prompt reset per step. | Harness injects continuous spatial coordinate vector into query embedding. | **Role IDs + Significance Coordinates** injected into transformer layers. |
| **`A2`** | **Semantic Oracle** | External register updates carry $c_k$ and outputs $OUT_k$; prompt reset per step. | Model must deduce column $k$ by integrating over generic delimiters in output prefix. | **Role IDs only** (64d learned subspace). Positional coordinates zeroed (NoPE). |
| **`A1-Implicit`** | **State Machine (Counting)** | External register updates carry $c_k$ and outputs $OUT_k$; prompt reset per step. | Model must count delimiters or output tokens $|OUT_k| = k$ to deduce active column. | **None**. Pure 1D token IDs from universal byte vocabulary $\mathcal{V}$. |
| **`A1-Explicit`** | **State Machine (Addressed)** | External register updates carry $c_k$ and outputs $OUT_k$; prompt reset per step. | Harness explicitly writes target column coordinate in prompt tokens: `[S] (k)`. | **None**. Pure 1D token IDs from universal byte vocabulary $\mathcal{V}$. |
| **`A0`** | **Autonomous (AR)** | **None**. Model receives problem statement once; maintains state in internal KV cache. | Model generates own column markers or trace tokens sequentially. | **None**. Pure 1D token IDs from universal byte vocabulary $\mathcal{V}$. |

---

### Axis II: Intra-Step Causal Topology ($\mathcal{S}_{\text{topology}}$)

This axis specifies the joint distribution factorization between the sum digit $o_k$, the outgoing carry $c_{k+1}$, and the halting signal $\text{FLAG}_k$ for column $k$.

#### 1. Factored / Order-Free Topology (`T-Factored`)
$$P(o_k, c_{k+1}, \text{FLAG}_k \mid s_k) = P(o_k \mid s_k) \cdot P(c_{k+1} \mid s_k) \cdot P(\text{FLAG}_k \mid s_k)$$
* Neither sum nor carry is causally dependent on the other.
* Requires the transformer to compute both modular addition and threshold carry directly from input operands.

#### 2. Causal Chain: Sum-First (`T-SumFirst`)
$$P(o_k, c_{k+1}, \text{FLAG}_k \mid s_k) = P(o_k \mid s_k) \cdot P(c_{k+1} \mid s_k, o_k) \cdot P(\text{FLAG}_k \mid s_k, o_k, c_{k+1})$$
* Sum digit $o_k$ is emitted first. Carry computation conditions causally on the emitted sum digit.

#### 3. Causal Chain: Carry-First (`T-CarryFirst`)
$$P(o_k, c_{k+1}, \text{FLAG}_k \mid s_k) = P(c_{k+1} \mid s_k) \cdot P(o_k \mid s_k, c_{k+1}) \cdot P(\text{FLAG}_k \mid s_k, c_{k+1}, o_k)$$
* Carry digit $c_{k+1}$ is emitted first. Sum computation conditions causally on the emitted carry digit (conditional linear subtraction).

---

### Axis III: Decoding Channel ($\mathcal{S}_{\text{channel}}$)

This axis specifies how outputs are extracted from the latent representation of the transformer.

| Channel Mode | Description | Decoding Mechanism | Parameter Cost |
| :--- | :--- | :--- | :--- |
| **`C-Weight`** | **Multi-Head Projection** | Single sequence forward pass. Multiple distinct linear heads ($W_{\text{sum}}, W_{\text{carry}}, W_{\text{flag}}$) project from the tail latent vector $\mathbf{h}_{\text{tail}}$. | $3 \times (d_{\text{model}} \times |\mathcal{V}|)$ projection weights. |
| **`C-Sequence`** | **Tied Copy with Query Keys** | Multiple sequence passes or multiple sequence tokens. Outputs are projected via parameter-free tied embedding head $W_{\text{head}} = E^T$ conditioned on trigger query tokens (`[S]`, `[C]`, `[F]`). | **0 additional parameters** (tied to embedding matrix $E$). |

---

### Axis IV: Positional & Symbolic Representation ($\mathcal{S}_{\text{repr}}$)

This axis decouples how coordinates, scopes, and symbol vocabularies are encoded into tokens and embeddings.

```
                           AXIS IV DECOMPOSITION
                           
    Sub-Axis IV-A: Positional Scheme
    ├── R-Pos-Sinusoidal:   Continuous sinusoidal positional encodings (1D or 2D).
    ├── R-Pos-Delimiter:    Coordinate-free (NoPE). Positions inferred via punctuation counters.
    └── R-Pos-SymbolicKV:   Coordinate-free (NoPE). Positions bound explicitly as numeral tuples (v, k).

    Sub-Axis IV-B: Scope & Delimiter Lexicon
    ├── R-Scope-Learned:    64d learned role vectors injected per token.
    ├── R-Scope-Semantic:   Specialized structural delimiters ([AXIOM], [A], [B], [C], [OUT]).
    ├── R-Scope-Generic:    Uniform delimiter (|) across all field boundaries.
    └── R-Scope-Anchored:   Generic delimiter (|) + explicit origin anchor (*) at LSD.

    Sub-Axis IV-C: Lexical Preamble
    ├── R-Axiom-Explicit:   Prompt begins with full ordered axiom string [AXIOM] s_0 s_1 ... s_{B-1}.
    └── R-Axiom-Zero:       No axiom header. Operands guarantee full symbol span; model deduces order.
```

---

## 4. Mapping the Existing Fleet into the Decoupled Taxonomy

Every model variant trained across Rounds 1–3 maps into this 4-axis coordinate space:

| Historical Name | Axis I: Agency | Axis II: Topology | Axis III: Channel | Axis IV: Representation & Bias | Key Empirical Discovery |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **`SM-MH-T1`** | `A3` (Dual Oracle) | `T-Factored` | `C-Weight` | Sinusoidal 2D PE + Role Embeddings + Explicit Axiom | 100% In-Domain EM, but 0.0% Length Extrapolation (Phase Drift). |
| **`SM-Seq-T1`** | `A3` (Dual Oracle) | `T-SumFirst` | `C-Sequence` | Sinusoidal 2D PE + Role Embeddings + Explicit Axiom | High In-Domain stability, but phase drift on length extrapolation. |
| **`AR-T2`** | `A0` (Autonomous) | `T-SumFirst` | `C-Sequence` | NoPE + Semantic Delimiters (`[A]`, `[B]`) + Explicit Axiom | Strong base generalization (41% Base-12), 0% extrapolation. |
| **`SM-MH-T4`** | `A2` (Semantic) | `T-Factored` | `C-Weight` | NoPE + 64d Role Embeddings + Generic `\|` + Explicit Axiom | **Total Collapse (0.0% EM)**: Weight heads fail without coordinates. |
| **`SM-Seq-T4`** | `A2` (Semantic) | `T-SumFirst` | `C-Sequence` | NoPE + 64d Role Embeddings + Generic `\|` + Explicit Axiom | **56.0% Base-3 record**: Eliminating PE fixes coordinate distortion. |
| **`SM-SeqRev-T4`**| `A2` (Semantic) | `T-CarryFirst` | `C-Sequence` | NoPE + 64d Role Embeddings + Generic `\|` + Explicit Axiom | **First-ever 40-digit Extrap EM (2.4%)**: Monotonic carry enables extrapolation. |
| **`SM-Seq-T5`** | `A1-Implicit` | `T-SumFirst` | `C-Sequence` | Pure NoPE + Dual Delimiters (`\|`, `*`) + Explicit Axiom | 51.0% EM at 100k; slow counter convergence. |
| **`SM-SeqRev-T5`**| `A1-Implicit` | `T-CarryFirst` | `C-Sequence` | Pure NoPE + Dual Delimiters (`\|`, `*`) + Explicit Axiom | **Grokking Breakthrough (46.2% at 75k)**: Emergence of global counter. |
| **`SM-Fork-T5`** | `A1-Implicit` | `T-Factored` | `C-Sequence` | Pure NoPE + Dual Delimiters (`\|`, `*`) + Explicit Axiom | Order-free evaluation without multi-head weight collapse. |
| **`SM-Seq-T6`** | `A1-Explicit` | `T-SumFirst` | `C-Sequence` | Pure NoPE + Symbolic Numeral Addresses + Explicit Axiom | $O(1)$ address lookup; eliminates delimiter counting dilution. |
| **`SM-SeqRev-T6`**| `A1-Explicit` | `T-CarryFirst` | `C-Sequence` | Pure NoPE + Symbolic Numeral Addresses + Explicit Axiom | Carry-first micro-AR with exact discrete address matching. |
| **`SM-Fork-T6`** | `A1-Explicit` | `T-Factored` | `C-Sequence` | Pure NoPE + Symbolic Numeral Addresses + Explicit Axiom | Branching parallel queries with discrete address matching. |
| **`SM-Seq-T7`** | `A1-Explicit` | `T-SumFirst` | `C-Sequence` | Pure NoPE + Symbolic Numeral Addresses + **Zero Preamble** | In-context grammar deduction from full-span arithmetic constraints. |
| **`SM-SeqRev-T7`**| `A1-Explicit` | `T-CarryFirst` | `C-Sequence` | Pure NoPE + Symbolic Numeral Addresses + **Zero Preamble** | Zero-preamble in-context deduction with carry-first stability. |
| **`SM-Fork-T7`** | `A1-Explicit` | `T-Factored` | `C-Sequence` | Pure NoPE + Symbolic Numeral Addresses + **Zero Preamble** | Pure relational arithmetic reasoning under parallel query branching. |

---

## 5. Unexplored Frontiers Unlocked by the Decoupled Taxonomy

Breaking down the design space reveals multiple high-priority scientific cells that were previously obscured:

### 5.1 The Autoregressive Ordering Frontier: `AR-SumFirst` vs. `AR-CarryFirst`

In full autonomous autoregression (`A0`), models have historically been trained with sum-first scratchpads:
$$\text{Trace: } \dots \ [k] \ \mathbf{o_k} \ [C] \ \mathbf{c_{k+1}} \dots$$

The taxonomy predicts that standard autoregressive models suffer from the exact same modular bottleneck that hindered `SM-Seq`. By reformulating the autoregressive trace to be **Carry-First**:
$$\text{Trace: } \dots \ [k] \ \mathbf{c_{k+1}} \ [S] \ \mathbf{o_k} \dots$$
we can test whether full autoregressive scratchpads achieve the same massive length-extrapolation boost observed in `SM-SeqRev`.

### 5.2 Bridging the Agency Gap: The `A1-Explicit` Autoregressive Model

Can an autonomous model (`A0`) learn to emit its own explicit symbolic coordinates $(v, k)$ into its trace without an external state machine harness?
* An `AR` model operating in `R-Pos-SymbolicKV` representation would emit:
  $$\text{Trace: } \dots \ o_0 (0) \ c_1 (1) \ o_1 (1) \ c_2 (2) \dots$$
* This bridges the gap between state machine reliability and full autoregressive autonomy.

### 5.3 Weight-Space vs. Sequence-Space under Symbolic Addressing

Now that Tier 6 provides discrete address tokens $(k)$, does `C-Weight` (Multi-Head) still collapse?
* In Round 2, `SM-MH` collapsed under NoPE because the single tail token had no coordinate to anchor the three heads.
* Under `R-Pos-SymbolicKV`, if the prompt terminates on the explicit address token `(k)`, does the explicit address token provide sufficient conditioning for linear heads to succeed?

---

## 6. Implementation and Migration Strategy

### 6.1 Configuration Schema Refactoring

The interactive configurator (`scripts/configure_experiment.py`) and experiment runner (`transformadder/cli.py`) will transition from `--paradigm` and `--tier` flags to clean, orthogonal flags:

```bash
python -m transformadder.cli \
  --agency=harnessed_addressed \    # A0, A1_implicit, A1_addressed, A2, A3
  --topology=carry_first \          # factored, sum_first, carry_first
  --channel=sequence_query \        # weight_head, sequence_query
  --pos-scheme=symbolic_kv \        # sinusoidal, delimiter, symbolic_kv
  --scope-scheme=generic \          # learned_roles, semantic_tokens, generic, anchored
  --axiom-preamble=none \           # explicit, none
  --batch-size=256 \
  --grad-accum-steps=8
```

### 6.2 Backward Compatibility Alias Layer

To preserve compatibility with all existing experiment bundles in `experiments/`, an explicit alias mapper will translate historical names (`sm_seq`, `sm_fork`, `tier 6`) directly into their 4-axis tensor coordinates:

```python
LEGACY_VARIANT_MAP = {
    ("sm_seq", 6): {
        "agency": "A1_addressed",
        "topology": "sum_first",
        "channel": "sequence_query",
        "pos_scheme": "symbolic_kv",
        "scope_scheme": "semantic",
        "axiom_preamble": "explicit",
    },
    ("sm_fork", 7): {
        "agency": "A1_addressed",
        "topology": "factored",
        "channel": "sequence_query",
        "pos_scheme": "symbolic_kv",
        "scope_scheme": "semantic",
        "axiom_preamble": "none",
    },
}
```

---

## 7. Conclusion

By decomposing TransformAdder into its fundamental constituents—**Harness Agency**, **Causal Topology**, **Decoding Channel**, and **Representation Scheme**—we transform an ad-hoc collection of empirical tricks into a rigorous, predictive, and scalable framework for studying algorithmic reasoning in transformers.
