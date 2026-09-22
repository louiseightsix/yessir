# Neural Rule Execution & In-Context Induction: Decoupled Architectural Specification

This directory houses the formal, publication-grade specification for an investigation into **in-context rule following, dynamic parameter binding, and prior suppression in transformers**.

The central inquiry is whether a standard causal transformer can learn a generalisable, base-invariant algorithmic circuit for procedural execution—decoupling abstract computational flow, state transitions, and successor identification from the specific concrete tokens used to represent parameters—and transfer zero-shot to completely unseen environments, unseen symbol alphabets, and extended sequence lengths.

Rather than treating arithmetic as an end in itself, this specification formalises base-conditioned addition as the **first concrete instance of a generalized, multi-task rule-following harness designed to extend directly toward ARC-AGI**.

The experimental design space is formalised as an **orthogonal 7-dimensional computational design tensor**:
$$\mathcal{M} = \langle \mathcal{S}_{\text{family}}, \mathcal{S}_{\text{agency}}, \mathcal{S}_{\text{topology}}, \mathcal{S}_{\text{channel}}, \mathcal{S}_{\text{repr}}, \mathcal{S}_{\text{dist}}, p_{\text{prior}} \rangle$$

---

## 1. Repository & Specification Conventions

1. **House Style**: British English (UK) is used throughout all specification documents, code docstrings, and comments (e.g. *generalisation*, *behaviour*, *artefact*, *factorisation*, *scaffolding*).
2. **Universal 1-Byte Vocabulary ($\mathcal{V}$)**:
   - Total vocabulary cardinality $|\mathcal{V}| = 256$ (`0x00` through `0xFF`), ensuring strict single-byte tokenisation with zero multi-byte or sub-word BPE tokenisation artifacts.
   - $\mathcal{V}_{\text{ctrl}} \subset \mathcal{V}$: Dedicated structural control tokens, probe triggers, and section delimiters reserved in the upper byte range (`0xF0`–`0xFF`).
   - $\Sigma = \mathcal{V} \setminus \mathcal{V}_{\text{ctrl}}$: Open candidate pool of candidate numeral bytes (`0x00`–`0xEF`, 240 bytes) from which dynamic alphabets are sampled.
   - $\Sigma_{\text{novel}} = \Sigma \setminus \{\texttt{'0'}, \dots, \texttt{'9'}\}$: Subset of novel symbols with zero pre-existing ASCII order prior.
3. **Mathematical Notation**:
   - $B \in \mathbb{N}_{\ge 2}$: Numerical radix (base). In canonical training regimes, $B \in \{2, 4, 8, 10\}$; in out-of-distribution evaluation, $B \in \{3, 12\}$.
   - $\mathcal{A} = [s_0, s_1, \dots, s_{B-1}]$: Ordered numeral alphabet where $s_i \in \Sigma$ and $\text{val}(s_i) = i$.
   - $a, b$: Input operands, non-empty sequences of numeral symbols from $\mathcal{A}$ in least-significant-digit-first ($LSD \to MSD$) order.
   - $a_k, b_k$: Operand digits at column $k$ ($a_k = s_0$ for $k \ge |a|$, and likewise for $b_k$).
   - $c_k$: Incoming carry to column $k$ ($c_0 = s_0$); $c_{k+1}$ is the outgoing carry produced by column $k$. For two-operand addition, $c_k \in \{s_0, s_1\}$.
   - $o_k$: Computed sum digit for column $k$, where $o_k \in \mathcal{A}$.
   - $K$: Index of the final column ($K = \max(|a|, |b|) - 1$ if terminal carry out is $s_0$, else $\max(|a|, |b|)$). A problem executes exactly $K + 1$ column transitions.
   - $OUT_k = [o_0, \dots, o_{k-1}]$: Cumulative prefix of sum digits produced prior to column $k$ ($OUT_0 = []$, $|OUT_k| = k$).
   - $\text{FLAG}_k \in \{\texttt{FLAG\_CONTINUE}, \texttt{FLAG\_FINISHED}\}$: Loop halting signal emitted with column $k$; $\texttt{FLAG\_FINISHED}$ iff $k = K$.

---

## 2. Token Display Aliases & Byte Definitions

Sequence illustrations and formal grammar definitions use canonical single-byte token aliases:

| Display Alias | Canonical Name | Byte Value | Description |
| :--- | :--- | :---: | :--- |
| `[BOS]` | `BOS` | `0xFE` | Beginning of sequence marker |
| `[START]` | `START` | `0xF5` | Autonomous trace initiation token |
| `[AXIOM]` | `DELIM_AXIOM` | `0xF0` | Axiom / rule definition section header |
| `[A]` | `DELIM_A` | `0xF1` | Operand A / Input section header |
| `[B]` | `DELIM_B` | `0xF2` | Operand B section header |
| `[CARRY]` | `DELIM_CARRY` | `0xF3` | State register / Incoming carry section header |
| `[OUT]` | `DELIM_OUT` | `0xF4` | Accumulated history section header |
| `\|` | `DELIM_GENERIC` | `0xFB` | Generic entry delimiter |
| `*` | `DELIM_ORIGIN` | `0xFA` | Origin anchor (LSD) or key-value separator |
| `[S]` | `TOKEN_S` | `0xF6` | Primary output query trigger (`o_k`) |
| `[C]` | `TOKEN_C` | `0xF7` | State update query trigger (`c_{k+1}`) |
| `[F]` | `TOKEN_FLAG` | `0xFC` | Halting query trigger |
| `[PROBE]` | `TOKEN_PROBE` | `0xFF` | Single-token parameter binding probe trigger |
| `[CONTINUE]` | `FLAG_CONTINUE` | `0xF8` | Loop continuation flag |
| `[FINISHED]` | `FLAG_FINISHED` | `0xF9` | Loop completion flag |
| `[PAD]` | `PAD` | `0xFD` | Right-padding token for batched tensors |

*Note: In all sequence illustrations, numerals are rendered as human-readable glyphs (`0`, `1`, `2`...) for clarity; in physical instances, numerals are arbitrary bytes sampled from $\Sigma$.*

---

## 3. Specification Document Map

The specification suite is organised into modular, logically decoupled documents:

* **[`01_research_goal.md`](01_research_goal.md)**: Universal rule-following ontology (Rule, Binding, Prior, Execution), spectrum of complexity toward ARC-AGI, core research questions, and explicit boundary scope.
* **[`02_mechanics.md`](02_mechanics.md)**: Generic State Machine execution contracts ($\mathcal{S}, \mathcal{I}, \mathcal{O}, \delta, \mathcal{H}$), Supervisory Agency ($\mathcal{S}_{\text{agency}}$), Intra-Step Causal Topologies ($\mathcal{S}_{\text{topology}}$), and Readout Channels ($\mathcal{S}_{\text{channel}}$).
* **[`03_representations.md`](03_representations.md)**: Prior stratification (`symbols_novel`, `canonical_order`, `canonical_permuted`), training prior dial $p_{\text{prior}}$, structural representation layouts ($\mathcal{S}_{\text{repr}}$), and curricula ($\mathcal{S}_{\text{dist}}$).
* **[`04_experimental_matrix.md`](04_experimental_matrix.md)**: The 7-axis design tensor, reserved `rule_family` axis, Admissibility Matrix, canonical models, and formal hypotheses H1 through H8.
* **[`05_evaluation_and_metrics.md`](05_evaluation_and_metrics.md)**: Stratified evaluation cells, the In-Context Binding Probe ($s_i \to s_{i+1}$), 4-way error taxonomy (Prior Intrusion, Out-of-Alphabet, Procedural Error, Syntax Malformation), and per-instance JSONL regression telemetry.
* **[`06_implementation_blueprint.md`](06_implementation_blueprint.md)**: Extensible `Rule` protocol / ABC, Universal Transformer blueprint, stratified generator contracts, per-instance result writer, and backward-compatible checkpoint loader.
