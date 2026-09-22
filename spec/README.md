# Zero-Shot Base-Conditioned Arithmetic: Decoupled Architectural Specification

This directory houses the formal, publication-grade specification for an investigation into **zero-shot algorithmic generalisation in transformers** trained from scratch on base-conditioned arithmetic.

The central research inquiry is whether a standard causal transformer can learn an abstract, base-invariant algorithmic circuit for addition—decoupling procedural column logic, successor identification, and carry propagation from the specific concrete tokens used to represent numerals—and transfer zero-shot to completely unseen numerical bases, unseen symbol alphabets, and extended sequence lengths.

This specification departs from historical compound abstractions ("tiers" and "paradigms") by formalising the experimental design space as an **orthogonal 5-axis computational tensor**:
$$\mathcal{M} = \langle \text{Agency}, \text{Topology}, \text{Channel}, \text{Representation}, \text{Curriculum} \rangle$$

---

## 1. Repository & Specification Conventions

1. **House Style**: British English (UK) is used throughout all specification documents, code docstrings, and comments (e.g. *generalisation*, *behaviour*, *artefact*, *factorisation*, *scaffolding*).
2. **Universal 1-Byte Vocabulary ($\mathcal{V}$)**:
   - Total vocabulary cardinality $|\mathcal{V}| = 256$ (`0x00` through `0xFF`), ensuring strict single-byte tokenisation with zero multi-byte or sub-word BPE tokenisation artifacts.
   - $\mathcal{V}_{\text{ctrl}} \subset \mathcal{V}$: Dedicated structural control tokens and section delimiters reserved in the upper byte range (`0xF0`–`0xFF`).
   - $\Sigma = \mathcal{V} \setminus \mathcal{V}_{\text{ctrl}}$: Open candidate pool of candidate numeral bytes (`0x00`–`0xEF`, 240 bytes) from which dynamic alphabets are sampled.
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
| `[BOS]` | `BOS` | `0xF0` | Beginning of sequence marker |
| `[START]` | `START` | `0xF1` | Autonomous trace initiation token |
| `[AXIOM]` | `DELIM_AXIOM` | `0xF2` | Axiom dictionary section header |
| `[A]` | `DELIM_A` | `0xF3` | Operand A section header |
| `[B]` | `DELIM_B` | `0xF4` | Operand B section header |
| `[CARRY]` | `DELIM_CARRY` | `0xF5` | Incoming carry section header |
| `[OUT]` | `DELIM_OUT` | `0xF6` | Accumulated sum history section header |
| `\|` | `DELIM_GENERIC` | `0xF7` | Generic section or entry delimiter |
| `*` | `DELIM_ORIGIN` | `0xF8` | Origin anchor (LSD) or key-value separator |
| `[S]` | `TOKEN_S` | `0xF9` | Sum query trigger / sum trace marker |
| `[C]` | `TOKEN_C` | `0xFA` | Carry query trigger / carry trace marker |
| `[F]` | `TOKEN_FLAG` | `0xFB` | Halting query trigger |
| `[CONTINUE]` | `FLAG_CONTINUE` | `0xFC` | Loop continuation flag |
| `[FINISHED]` | `FLAG_FINISHED` | `0xFD` | Loop completion flag |
| `[PAD]` | `PAD` | `0xFE` | Right-padding token for batched tensors |

*Note: In all sequence illustrations, numerals are rendered as human-readable glyphs (`0`, `1`, `2`...) for clarity; in physical instances, numerals are arbitrary bytes sampled from $\Sigma$.*

---

## 3. Specification Document Map

The specification suite is organised into modular, logically decoupled documents:

* **[`01_research_goal.md`](01_research_goal.md)**: Core research questions, theoretical motivations, formal operational invariants, and empirical falsification criteria.
* **[`02_mechanics.md`](02_mechanics.md)**: Execution mechanics: Supervisory Agency ($\mathcal{S}_{\text{agency}}$), Intra-Step Causal Topology ($\mathcal{S}_{\text{topology}}$), and Readout Decoding Channels ($\mathcal{S}_{\text{channel}}$).
* **[`03_representations.md`](03_representations.md)**: Positional and structural representation schemes ($\mathcal{S}_{\text{repr}}$), coordinate protocols, scoping lexicons, symbolic numeral addressing, and curricula ($\mathcal{S}_{\text{dist}}$).
* **[`04_experimental_matrix.md`](04_experimental_matrix.md)**: The 5-axis design tensor, the Admissibility Matrix, canonical models, and primary theoretical hypotheses.
* **[`05_evaluation_and_metrics.md`](05_evaluation_and_metrics.md)**: Out-of-distribution evaluation suite, test cells (radix shift, length extrapolation, alphabet permutation), and formal metric definitions.
* **[`06_implementation_blueprint.md`](06_implementation_blueprint.md)**: Engineering blueprint, deterministic data generation contracts, PyTorch architecture layouts, training protocols, and hardware efficiency guidelines.
