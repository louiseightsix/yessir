# Addition Experiment: Specification

> **Record, not maintained.** This is what the TransformAdder prototype tested, as of 2026-11-10. It is not the spec of the current experiment in `experiments/addition/`.

This directory is the specification for the **addition experiment**: can a transformer trained from scratch learn column addition in a way that is driven by an **in-context numeral system** (an ordered alphabet of digit symbols given in the prompt) rather than by symbols and bases memorised in its weights? And does it then transfer **zero-shot** to bases, alphabets and operand lengths it never saw in training?

Addition is a convenient rule to study because it is exact, multi-step, and has a single in-context parameter (the alphabet). The design space is described along six axes plus a prior dial (see [`04_experimental_matrix.md`](04_experimental_matrix.md)):
$$\mathcal{M} = \langle \mathcal{S}_{\text{agency}}, \mathcal{S}_{\text{topology}}, \mathcal{S}_{\text{channel}}, \mathcal{S}_{\text{repr}}, \mathcal{S}_{\text{dist}}, p_{\text{prior}} \rangle$$

---

## 1. Conventions

1. **House Style**: British English (UK) throughout specification documents, docstrings and comments (e.g. *generalisation*, *behaviour*, *artefact*, *factorisation*).
2. **Universal 1-Byte Vocabulary ($\mathcal{V}$)**:
   - $|\mathcal{V}| = 256$ (`0x00`–`0xFF`). Every numeral, delimiter and control token is exactly one byte token; there is no BPE or multi-character tokenisation.
   - $\mathcal{V}_{\text{ctrl}} \subset \mathcal{V}$: structural control tokens, query triggers and delimiters, reserved in `0xF0`–`0xFF` (Section 2).
   - $\Sigma = \mathcal{V} \setminus \mathcal{V}_{\text{ctrl}}$: the pool of 240 candidate numeral bytes (`0x00`–`0xEF`) from which alphabets are sampled.
3. **Notation**:
   - $B \in \mathbb{N}_{\ge 2}$: the base (radix). Training bases are $\{2, 10\}$ (bimodal) or $\{2, 4, 8, 10\}$ (multimodal); evaluation adds the unseen bases $\{3, 12\}$.
   - $\mathcal{A} = [s_0, s_1, \dots, s_{B-1}]$: the **axiom**, an ordered alphabet with $s_i \in \Sigma$ and $\text{val}(s_i) = i$. Where unambiguous, $\mathcal{A}$ also denotes the set of its symbols.
   - $a, b$: operands, non-empty sequences of symbols from $\mathcal{A}$, written least-significant digit first ($LSD \to MSD$). Most-significant zeros ($s_0$) are permitted.
   - $a_k, b_k$: operand digits at column $k$, with $a_k = s_0$ for $k \ge |a|$ (likewise $b_k$).
   - $c_k$: incoming carry to column $k$ ($c_0 = s_0$); $c_{k+1}$ is the outgoing carry. For two operands $c_k \in \{s_0, s_1\}$ always.
   - $o_k$: the sum digit of column $k$, $o_k \in \mathcal{A}$.
   - $K$: index of the final column. $K = \max(|a|, |b|) - 1$ if that column's carry out is $s_0$, otherwise $K = \max(|a|, |b|)$. A problem executes exactly $K + 1$ column steps.
   - $OUT = [o_0, \dots, o_K]$ ($LSD \to MSD$); $OUT_k = [o_0, \dots, o_{k-1}]$ is the prefix produced before column $k$ ($OUT_0 = []$).
   - $\text{FLAG}_k \in \{\texttt{FLAG\_CONTINUE}, \texttt{FLAG\_FINISHED}\}$; $\texttt{FLAG\_FINISHED}$ iff $k = K$.

---

## 2. Token Display Aliases & Byte Definitions

This table is the single source of truth for control-token bytes; `core/constants.py` mirrors it and `tests/test_mechanics.py` asserts that the two agree.

| Display Alias | Canonical Name | Byte Value | Description |
| :--- | :--- | :---: | :--- |
| `[BOS]` | `BOS` | `0xFE` | Beginning of sequence marker |
| `[START]` | `START` | `0xF5` | Autonomous trace initiation token |
| `[AXIOM]` | `DELIM_AXIOM` | `0xF0` | Axiom section header |
| `[A]` | `DELIM_A` | `0xF1` | Operand A section header |
| `[B]` | `DELIM_B` | `0xF2` | Operand B section header |
| `[CARRY]` | `DELIM_CARRY` | `0xF3` | Incoming carry section header |
| `[OUT]` | `DELIM_OUT` | `0xF4` | Accumulated output section header |
| `\|` | `DELIM_GENERIC` | `0xFB` | Generic entry delimiter |
| `*` | `DELIM_ORIGIN` | `0xFA` | Origin anchor (LSD) or key-value separator |
| `[S]` | `TOKEN_S` | `0xF6` | Sum digit query trigger (`o_k`) |
| `[C]` | `TOKEN_C` | `0xF7` | Carry query trigger (`c_{k+1}`) |
| `[F]` | `TOKEN_FLAG` | `0xFC` | Halting query trigger |
| `[PROBE]` | `TOKEN_PROBE` | `0xFF` | Single-token successor binding probe trigger |
| `[CONTINUE]` | `FLAG_CONTINUE` | `0xF8` | Loop continuation flag |
| `[FINISHED]` | `FLAG_FINISHED` | `0xF9` | Loop completion flag |
| `[PAD]` | `PAD` | `0xFD` | Right-padding token for batched tensors |

*In illustrations numerals are drawn as `0 1 2 …` for readability; in real instances they are arbitrary bytes from $\Sigma$.*

---

## 3. Run Naming: Paradigms and Tiers

The code and every run directory in `.prototypes/transformadder/results/` use the original **paradigm × tier** naming, e.g. `SM-SeqRev-T5-L6-SwiGLU-Causal-multimodal-seed5329`. It maps onto the axes as follows.

| Paradigm (`--paradigm`) | Name prefix | Mechanics tuple ([`04`](04_experimental_matrix.md) §2) |
| :--- | :--- | :--- |
| `sm_mh` | `SM-MH` | `SM-MultiHead` = ⟨Harnessed, Factored, Weight⟩ |
| `sm_fork` | `SM-Fork` | ⟨Harnessed, Factored, Sequence⟩ |
| `sm_seq` | `SM-Seq` | ⟨Harnessed, SumFirst, Sequence⟩ |
| `sm_seq_rev` | `SM-SeqRev` | ⟨Harnessed, CarryFirst, Sequence⟩ |
| `ar` (`--trace-order sum_first`) | `AR` | `AR-SumFirst` = ⟨Autonomous, SumFirst, Sequence⟩ |
| `ar` (`--trace-order carry_first`) | `AR` | `AR-CarryFirst` = ⟨Autonomous, CarryFirst, Sequence⟩ |

| Tier (`--tier`) | Representation layout ([`03`](03_representations.md) §3) | Curriculum |
| :---: | :--- | :--- |
| T1 | `sinusoidal_pe` (2D split PE: learned role + sinusoidal significance) | `D-Standard` |
| T2 | `semantic_nope` (role-specific delimiters, NoPE) | `D-Standard` |
| T3 | `generic_nope` (single generic delimiter, NoPE) | `D-Standard` |
| T4 | `role_embed_nope` (generic delimiter + learned role embedding) | `D-Standard` |
| T5 | `semantic_origin_nope` (semantic headers + origin anchor, NoPE) | `D-Standard` |
| T6 | `addressed_kv` (explicit little-endian numeral addresses) | `D-Standard` |
| T7 | `zero_preamble_kv` (addresses, no axiom preamble) | `D-FullSpan` |

Remaining name fields: `L<n>` = layers, `SwiGLU`/`GELU` = FFN, `Causal`/`PrefixLM` = attention mask, `bimodal`/`multimodal`/`base10` = training bases, optional `-NC1Canon` = fixed canonical-symbol training (negative control NC-1), `-Prior<p>`, `-Probe<p>`, then `seed<n>`.

`--paradigm` and `--tier` are authoritative. The CLI also has axis flags (`--agency`, `--topology`, `--channel`, `--representation`), but they predate this spec's naming and only partly match it; see [`06`](06_implementation.md) §7.

---

## 4. Document Map

* **[`01_research_goal.md`](01_research_goal.md)**: the hypothesis, the axiom, invariants, research questions, and success / falsification criteria.
* **[`02_mechanics.md`](02_mechanics.md)**: addition as a state machine, shared invariants, the halting contract, agency / topology / channel axes, sequence grammars and loss contracts.
* **[`03_representations.md`](03_representations.md)**: alphabet regimes and the prior dial $p_{\text{prior}}$, representation layouts (tiers), and curricula.
* **[`04_experimental_matrix.md`](04_experimental_matrix.md)**: the design space, admissibility rules, the model matrix, secondary levers, negative controls and ablations, confounds, and hypotheses H1–H8.
* **[`05_evaluation_and_metrics.md`](05_evaluation_and_metrics.md)**: zero-shot protocol, evaluation cells, metrics, the binding probe, the error taxonomy, and per-instance telemetry.
* **[`06_implementation.md`](06_implementation.md)**: code layout, model architecture, data generation, losses, training hyperparameters, and run artefacts.
