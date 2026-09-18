# Zero-Shot Base-Conditioned Addition: Experiment Specification

This directory houses the formal, modular specification for an investigation into **zero-shot algorithmic generalisation in transformers** trained from scratch on base-conditioned arithmetic.

The central research question is whether an autoregressive model can decouple abstract addition logic (column addition, successor identification, carry propagation) from the concrete symbols used to represent numerals, transferring zero-shot to completely unseen numerical bases purely from an in-context axiom defining the numeral system.


## Repository & Spec Conventions

1. **House Style**: British English (UK) is used throughout all specification documents and code comments (e.g. *generalisation*, *behaviour*, *artefact*).
2. **Mathematical Notation & Vocabulary Space**:
   - $\mathcal{V}$: Universal 1-byte token vocabulary, $|\mathcal{V}| = 256$ (`0x00` through `0xFF`).
   - $\mathcal{V}_{\text{ctrl}}$: Dedicated single-byte control tokens and structural delimiters ($\mathcal{V}_{\text{ctrl}} \subset \mathcal{V}$, reserved in range `0xF0`–`0xFF`), preserving strict 1-byte = 1-token representation. The byte assignments are defined **once**, in [`03_levers.md`](03_levers.md) Section 2.1; every other document references that table.
   - $\Sigma$: Open candidate pool of byte numerals ($\Sigma = \mathcal{V} \setminus \mathcal{V}_{\text{ctrl}}$, `0x00`–`0xEF`, 240 bytes) from which distinct numeral symbols are sampled.
   - $B$: Numerical base cardinality ($2 \le B \le 240$). In minimal bimodal training, $B \in \{2, 10\}$; in evaluation, $B \in \{3, 12\}$.
   - $\mathcal{A}$: In-context axiom, the ordered sequence of distinct symbols $[s_0, s_1, \dots, s_{B-1}]$ where $s_i \in \Sigma$ and $\text{val}(s_i) = i$. Where no ambiguity arises, $\mathcal{A}$ also denotes the *set* of its symbols (as in $o_k \in \mathcal{A}$).
   - $a, b$: Input operands, non-empty sequences of symbols from $\mathcal{A}$ in least-significant-digit-first ($LSD \to MSD$) order. Most-significant zeros ($s_0$) are permitted; operands are not required to be canonical numerals.
   - $a_k, b_k$: Digit of the operand at column $k$, with $a_k = s_0$ for $k \ge |a|$ (likewise $b_k$).
   - $c_k$: Incoming carry to column $k$ (with $c_0 = s_0$); $c_{k+1}$ is the outgoing carry produced by column $k$. For two-operand addition, $c_k \in \{s_0, s_1\}$ always.
   - $o_k$: Computed sum digit for column $k$ ($o_k \in \mathcal{A}$).
   - $K$: Index of the final column. $K = \max(|a|, |b|) - 1$ if the carry out of that column is $s_0$, otherwise $K = \max(|a|, |b|)$. A problem therefore executes exactly $K + 1$ column steps.
   - $OUT$: Complete sum sequence $[o_0, o_1, \dots, o_K]$, in $LSD \to MSD$ order. $OUT_k = [o_0, \dots, o_{k-1}]$ is the prefix accumulated *before* column $k$ is processed ($OUT_0 = []$, $|OUT_k| = k$).
   - $\text{FLAG}_k \in \{\texttt{FLAG\_CONTINUE}, \texttt{FLAG\_FINISHED}\}$: Loop control signal emitted with column $k$; $\texttt{FLAG\_FINISHED}$ iff $k = K$.
   - **1-Byte Invariant**: Every numeral, structural delimiter, and loop control flag is strictly a single byte token; no BPE merging or multi-character string tokenisation is permitted.
3. **Token Display Aliases**: Sequence illustrations use short display names. Each alias denotes exactly one canonical token from [`03_levers.md`](03_levers.md) Section 2.1:

   | Alias | Canonical Token | Alias | Canonical Token |
   | :--- | :--- | :--- | :--- |
   | `[BOS]` | `BOS` | `[START]` | `START` |
   | `[AXIOM]` | `DELIM_AXIOM` | `[S]` | `TOKEN_S` |
   | `[A]` | `DELIM_A` | `[C]` | `TOKEN_C` |
   | `[B]` | `DELIM_B` | `[F]` | `TOKEN_FLAG` |
   | `[CARRY]` | `DELIM_CARRY` | `[FINISHED]` | `FLAG_FINISHED` |
   | `[OUT]` | `DELIM_OUT` | `[CONTINUE]` | `FLAG_CONTINUE` |
   | `\|` | `DELIM_GENERIC` | `[PAD]` | `PAD` |

   In illustrations, numerals are drawn as `0 1 2 …` for readability; in actual problem instances they are arbitrary bytes sampled from $\Sigma$.
4. **Document Map**:
   - [`01_research_goal.md`](01_research_goal.md): hypothesis, invariants, success and falsification criteria.
   - [`02_mechanics.md`](02_mechanics.md): runtime paradigms (`SM-MH`, `SM-Seq`, `AR`), sequence grammars, harness contracts.
   - [`03_levers.md`](03_levers.md): design levers (tiers, vocabulary, embedding, heads, depth, masking, FFN, base regime).
   - [`04_variants.md`](04_variants.md): gated primary model matrix and ordered experiment sequence.
   - [`05_ablation_variants.md`](05_ablation_variants.md): confound glossary, ablations and negative controls.
   - [`06_implementation.md`](06_implementation.md): pure PyTorch engineering blueprint, tensor layouts, and test contracts.
   - [`07_potential_outcomes.md`](07_potential_outcomes.md): taxonomy of empirical sub-results, scientific discoveries, and theoretical branches.
5. **Implementation Status**: Codebase skeleton and test harnesses implemented in `transformadder/`, `experiments/`, and `tests/`.
