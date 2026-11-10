# Research Goal: Zero-Shot Base-Conditioned Addition

**Specification Document 01**

---

## 1. Core Hypothesis

A transformer trained from scratch to execute step-by-step addition can separate the addition algorithm from the specific symbols and bases it was trained on.

Each problem states its numeral system in context, as an ordered alphabet called the **axiom**. If the model learns to read the axiom rather than memorise symbol values, then when it is trained on two or more bases and shown an unseen base (defined only by an axiom of a new length), it will add correctly zero-shot: no parameter updates and no in-context demonstrations.

```
       In-Context Problem                          Execution (02_mechanics.md)
+------------------------------------+     +-----------------------------------------------+
| Axiom (ordered digit alphabet)     |     | Harnessed state machine (SM-*)                |
| Operands a, b                      | --> |   one column per step:                        |
| (defines base and problem)         |     |   (c_k, OUT_k) -> (o_k, c_{k+1}, FLAG_k)       |
|                                    |     | Autonomous trace (AR-*)                       |
|                                    |     |   one prompt -> unrolled scratchpad            |
+------------------------------------+     +-----------------------------------------------+
```

---

## 2. The Axiom (In-Context Numeral System)

$$\mathcal{A} = [s_0, s_1, \dots, s_{B-1}], \quad s_i \in \Sigma = \mathcal{V} \setminus \mathcal{V}_{\text{ctrl}}$$

- **Meaning by position**: a symbol's value is its 0-indexed position in $\mathcal{A}$: $\text{val}(s_i) = i$. The base is the axiom's length, $B = |\mathcal{A}|$.
- **Randomised symbols**: in the primary regime, every instance (training and evaluation) samples $B$ distinct bytes uniformly from all 240 bytes of $\Sigma$, in random order. No byte has a stable value across instances, so symbol meaning must be read from the prompt.
- **No held-out symbols**: every byte of $\Sigma$ appears in training. Holding bytes out would confound axiom binding with untrained embeddings ([`04`](04_experimental_matrix.md) §6, Untrained Embedding Confound). What is novel at evaluation is the **base** (the axiom length and the arithmetic it implies), not the bytes.
- **Successor and wrap-around**: the model must infer $\text{succ}(s_i) = s_{i+1}$ and the modular wrap-around with carry purely from the order of $\mathcal{A}$. For $\mathcal{A} = [\texttt{0}, \texttt{1}, \texttt{2}, \texttt{X}]$ it must infer $B = 4$, $\text{val}(\texttt{X}) = 3$, and $\texttt{2} + \texttt{1} = \texttt{X}$.

The alphabet regimes used to *control* how much the model's learned prior agrees or conflicts with the axiom (`symbols_random`, `canonical_order`, `canonical_permuted`, and the training dial $p_{\text{prior}}$) are defined in [`03_representations.md`](03_representations.md) §2.

---

## 3. Invariants

### 3.1 Symbol collision: a symbol means nothing outside its axiom
Because axioms are resampled per instance, the same byte takes different values from one problem to the next, both within a base and across bases. Byte `X` may be $s_2$ in a Base-3 instance (where $\texttt{X} + s_1$ wraps to $s_0$ with carry) and $s_2$ in a Base-10 instance (where $\texttt{X} + s_1 = s_3$, no carry), or $s_7$, or absent. Any static byte-to-value association is wrong on most instances; values and the carry threshold $B$ must come from the axiom.

### 3.2 Length invariance
The learned procedure must:
- align operands of different lengths without padding, treating a missing digit ($k \ge |a|$ or $k \ge |b|$) as $s_0$;
- propagate carry chains of any length;
- extend to operands longer than any seen in training, bounded only by the context window.

### 3.3 Strict zero-shot evaluation
- **Frozen weights**: no parameter updates at evaluation.
- **No demonstrations**: each prompt contains exactly one problem and its axiom.
- **Procedural data**: all data is a pure function of seed and step index; no dataset on disk.
- **Exact match is primary**: a problem is correct only if every output digit and the halting decision match the reference.

---

## 4. Research Questions

1. **Base transfer (the core question)**: does training on several bases produce a procedure that reads $B$ and digit values from the axiom, and so transfers to Base-3 (interpolation) and Base-12 (extrapolation)?
2. **Agency**: can the model carry its own execution state (column pointer, carry, halting) across an autonomous trace, or does it need an external harness to re-present state each step? ([`02`](02_mechanics.md) §4)
3. **Causal topology**: within a column, does emitting the carry before the sum digit (Carry-First: a monotone threshold, then a conditional subtraction) make the step easier to learn and more transferable than Sum-First? ([`02`](02_mechanics.md) §5)
4. **Readout channel**: do static linear heads tie predictions to absolute coordinates, and do sequence-space query triggers with a tied copy head restore coordinate-free retrieval? ([`02`](02_mechanics.md) §6)
5. **Representation**: how do positional encodings, delimiter schemes and explicit numeral addresses affect column alignment and length extrapolation? Can the alphabet be inferred with no axiom at all? ([`03`](03_representations.md) §3)
6. **Prior suppression**: when training makes the conventional digit order familiar ($p_{\text{prior}} > 0$), does the model still follow a *permuted* axiom, and when it fails, does it fall back to the familiar order (prior intrusion)? ([`03`](03_representations.md) §2)
7. **Binding vs. execution**: when the model fails, did it fail to read the axiom, or fail to execute the multi-step procedure? A single-token successor probe separates the two. ([`05`](05_evaluation_and_metrics.md) §4)

---

## 5. Success and Falsification Criteria

Bases, cells and metrics are defined in [`05_evaluation_and_metrics.md`](05_evaluation_and_metrics.md). Each configuration is trained with 3 seeds; a criterion is met only if all seeds meet it.

### 5.1 Supported
The core hypothesis is **supported** if and only if all of the following hold:
1. **Algorithmic transfer**: free-running exact match $> 99\%$ at standard length on Base-3 and Base-12, zero-shot, on both random and canonical axioms.
2. **Step and halting fidelity**: teacher-forced step-transition accuracy $\ge 99.9\%$ on Base-3 and Base-12 (all of $o_k$, $c_{k+1}$, $\text{FLAG}_k$ correct), and halting accuracy $\ge 99.9\%$.
3. **Length extrapolation**: free-running exact match $\ge 90\%$ at extrapolation length on Base-3 and Base-12.

### 5.2 Falsified
Given in-domain exact match $> 99\%$ (the model has learned addition on its training bases), the hypothesis is **falsified** if any of the following hold at standard length:
- **Binding failure**: exact match $< 5\%$ on Base-3 or Base-12; the model has memorised per-base routines.
- **Recitation failure**: on Base-3 or Base-12, most erroneous transitions match what a *training* base's arithmetic would produce (e.g. carrying at 10 or 2 under a Base-12 axiom).
- **Symbol dependence**: exact match differs by more than 5 percentage points between random and canonical axioms for the same base.

### 5.3 Partial and inconclusive
Anything neither supported nor falsified (e.g. 60% on Base-12, or transfer to Base-3 but not Base-12) is a **partial** outcome, reported by base, length band and error type. If in-domain exact match is $\le 99\%$, the run is **inconclusive** (a training failure), not evidence about the hypothesis.

---

## 6. Scope

1. **Exact rules only**: the rule is formal and unambiguous; fuzzy or conversational compliance is out of scope.
2. **No rule selection**: the active axiom is always given; the model never has to decide which of several rules applies (except under `zero_preamble_kv`, where it must infer the alphabet from addressed operands).
3. **Structural failures only**: failures here come from attention capacity, context distance, coordinate drift, procedural error or prior intrusion, not from natural-language understanding.
