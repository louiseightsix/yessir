# Research Goal: Zero-Shot Algorithmic Generalisation of Base-Conditioned Addition

## 1. Core Hypothesis

A transformer trained from scratch to execute step-by-step addition can decouple the abstract addition algorithm from the specific symbols used to train it. 

By defining the alphabet of digits in an in-context **axiom** and executing addition via an explicit runtime contract—either as an **Iterative Discrete State Machine (`SM`)** or as a **Continuous Autoregressive Trace (`AR`)** (detailed in [`02_mechanics.md`](02_mechanics.md))—the model will learn a generalised symbol-manipulation procedure. When trained on two or more numerical bases and presented with a novel, unseen numerical base—defined solely via an axiom of previously unseen length—the model will execute the addition algorithm zero-shot without requiring any parameter updates or in-context demonstrations.

```
       In-Context Problem Prefix                 Execution Mechanics (02_mechanics.md)
+------------------------------------+     +-----------------------------------------------+
| Axiom (Ordered Digit Alphabet)     |     | Paradigm A: Discrete State Machine (SM)       |
| Operands A, B (Unchanged Context)  | --> | • Step loop: (c_k, OUT_k) -> (o_k, c_{k+1}, FLAG) |
| (Defines base & problem instances) |     | Paradigm B: Autoregressive Trace (AR)         |
|                                    |     | • Single prompt -> unrolled scratchpad rollout|
+------------------------------------+     +-----------------------------------------------+
```

---

## 2. Information Flow and Context Structure

The computation is anchored on an in-context numeral definition, with state execution handled according to the selected runtime paradigm:

### 2.1 The Axiom (Base Definition)
The axiom is an ordered sequence of distinct symbols that establishes the numeral system for the current problem:
$$\mathcal{A} = [s_0, s_1, s_2, \dots, s_{B-1}], \quad s_i \in \Sigma = \mathcal{V} \setminus \mathcal{V}_{\text{ctrl}}$$

- **Semantics by Position**: The mathematical value of any symbol $s \in \mathcal{A}$ is defined strictly by its 0-indexed position within $\mathcal{A}$:
  $$\text{val}(s_i) = i, \quad \text{for } 0 \le i < B$$
- **Randomised Symbol Sampling**: To prevent the network from memorising static symbol-to-value associations in weights, the axiom of **every** problem instance (training and evaluation alike) is formed by sampling $B$ distinct bytes uniformly from the full pool $\Sigma$ (240 bytes), in random order. No byte has a stable value across instances, so the model must bind symbol semantics dynamically from the active prompt. This is the sole training regime of the primary experiment.
- **No Held-Out Symbols**: Every byte of $\Sigma$ is eligible in training. Holding bytes out for evaluation would conflate axiom binding with the Untrained Embedding Confound (first-layer weights reading a never-seen one-hot column receive no gradient; see [`05_ablation_variants.md`](05_ablation_variants.md) Section 1). What is novel at evaluation is the **base** (the axiom length $B$ and the arithmetic it implies), never the bytes.
- **Successor Function**: The model must infer the successor function $\text{succ}(s_i) = s_{i+1}$ and modular wrap-around $(\text{with carry})$ strictly by reading the ordered sequence $\mathcal{A}$ in its context. For example, if the axiom is $\mathcal{A} = [\texttt{0}, \texttt{1}, \texttt{2}, \texttt{X}]$, the model must infer that $B=4$, $\text{val}(\texttt{X}) = 3$, and $\texttt{2} + \texttt{1} = \texttt{X}$.

### 2.2 Execution Mechanics & State Transition
The runtime execution mechanics and sequence grammars are formally specified in [`02_mechanics.md`](02_mechanics.md). Computation is structured across two foundational paradigms:

1. **Paradigm A: Iterative Discrete State Machine (`SM`)**:
   - The prompt contains unchanged operands $a$ and $b$ formatted in **algorithmic reading order ($LSD \to MSD$, least-significant digit first)**, followed by incoming carry $c_k$, accumulated output $OUT_k$, and a terminal sum query token `[S]`.
   - The model dynamically indexes column $k$, processes one column per step, and produces 3 outputs: sum digit $o_k$, next carry $c_{k+1}$, and loop control $\text{FLAG}$.
   - **Sub-Variants of the External Harness Mechanic**:
     - **Paradigm A1: Multi-Head Single Forward Pass (`SM-MH`)**: Executes exactly 1 forward pass per column, emitting all 3 outputs simultaneously via parallel prediction heads off the final token representation.
     - **Paradigm A2: Sequential Micro-Autoregression (`SM-Seq`)**: Executes 3 sequential autoregressive steps per column ($o_k \to c_{k+1} \to \text{FLAG}$) via a standard single-head decoder.
     - **Paradigm A3: Carry-First Sequential Micro-Autoregression (`SM-SeqRev`)**: As A2, with the micro-step order inverted ($c_{k+1} \to o_k \to \text{FLAG}$).
     - **Paradigm A4: Branching State Machine (`SM-Fork`)**: Three independent queries per column (`[S]` $\to o_k$, `[C]` $\to c_{k+1}$, `[F]` $\to \text{FLAG}$) off the same base prompt via a standard single-head decoder; no emission order.
   - In all sub-variants, the external harness maintains zero arithmetic logic, merely accumulating the emitted digits and re-injecting the persistent state until `FINISHED`.
2. **Paradigm B: Continuous Autoregressive Trace (`AR`)**:
   - The prompt prefix $(\texttt{[BOS]}, \mathcal{A}, a, b, \texttt{[START]})$ is provided once.
   - The model unrolls the entire column-by-column computation autoregressively via plain greedy decoding, emitting an intermediate scratchpad trace ($\texttt{[S]} o_k \texttt{[C]} c_{k+1}$) until `[FINISHED]`.
   - The outgoing carry $c_{k+1}$ immediately precedes column $k+1$ in context, enabling 1-step Markovian carry retrieval.
   - The harness performs zero state re-injection and zero arithmetic. The multi-head / sequential distinction is exclusive to Paradigm A and does not apply to Paradigm B.

---

## 3. Invariants and Constraints

The experiment is governed by three non-negotiable scientific constraints:

### 3.1 Symbolic Collision and Positional Semantics
Because axioms are resampled for every instance (Section 2.1), the *same byte* takes on different values, and the *same value* different bytes, from one problem to the next, both within a base and across bases. For example, the byte $\texttt{X}$ may be $s_2$ in one Base-3 instance and $s_2$ in a Base-10 instance:
- Under the Base-3 axiom, $\texttt{X} + s_1$ wraps to $s_0$ and generates a carry.
- Under the Base-10 axiom, $\texttt{X} + s_1 = s_3$ and does not generate a carry.

In a third instance $\texttt{X}$ may be $s_7$, or absent altogether.

**The Invariant**: A symbol has no meaning outside the active axiom. The model must derive every symbol's value, and the carry threshold $B$, exclusively from the symbol's position in, and the length of, the prompt's axiom; any static, globally learned association between a byte and a value is necessarily wrong on most instances.

> [!NOTE]
> A training regime with *fixed canonical* symbols (where static associations can form and must be overridden at evaluation) is not part of the primary experiment; it is specified as a negative control in [`05_ablation_variants.md`](05_ablation_variants.md).

### 3.2 Arbitrary Scaling (Length Invariance)
The learned algorithm must be length-invariant. It must:
- Align operands of mismatched lengths ($|a| \ne |b|$) without physical padding; when offset $k \ge |a|$ or $k \ge |b|$, the model must *learn* to detect the absence of an operand digit and treat it as $s_0$ (arithmetic zero). Attention cannot attend to "nothing"; the constant `[BOS]` token that opens every sequence provides the null key for such look-ups (see [`02_mechanics.md`](02_mechanics.md) Section 2).
- Propagate carry chains of arbitrary length ($1, 2, \dots, K$ consecutive carries).
- Generalise to sequence lengths substantially longer than those observed during training, bounded only by the physical context window.

### 3.3 Strict Zero-Shot Boundary
Evaluation on novel bases is conducted under strict zero-shot constraints:
- **Zero Weight Updates**: Model weights are frozen.
- **Zero In-Context Demonstrations**: The prompt contains exactly one problem instance preceded by its axiom. No few-shot exemplar problems or solved demonstrations are provided in context.
- **Pure Axiomatic Conditioning**: The model receives only the novel axiom $\mathcal{A}_{\text{eval}}$ and the raw operands $a, b$.

---

## 4. Evaluation and Success Conditions

### 4.1 Base Distributions
* **Core Hypothesis**: A model trained on a set of two or more distinct numerical bases $\mathcal{B}_{\text{train}} \subset \mathbb{N}_{\ge 2}$ will generalise zero-shot to unseen bases $B \notin \mathcal{B}_{\text{train}}$ via in-context axiom conditioning. The hypothesis is *tested* on $\mathcal{B}_{\text{eval}} = \{3, 12\}$ ($\mathcal{B}_{\text{eval}} \cap \mathcal{B}_{\text{train}} = \emptyset$ in every regime); claims about other unseen bases are extrapolations from these two.
* **Training Regimes**:
  * **Minimal Bimodal Baseline**: $\mathcal{B}_{\text{train}} = \{2, 10\}$. The strict minimal lower-bound regime testing if two bases suffice to induce the abstract algorithm.
  * **Multimodal Diverse Regime**: $\mathcal{B}_{\text{train}} = \{2, 4, 8, 10\}$. Tests whether exposing the model to diverse cardinalities mitigates the Binary-Switch Confound ([`05_ablation_variants.md`](05_ablation_variants.md) Section 1) and yields more robust zero-shot transfer.
* **Operand Lengths**:
  * **Standard length**: $|a|$ and $|b|$ drawn independently and uniformly from $[1, 20]$. This is the training distribution.
  * **Extrapolation length**: $|a|$ and $|b|$ drawn independently and uniformly from $[21, 40]$.
  * Operand digits are drawn from $\mathcal{A}$; most-significant zeros are permitted. A fixed fraction of instances carries a deliberately constructed long carry chain, since uniformly random digits almost never produce one (see [`06_implementation.md`](06_implementation.md) Section 4.1).
* **Evaluation Targets**:
  * **In-Domain Verification**: Every base $B \in \mathcal{B}_{\text{train}}$ of the active regime, at standard length.
  * **Interpolation**: Base-3 ($B=3$), lying strictly between training cardinalities in both regimes.
  * **Extrapolation**: Base-12 ($B=12$), beyond the maximum cardinality seen during training.
  * **Length Extrapolation**: Extrapolation length, evaluated across both seen and unseen bases.
  * **Axiom Symbols**: Every evaluation cell is run with (i) *random* axioms sampled exactly as in training, and (ii) *canonical* axioms using the conventional ASCII digits (`0`–`9`, then `A`, `B`) in order. Under the hypothesis the two must agree; the canonical set is a fixed special case of the random distribution, included for interpretability and as a parity check.
* **Evaluation Protocol**: Each evaluation cell (base $\times$ length band $\times$ axiom set) comprises 10,000 freshly sampled problems, disjoint in seed from training. Every configuration is trained with 3 independent seeds; a criterion is met only if it is met by **all** seeds.

### 4.2 Success Criteria
Metrics are defined in [`06_implementation.md`](06_implementation.md) Section 7.2. The hypothesis is **supported** if and only if all of the following hold:
1. **Algorithmic Transfer**: Free-running exact-match accuracy (full $OUT$ sequence correct *and* correct halting) $> 99\%$ at standard length in both Base-3 and Base-12, zero-shot, on both axiom sets.
2. **Step-Transition & Halting Fidelity**: Teacher-forced step-transition accuracy $\ge 99.9\%$ in Base-3 and Base-12, where a transition ($OUT_k, c_k \to o_k, c_{k+1}, \text{FLAG}_k$, with $OUT_{k+1} = OUT_k \parallel [o_k]$) counts as correct only if all three outputs are correct; and halting accuracy $\ge 99.9\%$.
3. **Extrapolative Scaling**: Free-running exact-match accuracy $\ge 90\%$ at extrapolation length in Base-3 and Base-12.

### 4.3 Falsification Criteria
The hypothesis is **falsified** if, with in-domain exact match $> 99\%$ (i.e. the model has demonstrably learned addition on $\mathcal{B}_{\text{train}}$), any of the following holds at standard length:
- **Binding failure**: exact-match accuracy $< 5\%$ on Base-3 or Base-12. The model has memorised base-specific routines and fails to bind to the axiom.
- **Recitation failure**: on Base-3 or Base-12, the majority of erroneous step transitions coincide with the output that a training base's arithmetic would produce (e.g. carrying at threshold 10 or 2 under a Base-12 axiom).
- **Symbol-dependence failure**: exact-match accuracy differs by more than 5 percentage points between the random and canonical axiom sets for the same base, showing that behaviour depends on which bytes are used rather than on their axiom positions.

### 4.4 Partial Outcomes
Any result that is neither *supported* nor *falsified* (for example 60% exact match on Base-12, or transfer to Base-3 but not Base-12) is reported as a **partial** outcome, broken down by base, length band, and step-transition error type. If in-domain exact match is $\le 99\%$, the run is **inconclusive** (a training failure), not evidence about the hypothesis.
