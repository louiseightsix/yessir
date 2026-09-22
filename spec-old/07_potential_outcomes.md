# Potential Experimental Outcomes & Scientific Discoveries

This document catalogues the full taxonomy of theoretical hypotheses, empirical sub-results, and potential scientific discoveries that TransformAdder is positioned to establish. 

Rather than viewing the project as a binary "does it add or not" test, TransformAdder decomposes arithmetic execution into fundamental mechanistic questions concerning transformer inductive biases, emergent coordinate induction, circuit compression, and in-context grounding.

---

## 1. Orthogonal Subspace PE vs. Length Extrapolation (The Tier 1 Discovery)

### The Core Question
Can an explicit positional encoding achieve zero-shot length extrapolation ($L \in [21, 40]$ from training on $L \le 20$) if it is **dimensionally orthogonal** to symbol identity and **reset per operand**?

$$\mathbf{x}_t = [\mathbf{e}_{b_t} \parallel \mathbf{e}_{\text{role}}(r_t) \parallel \mathbf{e}_{\text{sig}}(p_t)], \quad \mathbf{e}_{b_t} \in \mathbb{R}^{256}, \quad \mathbf{p}_t \in \mathbb{R}^{256}$$

* In TransformAdder, column alignment between Operand A and Operand B requires zero relative distance ($\Delta = p_A - p_B = k - k = 0$).
* The inner product of sinusoidal vectors at $\Delta = 0$ is identically constant for all $k$:
  $$\mathbf{e}_{\text{sig}}(k)^\top \mathbf{e}_{\text{sig}}(k) = \sum_{i=0}^{95} \left(\sin^2(k\omega_i) + \cos^2(k\omega_i)\right) \equiv 96$$

### Potential Outcomes
* **Outcome 1A (Extrapolation Success — Refutation of Conventional Wisdom)**:
  * *Observation*: `SM-MH-T1` and `SM-Seq-T1` achieve near 100% Exact Match on unseen lengths 21–40.
  * *Scientific Impact*: Disproves the widespread belief that sinusoidal positional encodings inherently break on length extrapolation. Demonstrates that historical failures in standard LLMs were an artefact of **additive superposition** ($\mathbf{e}_{\text{token}} + \mathbf{e}_{\text{pos}}$) and **1D monotonic sequence indexing**, rather than a limitation of sinusoidal basis functions. Orthogonal subspace isolation is sufficient for length generalisation.
* **Outcome 1B (Extrapolation Failure — Vindication of NoPE Necessity)**:
  * *Observation*: Models achieve 100% on $L \le 20$ but collapse on $L > 20$.
  * *Scientific Impact*: Demonstrates that even when the attention dot product is mathematically shift-invariant ($\Delta = 0$), non-linear Feed-Forward Networks inevitably overfit to the observed coordinate phase domain ($p \le 20$). This establishes an empirical theorem: *Non-linear neural layers cannot generalise over explicit unbounded coordinates*, proving why **Tier 2 / Tier 3 (NoPE)** is an architectural requirement for infinite-length generalisation.

---

## 2. Emergent Coordinate Induction (The NoPE Frontier: Tier 2 vs. Tier 3)

### The Core Question
Can a transformer with **zero positional encoding (NoPE)** synthesise relative distance counters from discrete delimiters, and what is the minimum required scaffolding?

### Potential Outcomes
* **Outcome 2A (Semantic Delimiters Induce Linear Counting — Tier 2 Success)**:
  * *Observation*: `SM-[M*]-T2` achieves 100% length extrapolation on lengths 21–40.
  * *Scientific Impact*: Confirms that self-attention heads can synthesise an emergent relative coordinate system without any geometric guidance, by measuring attention decay or token hop distance backwards from role delimiters (`[A]`, `[B]`).
* **Outcome 2B (Dual Emergence Lower Bound — Tier 3 Success vs. Failure)**:
  * *Observation*: In Tier 3, where only a single generic delimiter (`|`) exists, the model must simultaneously track:
    1. *Global block identity*: Counting how many `|` tokens have occurred so far.
    2. *Local column offset*: Counting how many tokens have elapsed since the latest `|`.
  * *Scientific Impact*:
    * *If Tier 3 succeeds*: Establishes that causal self-attention can maintain nested hierarchical counters with zero architectural scaffolding.
    * *If Tier 3 fails while Tier 2 succeeds*: Pinpoints the exact boundary of inductive bias needed for computation—proving that transformers require distinct semantic anchors to partition multi-operand state machines.

---

## 3. Decoding Topology: Parameter Projections vs. Parameter-Free Copy Circuits

### The Core Question
Does an explicit parameterised output head ($W_{\text{sum}}, W_{\text{carry}}, W_{\text{flag}}$) provide any representational benefit over a zero-parameter tied copy circuit ($\mathbf{y}_t = \text{RMSNorm}_0(\mathbf{h}_t)[:256]$)?

### Potential Outcomes
* **Outcome 3A (Tied Copy Head Dominance / Equivalence)**:
  * *Observation*: `SM-Seq` matches or exceeds `SM-MH` accuracy with 264K fewer parameters.
  * *Scientific Impact*: Proves that arithmetic symbol generation in transformers is fundamentally a **subspace routing problem** rather than an arithmetic calculation problem in the output head. Because the identity basis is fixed in dims $0 \dots 255$, the internal layers do all the computational work, rendering linear projection heads superfluous.
* **Outcome 3B (Multi-Head Parallel Advantage)**:
  * *Observation*: `SM-MH` outperforms `SM-Seq`, or trains significantly faster.
  * *Scientific Impact*: Indicates that simultaneously predicting $(o_k, c_{k+1}, \text{FLAG})$ benefits from specialised projection weights that decouple gradients, whereas sequential prompt conditioning (`[S]`, `[C]`) forces residual interference across micro-steps.

---

## 4. Scaffolding vs. Autonomy (State Machine Harness vs. Autoregressive Rollout)

### The Core Question
What is the exact computational cost of autonomy? How much does performance degrade when removing external state-machine scaffolding and forcing the model to generate its own scratchpad?

```text
State Machine (Scaffolded):   Harness orchestrates (prompt_k -> target_k) transitions.
Autoregressive (Autonomous): Model unrolls [S] o_0 [C] c_1 ... FLAG_FINISHED continuously.
```

### Potential Outcomes
* **Outcome 4A (Autoregressive Equivalence)**:
  * *Observation*: `AR-T3` matches `SM-T3` accuracy across all lengths.
  * *Scientific Impact*: Demonstrates that the state machine loop can be fully internalised into the autoregressive generation process without compounding error. The model acts as its own operating system and driver.
* **Outcome 4B (Compounding Execution Drift)**:
  * *Observation*: `SM-T3` maintains 100% accuracy out to 40 digits, whereas `AR-T3` degrades after length 15–20.
  * *Scientific Impact*: Decouples **algorithmic competence** from **execution drift**. Proves that while transformers possess the representational capacity to execute arithmetic transitions, autonomous autoregressive generation suffers from probability dispersion or attention diffusion over long generation horizons.

---

## 5. Circuit Depth Lower Bounds ($N=2$ vs. $N=3$ vs. $N=6$)

### The Core Question
What is the minimum circuit depth required for in-context base-conditioned arithmetic?

* **Standard Theoretical Lower Bound ($N = 3$)**:
  * Layer 1: Attend to column $k$ of operands A and B via significance coordinate matching.
  * Layer 2: Attend to in-context axiom table to retrieve sum $o_k$ and carry $c_{k+1}$.
  * Layer 3: Compose carry propagation and route final symbol into output identity subspace.
* **The Compression Hypothesis ($N = 2$)**:
  * Can Layer 1 route operands and axiom simultaneously, allowing Layer 2 FFN to compute carry/sum in a single step?

### Potential Outcomes
* **Outcome 5A (Validation of the $N \ge 3$ Circuit Lower Bound)**:
  * *Observation*: 6-layer and 3-layer models achieve 100% EM, but 2-layer models fail completely or plateau at shallow accuracy.
  * *Scientific Impact*: Empirically validates circuit complexity theory for in-context formal languages, proving that the composition of operand retrieval, axiom indexing, and carry state requires at least 3 attention-FFN layers.
* **Outcome 5B (Discovery of 2-Layer Algorithmic Folding)**:
  * *Observation*: `SM-MH-T1-L2` achieves 100% EM across all bases.
  * *Scientific Impact*: Proves that transformer layers can fold multi-step relational lookups into single attention-FFN compositions via polysemantic residual channels, beating the naive sequential circuit depth bound.

---

## 6. Feed-Forward Non-Linearity: Bilinear Gating (SwiGLU) vs. Standard MLP (GELU)

### The Core Question
Is bilinear gating (SwiGLU) fundamentally superior for discrete modular arithmetic compared to continuous standard activations (GELU)?

$$\text{SwiGLU}(x) = W_{\text{down}}(\text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x)$$

### Potential Outcomes
* **Outcome 6A (Gating is Essential for Modular Arithmetic)**:
  * *Observation*: SwiGLU models converge in half the steps and achieve 100% OOD base generalisation, whereas GELU models struggle on Base-3 / Base-12.
  * *Scientific Impact*: Confirms that the elementwise product of two projected streams ($\text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x$) provides a natural mathematical inductive bias for computing division, modulo thresholds, and carry conditions ($v_a + v_b \ge B$).
* **Outcome 6B (Activation Invariance)**:
  * *Observation*: GELU achieves identical accuracy to SwiGLU.
  * *Scientific Impact*: Demonstrates that standard continuous feed-forward networks can approximate discrete piecewise modular thresholds through high-dimensional hyperplane partitioning without needing explicit multiplicative gating.

---

## 7. In-Context Axiom Grounding vs. Base Memorisation (Curriculum Generalisation)

### The Core Question
Does exposure to multiple bases during training ($\mathcal{B}_{\text{train}} = \{2, 4, 8, 10\}$) force the transformer to learn a universal in-context lookup algorithm, whereas single-base training collapses into static weight memorisation?

### Potential Outcomes
* **Outcome 7A (Phase Transition to In-Context Generalisation)**:
  * *Observation*:
    * Multi-base model achieves 100% zero-shot accuracy on unseen **Base-3 (interpolation)** and **Base-12 (extrapolation)**.
    * Single-base model (trained only on Base-10) achieves 100% on Base-10 but **0% on Base-3 and Base-12**, despite the axiom prefix being provided.
  * *Scientific Impact*: Proves that multi-base diversity induces a **meta-learning phase transition**: the model abandons weight-memorised arithmetic in favour of an in-context axiom retrieval algorithm.
* **Outcome 7B (Single-Base In-Context Induction)**:
  * *Observation*: Even when trained exclusively on Base-10, the model generalises zero-shot to Base-3 and Base-12.
  * *Scientific Impact*: Discovers that transformers have an intrinsic inductive bias toward in-context table lookup over weight memorisation when given dynamic symbol permutations, even without multi-base curriculum pressure.

---

## 8. Summary Matrix of Potential Scientific Discoveries

| Investigation Area | Test Contrast | Primary Outcome A | Primary Outcome B |
| :--- | :--- | :--- | :--- |
| **Positional Encoding** | Tier 1 (Split PE) vs Length Extrap. | Subspace orthogonality solves PE length extrapolation. | PE phase drift mandates NoPE for infinite extrapolation. |
| **Coordinate Induction** | Tier 2 vs Tier 3 NoPE | Emergent counting synthesised from semantic delimiters. | Dual emergent counting requires hierarchical delimiter tokens. |
| **Output Decoding** | `SM-MH` vs `SM-Seq` | Zero-parameter copy circuit matches parameterised heads. | Multi-head projections prevent micro-step cross-interference. |
| **Scaffolding** | State Machine vs Autoregressive | Autonomous trace matches external harness execution. | Compounding drift degrades long-horizon autonomous rollout. |
| **Circuit Depth** | $L=2$ vs $L=3$ vs $L=6$ | $N \ge 3$ represents strict circuit complexity lower bound. | Algorithmic folding compresses circuit into $N=2$. |
| **FFN Non-Linearity** | SwiGLU vs GELU | Bilinear gating is essential for modular carry engine. | Smooth MLP activations simulate discrete modulo thresholds. |
| **Curriculum Base** | Multi-Base vs Base-10 Only | Multi-base diversity triggers in-context grounding phase transition. | In-context axiom retrieval learned even under single-base regime. |
