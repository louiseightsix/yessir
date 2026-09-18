# Ablation & Negative Control Variants

This document specifies the formal protocols for all ablation models, negative controls, and experimental interventions in **TransformAdder**, together with the complete glossary of experimental confounds.

Each control or ablation is designed to isolate a single causal mechanism in the architecture or training regime, establishing unambiguous falsification criteria.

---

## 1. Confound Glossary

### 1.1 Binary-Switch Confound
With only two training bases ($\mathcal{B}_{\text{train}} = \{2, 10\}$), the model can achieve perfect in-domain accuracy without learning a base-general algorithm: it need only classify the axiom into one of two cases (e.g. "short axiom" vs "long axiom") and switch between two memorised routines with hard-wired carry thresholds (2 and 10). Such a model never computes the threshold from $|\mathcal{A}|$, and fails on Base-3 and Base-12 by reciting one of its two routines. The Multimodal Diverse Regime ($\{2, 4, 8, 10\}$) mitigates this by raising the cost of a per-base switch relative to computing $B$ from the axiom; it cannot exclude an $n$-way switch in principle, which is why the recitation-failure criterion ([`01_research_goal.md`](01_research_goal.md) Section 4.3) inspects the *form* of errors on unseen bases.

### 1.2 Positional OOD Confound
A failure on longer operands or larger bases caused by positional representations never encountered during training, rather than by a failure of the learned algorithm. Two distinct forms:
* **Table overflow**: A learned positional lookup table has no trained entry for $p > 19$. Avoided in Tier 1 by functional sinusoidal encoding, and in Tiers 2–3 by having zero positional inputs (NoPE).
* **Coordinate phase domain shift**: In Tier 1, although relative column matching between operands is zero-distance ($\Delta = p_A - p_B = 0$) and mathematically shift-invariant, slow sinusoidal frequencies ($\omega_i \approx 10^{-4}$) on lengths $21 \dots 40$ produce input magnitudes outside the trained phase interval $[0, 20]$. This may cause non-linear SwiGLU layers to saturate, masquerading as algorithmic failure. Tier 2 and Tier 3 (NoPE) eliminate this confound completely.

### 1.3 Untrained Embedding Confound
If a byte from the symbol pool $\Sigma$ never appears during training, every parameter that reads its one-hot input column (first-layer $W_Q, W_K, W_V$ projection slices, FFN input weights) or writes its output logit receives no gradient. A failure on a problem containing that byte then reflects untrained parameters, not a failure of axiom binding. It is avoided by sampling axioms uniformly from all 240 bytes of $\Sigma$ in training, ensuring every byte is observed across all structural roles ([`01_research_goal.md`](01_research_goal.md) Section 2.1).

### 1.4 Delimiter Ambiguity & Stream Confound
In Tier 3 (Generic Delimiters + NoPE), all stream boundaries are denoted by the single identical delimiter token `[DELIM_GENERIC]` (`0xFB`). If self-attention fails to accurately count the ordinal index of `[DELIM_GENERIC]`, the model cannot determine whether a sequence of numerals corresponds to the axiom, Operand A, Operand B, or the accumulated output. This confound is isolated by contrasting Tier 3 with Tier 2 (where each stream is explicitly anchored by distinct semantic delimiters `[AXIOM]`, `[A]`, `[B]`, `[CARRY]`, `[OUT]`).

---

## 2. Negative Controls

Negative controls are designed to deliberately cripple a specific inductive mechanism to verify that the primary model's success is genuinely attributable to the hypothesised cause.

### 2.1 NC-1: Fixed Canonical Symbol Training Control
* **Hypothesis**: Dynamic per-instance axiom sampling from $\Sigma$ is the necessary and sufficient driver of in-context symbol binding. If trained on fixed symbols, the model collapses into static weight memorisation.
* **Protocol**:
  1. **Training Regime**: The model is trained exclusively using fixed canonical ASCII symbols: $\mathcal{A} = [48, 49, \dots, 48 + B - 1]$ (i.e. `'0'`, `'1'`, `'2'`, ...). No symbol permutations or random byte assignments are ever presented during training.
  2. **Evaluation**: Evaluated on problems using randomised byte symbols sampled from $\Sigma \setminus \{\text{ASCII}\}$.
* **Expected Outcome & Falsification**:
  * The model achieves 100% Exact Match on the canonical ASCII symbols, but drops to near 0% on randomised axioms.
  * Confirms that fixed one-hot representations alone do not prevent weight-memorised token associations; dynamic instance-level randomisation is mandatory to force axiom table lookup.

### 2.2 NC-2: Single-Base Curriculum Control (Base-10 Only)
* **Hypothesis**: Exposure to multiple numerical bases during training ($\{2, 4, 8, 10\}$) forces the transformer to learn a universal in-context lookup algorithm, whereas single-base training collapses into static carry-threshold memorisation.
* **Protocol**:
  1. **Training Regime**: The model is trained exclusively on Base-10 arithmetic ($B=10$), using randomised symbol permutations for the 10 numerals.
  2. **Evaluation**: Evaluated zero-shot on unseen Base-3 (interpolation) and Base-12 (extrapolation).
* **Expected Outcome & Falsification**:
  * *Outcome A (Memorisation collapse)*: Model achieves 100% on Base-10, but fails on Base-3 (frequently emitting modulo-10 digits or failing to trigger carry at 3) and Base-12 (unable to process numerals with value $\ge 10$). Proves the *In-Context Grounding Hypothesis*: multi-base diversity is required to trigger algorithmic meta-learning.
  * *Outcome B (Zero-shot generalisation)*: Model achieves 100% on Base-3 and Base-12, demonstrating an intrinsic inductive bias toward in-context axiom lookup over weight memorisation.

### 2.3 NC-3: Shuffled Axiom Counterfactual Evaluation Control
* **Hypothesis**: The model computes output digits by dynamically reading the in-context axiom prefix $\mathcal{A}$ rather than relying on internal heuristic shortcuts or residual bias.
* **Protocol**:
  1. **Baseline**: Take an evaluation problem $(a, b)$ under valid axiom $\mathcal{A} = [s_0, s_1, s_2, \dots, s_{B-1}]$. The model predicts ground-truth output $OUT$.
  2. **Counterfactual Intervention**: Without modifying the operand byte sequences $a$ and $b$, permute two symbols in the axiom: $\mathcal{A}' = [s_1, s_0, s_2, \dots]$ (swapping the symbols for value 0 and value 1).
  3. **Re-Evaluation**: Pass the intervened prompt $(\mathcal{A}', a, b)$ to the model. Under $\mathcal{A}'$, the mathematically sound sum is $OUT' \ne OUT$.
* **Expected Outcome & Falsification**:
  * The model outputs $OUT'$, proving causal dependence on the axiom prefix.
  * If the model continues to output $OUT$, it is falsified as failing in-context grounding.

### 2.4 NC-4: Subspace Zeroing Intervention Control (Tier 1)
* **Hypothesis**: In Tier 1, column alignment relies strictly on the orthogonal positional coordinates ($p_t$) in dimensions $256 \dots 511$.
* **Protocol**:
  1. At test time, zero out the positional subspace of the input embeddings:
     $$\mathbf{x}_t \leftarrow [\mathbf{e}_{b_t} \parallel \mathbf{0}_{256}]$$
  2. Measure Exact Match across In-Distribution problems ($L \le 20$).
* **Expected Outcome & Falsification**:
  * Accuracy drops immediately to near-chance levels.
  * Proves that Tier 1 models actually utilise the geometric subspace for column alignment rather than finding unintended bypass channels.

---

## 3. Architectural & Mechanistic Ablations

Architectural ablations isolate the contribution of specific structural components by replacing them with simpler or standard alternatives.

### 3.1 ABL-1: Circuit Depth Compression ($N_{\text{layers}} \in \{2, 3\}$ vs $6$)
* **Hypothesis**: The minimal algorithmic circuit for in-context base-conditioned addition requires 3 sequential operations: (1) operand column retrieval, (2) axiom lookup and sum/carry evaluation, and (3) carry state propagation. A 2-layer model cannot resolve this circuit unless polysemantic residual folding occurs.
* **Configurations**:
  * `SM-[M*]-T1-L2-SwiGLU-Causal` ($N_{\text{layers}} = 2$)
  * `AR-T3-L3-SwiGLU-Causal` ($N_{\text{layers}} = 3$)
  * Baseline: $N_{\text{layers}} = 6$
* **Evaluation**: Compare Exact Match curves on ID ($L \le 20$) and unseen Base-3 / Base-12.
* **Falsification Criteria**: If $N=2$ fails to converge or plateaus while $N=3$ achieves 100% EM, it empirically proves the 3-layer circuit complexity lower bound for in-context arithmetic.

### 3.2 ABL-2: Feed-Forward Non-Linearity Engine (`GELU` vs `SwiGLU`)
* **Hypothesis**: Bilinear gating ($\text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x$) provides a strong mathematical inductive bias for computing modular thresholds and carry divisions ($v_A + v_B \ge B$) compared to standard continuous activations.
* **Configurations**:
  * `SM-[M*]-T1-L6-GELU-Causal`: Pre-LN transformer replacing SwiGLU with standard 2-layer MLP ($W_1 \in \mathbb{R}^{512 \times 2048}$, $\text{GELU}$, $W_2 \in \mathbb{R}^{2048 \times 512}$). Parameter count matched to reference SwiGLU.
  * Baseline: Reference SwiGLU ($d_{\text{ffn}} = 1376$).
* **Evaluation**: Convergence speed (steps to 99% EM) and out-of-distribution base transfer on Base-3 and Base-12.

### 3.3 ABL-3: Attention Visibility Regime (`Prefix-LM` vs Strict Causal)
* **Hypothesis**: Bidirectional attention over the static prompt prefix (axiom $\mathcal{A}$ and operands $a, b$) improves column alignment and operand length parsing compared to strict causal masking.
* **Configurations**:
  * `GRID-PREFIX` ([`04_variants.md`](04_variants.md) Section 3), Tier 1 only: custom attention mask where the static prefix $(\texttt{[BOS]}, \mathcal{A}, a, b)$ attends bidirectionally; all subsequent tokens attend causally ([`03_levers.md`](03_levers.md) Section 6). Not defined for Tiers 2–3: under NoPE a bidirectional prefix loses the axiom's order.
  * Baseline: Strict lower-triangular causal attention mask over the entire sequence.
* **Evaluation**: Length extrapolation ($L \in [21, 40]$) and attention entropy on operand tokens.

### 3.4 ABL-4: Decoding Topology (Zero-Parameter Copy Circuit vs Learned Multi-Head Projections)
* **Hypothesis**: Arithmetic symbol emission is an address-routing task in representation space. A parameter-free tied copy head ($\mathbf{y}_t = \text{RMSNorm}_0(\mathbf{h}_t)[:256]$) provides equivalent accuracy to learned projection matrices while eliminating 264K parameters and preventing output weight memorisation.
* **Configurations**:
  * `SM-Seq-T1-L6-SwiGLU-Causal` (Zero-parameter tied identity copy circuit).
  * `SM-MH-T1-L6-SwiGLU-Causal` (Three learned projection heads $W_{\text{sum}}, W_{\text{carry}}, W_{\text{flag}}$).
* **Evaluation**: Direct comparison at **Gate 1** of the experiment pipeline across parameter efficiency, convergence rate, and length extrapolation.

---

## 4. Ablation & Control Protocol Summary Table

| Identifier | Type | Target Mechanism | Baseline Configuration | Intervention / Alternative | Primary Diagnostic Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NC-1** | Negative Control | In-context symbol binding | Dynamic sampling from $\Sigma$ | Fixed ASCII symbols (`'0'`–`'9'`) | EM on randomised test axioms (expected: 0%) |
| **NC-2** | Negative Control | Universal base meta-learning | Multi-base $\{2, 4, 8, 10\}$ | Single-base $\{10\}$ | EM on unseen Base-3 & Base-12 |
| **NC-3** | Negative Control | Causal axiom dependence | Standard prompt $(\mathcal{A}, a, b)$ | Swapped axiom $(\mathcal{A}', a, b)$ | Concordance with $OUT'$ vs $OUT$ |
| **NC-4** | Negative Control | Coordinate reliance (Tier 1) | Full split embedding ($d=512$) | Positional slice zeroed ($\mathbf{p}_t = \mathbf{0}$) | In-distribution EM drop (expected: near chance) |
| **ABL-1** | Architecture | Circuit depth lower bound | $N_{\text{layers}} = 6$ | $N_{\text{layers}} \in \{2, 3\}$ | Minimum $N$ achieving 99% ID and OOD EM |
| **ABL-2** | Architecture | Carry & modulo engine | SwiGLU ($d_{\text{ffn}} = 1376$) | Standard GELU MLP ($d_{\text{ffn}} = 2048$) | Steps to convergence & OOD base transfer |
| **ABL-3** | Architecture | Prefix token representation | Strict causal masking | Prefix-LM bidirectional masking | Length extrapolation EM ($L \in [21, 40]$) |
| **ABL-4** | Architecture | Output head parameterisation | Multi-Head linear projections | Zero-parameter tied copy circuit | Gate 1 showdown: EM and parameter efficiency |
