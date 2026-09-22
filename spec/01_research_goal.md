# Research Goals, Operational Invariants, and Scientific Hypotheses

**Specification Document 01**  
**TransformAdder Project**

---

## 1. Primary Research Question

Standard transformer language models struggle with out-of-distribution generalisation on elementary algorithmic tasks such as multi-digit addition. When trained on fixed-length numbers in canonical decimal representation, autoregressive models notoriously fail when tested on inputs longer than those seen during training (length extrapolation) or when presented with alternative numerical formats.

This project investigates **zero-shot algorithmic induction in transformers**. Rather than memorising fixed digit transition tables in a single canonical radix, we ask:

> **Can a standard causal transformer learn a base-invariant algorithmic circuit for addition—decoupling column alignment, successor lookup, and carry propagation from the concrete symbols used to represent numerals—and transfer zero-shot to completely unseen numerical radices, unseen symbol alphabets, and arbitrary operand lengths?**

By supplying an explicit definition of the symbol sequence (the *axiom* or dictionary preamble) or forcing the model to infer digit succession from operand context, the task transforms from associative lookup into in-context procedural execution.

---

## 2. Theoretical Framing & Central Inquiries

The experimental program is designed around five core scientific questions that address the structural limits of attention-based sequence models:

### 2.1 Question 1: Supervisory Agency & State Virtualization
*Does external state management by a supervisory harness provide an artificial scaffold that masks fundamental limitations in autonomous autoregressive attention?*
* **Theoretical Motivation**: In a harnessed State Machine regime, Python maintains the carry state $c_k$ and the accumulated output sequence $OUT_k$, feeding the model fresh context slices for each step $\Delta s_k$. In an autonomous Autoregressive regime, the model must maintain all step counters, intermediate carries, and output histories within its own internal KV-cache. We seek to determine whether autonomous self-attention can maintain stable counter circuits across extended sequence lengths without accumulating phase error or context drift.

### 2.2 Question 2: Intra-Step Causal Topology & Directional Asymmetry
*Does the order of token emission within a column transition fundamentally determine computational tractability?*
* **Theoretical Motivation**: Computing column addition requires determining two outputs:
  $$\text{Sum: } o_k = (a_k + b_k + c_k) \pmod B$$
  $$\text{Carry: } c_{k+1} = \lfloor (a_k + b_k + c_k) / B \rfloor$$
  Under an order-free factored model ($P(o_k \mid s_k) \cdot P(c_{k+1} \mid s_k)$), the model must predict both values simultaneously from the same latent state. Under a sequential causal chain, the model emits one output and conditions on it to emit the second.
* **The Directional Asymmetry Hypothesis**:
  - In a **Sum-First** chain ($s_k \to o_k \to c_{k+1}$), predicting $o_k$ first requires computing a non-linear modular residue with zero intermediate scratchpad tokens.
  - In a **Carry-First** chain ($s_k \to c_{k+1} \to o_k$), computing carry first is a monotonic threshold decision ($a_k + b_k + c_k \ge B$), which linear self-attention layers can readily solve via linear separation. Conditioned on $c_{k+1}$, computing $o_k$ simplifies to conditional linear subtraction ($a_k + b_k + c_k - c_{k+1} \cdot B$).
  - We hypothesise that Carry-First causal topology provides a decisive algorithmic advantage over Sum-First in both harnessed micro-steps and autonomous scratchpads.

### 2.3 Question 3: Decoding Readout Channel & Coordinate Independence
*Why do static linear decoders fail when continuous positional encodings are removed, and can parameter-free tied copy decoders restore coordinate-free induction?*
* **Theoretical Motivation**: Standard sequence models decode predictions through static linear projection heads ($W \mathbf{h} + \mathbf{b}$). In a positional-encoding-free (NoPE) regime, a single tail hidden vector $\mathbf{h}_{\text{tail}}$ lacks a continuous coordinate basis to anchor distinct projection heads for sum, carry, and halting flags simultaneously. Conversely, sequence-space query triggers (`[S]`, `[C]`, `[F]`) paired with a parameter-free tied copy head ($W_{\text{head}} = E^T$) allow the model to route attention dynamically to relevant vocabulary items in context, decoupling prediction from static weight projections.

### 2.4 Question 4: Representational Inductive Bias & Address Matching
*Can transformers perform length extrapolation in coordinate-free spaces via relative delimiter counting, or is exact length generalisation strictly dependent on discrete key-value addressing?*
* **Theoretical Motivation**: Continuous positional encodings (sinusoidal or learned) inevitably suffer from out-of-distribution phase drift when sequence lengths exceed the training support. Delimiter counting forces self-attention to implement soft integration (counting delimiters in the residual stream), which exhibits high sample complexity and delayed convergence. In contrast, discrete numeral key-value addressing (tagging each column with an explicit symbol tuple $(v, k)$) reduces column lookup to exact in-context token matching, hypothesised to achieve zero-shot length generalisation independent of sequence length.

### 2.5 Question 5: Curriculum & Zero-Preamble Deduction
*Can a transformer infer the base alphabet entirely in-context from the operand operands without an explicit axiom dictionary?*
* **Theoretical Motivation**: If the explicit axiom header `[AXIOM] s_0 ... s_{B-1}` is omitted, the model must deduce the successor order of the alphabet strictly from co-occurrences and carry relationships within the operand strings. To make this problem solvable in-context, every digit in the radix must appear in the operands, imposing a structural curriculum constraint ($L \ge B$). We investigate whether transformers can learn this inductive deduction and whether the curriculum shift itself impacts algorithmic robustness.

---

## 3. Operational Invariants

To guarantee scientific validity, comparability, and reproducibility across all conditions, the experimental framework enforces five inviolable operational invariants:

1. **Universal 1-Byte Vocabulary Invariant**:
   - The token vocabulary is strictly the 256 individual byte values (`0x00`–`0xFF`). Sub-word tokenisers (BPE, WordPiece, SentencePiece) are strictly forbidden. Every symbol in the alphabet $\mathcal{A}$, every control marker, and every delimiter occupies exactly one token slot.
2. **Procedural Determinism (Zero Disk I/O)**:
   - Data generation is a pure mathematical function of the dataset seed and step index:
     $$\mathcal{D}(\text{seed}, \text{step}) \to \text{Batch}$$
   - No pre-generated datasets are read from disk. Training batches are synthesized on-the-fly with zero I/O bottlenecks.
3. **Strict Zero-Shot Evaluation Protocol**:
   - Out-of-distribution evaluation suites evaluate the exact same model weights on unseen radices ($B=3, 12$), extended sequence lengths ($L=21..40$), and scrambled symbol alphabets with **zero fine-tuning**, zero gradient updates, and zero few-shot exemplar prompts.
4. **Exact Match (EM) as the Primary Scientific Metric**:
   - While token-level cross-entropy loss and per-step accuracy are logged for training telemetry, the gold standard of algorithmic capability is **Full-Sequence Exact Match (EM)**: an evaluation problem is counted as correct if and only if every single output digit and the termination flag match the ground-truth reference sequence.
5. **Architectural Parity**:
   - All primary model comparisons maintain identical parameter budgets, hidden dimension ($d_{\text{model}} = 256$), feed-forward intermediate expansion ($d_{\text{ff}} = 1024$), number of layers ($L = 6$), and number of attention heads ($n_{\text{heads}} = 4$), isolating representational and topological inductive biases as the sole experimental variables.

---

## 4. Formal Falsification & Success Criteria

An architectural configuration is considered to have discovered a robust algorithmic solution if and only if it satisfies the following criteria:

| Evaluation Dimension | Success Criterion (Verified Algorithmic Induction) | Falsification Criterion (Heuristic Memorisation) |
| :--- | :--- | :--- |
| **In-Domain ($B \in \{2, 4, 8, 10\}, L \le 20$)** | $> 99.0\%$ Full-Sequence Exact Match. | Failure to converge to near-perfect accuracy on the training distribution ($< 95.0\%$). |
| **Length Extrapolation ($L = 21..40$)** | $> 90.0\%$ Full-Sequence Exact Match on 40-digit operands without degradation at boundary. | Complete collapse to $0.0\%$ Exact Match on lengths exceeding $L=20$ (e.g. continuous PE phase drift). |
| **Radix Interpolation ($B = 3$)** | $> 90.0\%$ Full-Sequence Exact Match on unseen odd base. | Catastrophic failure ($< 10.0\%$ EM), indicating memorisation of even-radix binary/decimal power structures. |
| **Radix Extrapolation ($B = 12$)** | $> 80.0\%$ Full-Sequence Exact Match on base exceeding maximum training radix ($B=10$). | Collapse on larger alphabets ($< 5.0\%$ EM), indicating fixed positional table capacity limits. |
| **Axiom Permutation ($\mathcal{A} \sim \text{Perm}(\Sigma)$)** | Invariant performance under random alphabet re-shuffling. | Performance drops when canonical ASCII digit order (`0`..`9`) is replaced by arbitrary random byte symbols. |
