# Evaluation Protocol, Out-of-Distribution Cells, and Formal Metrics

**Specification Document 05**  
**TransformAdder Project**

---

## 1. Zero-Shot Out-of-Distribution Protocol

To test for genuine algorithmic induction rather than statistical memorisation or pattern interpolation, models are subjected to a rigorous **multi-axis zero-shot out-of-distribution (OOD) evaluation suite**.

All evaluation tasks are performed strictly zero-shot:
* **Zero Parameter Adaptation**: Model weights $\theta$ are frozen; no fine-tuning or adaptation steps are permitted.
* **Zero In-Context Exemplars**: Prompts contain strictly the single problem instance; no few-shot demonstrations are prepended.
* **Deterministic Seeds**: Evaluation sets are procedurally synthesized using dedicated, fixed evaluation random seeds distinct from training seeds.

---

## 2. The Six Evaluation Cells

The evaluation suite tests model performance across three independent axes of generalisation: length extrapolation, radix shift (interpolation and extrapolation), and lexical alphabet permutation.

```
                           THE 6 EVALUATION TEST CELLS
                           
  Cell 1: In-Domain Baseline           B in {2, 4, 8, 10},  L in [1, 20]
  Cell 2: Length Extrapolation         B in {2, 4, 8, 10},  L in [21, 40]   (2x Training Length)
  Cell 3: Unseen Radix Interpolation   B = 3 (Odd Radix),   L in [1, 20]
  Cell 4: Unseen Radix Extrapolation   B = 12 (Radix > 10), L in [1, 20]
  Cell 5: Scrambled Axiom Permutation  B in {2, 4, 8, 10},  A ~ Perm(Sigma)
  Cell 6: Combined Stress Test         B = 12,              L in [21, 40],  A ~ Perm(Sigma)
```

### 2.1 Cell 1: In-Domain Test
* **Parameters**: Radices $B \in \{2, 4, 8, 10\}$, operand lengths $L_a, L_b \in [1, 20]$. Evaluated on both canonical ASCII alphabets (`0`..`9`) and randomly permuted byte alphabets.
* **Objective**: Verifies baseline optimization convergence. Near-perfect performance ($> 99.0\%$ EM) is a prerequisite for valid out-of-distribution claims.

### 2.2 Cell 2: Length Extrapolation (Double-Length Stress Test)
* **Parameters**: Training radices $B \in \{2, 4, 8, 10\}$, operand lengths $L_a, L_b \in [21, 40]$.
* **Objective**: Evaluates whether the learned addition circuit can execute indefinitely across columns without accumulating spatial error or phase drift. Tests whether positional representations (e.g. `sinusoidal_pe` vs. `addressed_kv`) generalise beyond their training length window.

### 2.3 Cell 3: Unseen Radix Interpolation (Base 3)
* **Parameters**: Odd radix $B = 3$, operand lengths $L \in [1, 20]$.
* **Objective**: All standard training radices $\{2, 4, 8, 10\}$ share even parity or power-of-two arithmetic structures. Testing on $B = 3$ verifies that the model has learned base-conditioned modular arithmetic rather than hardcoded parity heuristics.

### 2.4 Cell 4: Unseen Radix Extrapolation (Base 12)
* **Parameters**: Radix $B = 12$ (duodecimal), operand lengths $L \in [1, 20]$.
* **Objective**: Evaluates generalisation to an alphabet cardinality strictly larger than any seen during training ($\max B_{\text{train}} = 10$). Tests whether successor lookup and threshold circuits can scale dynamically to larger alphabet spaces.

### 2.5 Cell 5: Scrambled Axiom Permutation
* **Parameters**: Radices $B \in \{2, 4, 8, 10\}$, where the alphabet $\mathcal{A} = [s_0, \dots, s_{B-1}]$ is sampled as an arbitrary permutation of random non-ASCII byte tokens from candidate pool $\Sigma$.
* **Objective**: Verifies that digit semantics are bound strictly to the in-context dictionary definition rather than spurious ASCII character biases.

### 2.6 Cell 6: Combined Stress Test (The Grand Challenge)
* **Parameters**: Unseen radix $B = 12$, double sequence lengths $L \in [21, 40]$, and randomly scrambled byte alphabets.
* **Objective**: The ultimate test of algorithmic universality. Concurrently exercises length generalisation, alphabet scale generalisation, and arbitrary symbol mapping.

---

## 3. Formal Metric Definitions

Algorithmic performance is quantified across five hierarchical metrics:

### 3.1 Full-Sequence Exact Match (EM) — Primary Gold Standard
The primary measure of algorithmic capability. A problem instance $i$ is counted as correct ($EM_i = 1$) if and only if every single predicted output digit and the termination flag exactly match the ground-truth reference:
$$\text{EM} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}_{\left( \widehat{OUT}^{(i)} = OUT_{\text{true}}^{(i)} \ \wedge \ \widehat{\text{status}}^{(i)} = \texttt{"finished"} \right)}$$
Any single error in any column renders $EM_i = 0$.

### 3.2 Per-Step Prediction Accuracy ($\text{Acc}_{\text{step}}$)
Measures the micro-level accuracy across all individual token predictions:
$$\text{Acc}_{\text{step}} = \frac{\sum_{i=1}^N \sum_{k=0}^{K^{(i)}} \left( \mathbf{1}_{\hat{o}_k = o_k} + \mathbf{1}_{\hat{c}_{k+1} = c_{k+1}} + \mathbf{1}_{\widehat{\text{FLAG}}_k = \text{FLAG}_k} \right)}{\sum_{i=1}^N 3(K^{(i)} + 1)}$$
Useful for detecting partial circuit formation or grokking transitions when EM is still near zero.

### 3.3 Carry Chain Cascade Accuracy ($\text{Acc}_{\text{carry}}$)
Measures accuracy specifically on long carry propagation chains (problems where an initial carry $c_i = 1$ propagates across $M \ge 3$ consecutive columns where $a_j + b_j = B - 1$):
$$\text{Acc}_{\text{carry}} = \frac{1}{|\mathcal{S}_{\text{chain}}|} \sum_{j \in \mathcal{S}_{\text{chain}}} \mathbf{1}_{\hat{c}_j = c_j}$$
Isolates whether the transformer has acquired a recursive carry-lookahead circuit or merely relies on local heuristic addition.

### 3.4 Halting & Loop Termination Accuracy ($\text{Acc}_{\text{halt}}$)
Quantifies the precision and recall of the loop termination signal:
$$\text{Acc}_{\text{halt}} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}_{\widehat{\text{FLAG}}_{K^{(i)}} = \texttt{FLAG\_FINISHED}} \cdot \prod_{k=0}^{K^{(i)}-1} \mathbf{1}_{\widehat{\text{FLAG}}_k = \texttt{FLAG\_CONTINUE}}$$

---

## 4. Evaluation Budget & Hardware Efficiency

1. **Evaluation Schedule**:
   - **Periodic Checkpoint Evaluation**: Every 5,000 training steps, a fast evaluation suite (100 problem instances per cell) is executed across all 6 evaluation cells to log training dynamics.
   - **Final Terminal Evaluation**: At step 100,000, a comprehensive evaluation suite (10,000 problem instances per cell) is executed to yield publication-grade statistical confidence intervals.
2. **Evaluation Hardware Optimization**:
   - **BF16 Autocast**: Evaluation forward passes are executed under `torch.amp.autocast(dtype=torch.bfloat16)` to guarantee that PyTorch SDPA routes directly to the FlashAttention-2 kernel rather than falling back to memory-intensive fp32 math.
   - **Length-Aware Chunking**: To prevent Out-Of-Memory (`CUDA OOM`) exceptions on long-sequence extrapolation cells (where addressed sequences reach up to ~920 tokens), evaluation batches are processed in bounded sub-chunks of 32 rows under `torch.no_grad()`.
