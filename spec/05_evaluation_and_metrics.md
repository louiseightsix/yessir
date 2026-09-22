# Evaluation Protocol, Instance Telemetry, and Prior Suppression Metrics

**Specification Document 05**  
**TransformAdder Project: Foundational Framework**

---

## 1. Zero-Shot Out-of-Distribution Protocol

To evaluate genuine in-context rule following, models are tested strictly zero-shot:
* **Zero Parameter Updates**: Model weights $\theta$ remain frozen.
* **Zero Few-Shot Exemplars**: Prompts contain only the active rule definition and the query instance.
* **Deterministic Synthesis**: Evaluated using dedicated, fixed evaluation random seeds.

---

## 2. The Six Evaluation Cells (Stratified Across Prior Regimes)

Each of the six evaluation cells is stratified across the three alphabet regimes (`symbols_novel`, `canonical_order`, `canonical_permuted`) to isolate prior conflict from structural generalisation:

```
                           THE 6 EVALUATION TEST CELLS
                           
  Cell 1: In-Domain Baseline           B in {2, 4, 8, 10},  L in [1, 20]
  Cell 2: Length Extrapolation         B in {2, 4, 8, 10},  L in [21, 40]   (2x Training Length)
  Cell 3: Unseen Radix Interpolation   B = 3 (Odd Radix),   L in [1, 20]
  Cell 4: Unseen Radix Extrapolation   B = 12 (Radix > 10), L in [1, 20]
  Cell 5: Prior-Conflict Permutation   B in {2, 4, 8, 10},  canonical_permuted (Rule vs. Prior)
  Cell 6: Combined Stress Test         B = 12,              L in [21, 40],  canonical_permuted
```

---

## 3. The In-Context Binding Probe: Decoupling Retrieval from Use

In algorithmic rule-following, models frequently fail without researchers knowing *where* the breakdown occurred. Did the self-attention mechanism fail to bind the rule parameters, or did it fail during the multi-step sequential execution?

To resolve this ambiguity, the evaluation harness executes a **Single-Token Successor Binding Probe**:

```
                              THE BINDING PROBE PROTOCOL
                              
  Task Prompt:       [BOS] [AXIOM] s_0 s_1 s_2 ... [A] a [B] b ... [S]
                       │
  Binding Probe:     [BOS] [AXIOM] s_0 s_1 s_2 ... [A] a [B] b ... [PROBE] s_i
                       │
  Target Prediction: s_{i+1}   (Evaluated via tied copy head E^T)
```

1. **Probe Mechanics**: Given the exact same problem prompt (containing the active alphabet $\mathcal{A}$ and operands), the harness appends a probe trigger `[PROBE]` followed by an alphabet symbol $s_i \in \mathcal{A}$. The model must emit its immediate successor $s_{i+1} = \mathcal{A}[(\text{val}(s_i) + 1) \pmod B]$.
2. **The Retrieval-Execution Gap ($\Delta_{\text{exec}}$)**:
   $$\text{Gap} = \text{Accuracy}_{\text{probe}} - \text{EM}_{\text{task}}$$
   - If $\text{Accuracy}_{\text{probe}}$ is high ($> 95\%$) but $\text{EM}_{\text{task}}$ is low, the model successfully binds the rule parameters in-context, but fails in procedural execution or carry tracking.
   - If $\text{Accuracy}_{\text{probe}}$ is low, the failure is fundamentally a parameter binding or retrieval failure.

---

## 4. Fine-Grained Error Classification Taxonomy

Every failed problem instance ($EM_i = 0$) is inspected and classified into one of four mutually exclusive error types:

```
                           THE ERROR CLASSIFICATION TAXONOMY
                           
  ┌─────────────────────────┬────────────────────────────────────────────────────────┐
  │ Error Class             │ Operational Definition                                 │
  ├─────────────────────────┼────────────────────────────────────────────────────────┤
  │ 1. prior_intrusion      │ Emitted digit is WRONG under the in-context rule, but  │
  │                         │ CORRECT under the canonical prior ('0'..'9').           │
  │ 2. out_of_alphabet      │ Emitted token does NOT belong to the active alphabet A. │
  │ 3. procedural_error     │ Emitted token is in A, but represents an incorrect     │
  │                         │ arithmetic calculation.                                │
  │ 4. syntax_malformation  │ Emitted structural control token is misplaced or       │
  │                         │ corrupted (e.g. premature [FINISHED]).                 │
  └─────────────────────────┴────────────────────────────────────────────────────────┘
```

The **Prior Intrusion Rate**:
$$\mathcal{R}_{\text{prior}} = \frac{N_{\text{prior\_intrusion}}}{N_{\text{errors}}}$$
measures the degree to which failures under `canonical_permuted` are driven by active prior regression rather than generic degradation.

---

## 5. Per-Instance Telemetry & Regressor Logging

Aggregated summary metrics discard critical continuous covariates. The evaluation harness records **per-instance JSONL records** for every evaluated problem, enabling post-hoc regression analysis:

```json
{
  "instance_id": "cell2_b10_00042",
  "rule_family": "addition",
  "base": 10,
  "alphabet_condition": "canonical_permuted",
  "p_prior": 0.5,
  "len_a": 28,
  "len_b": 31,
  "prompt_token_length": 348,
  "rule_to_query_distance": 294,
  "entry_count_between": 59,
  "step_k": 14,
  "outcome_em": 0,
  "probe_accuracy": 1.0,
  "error_type": "prior_intrusion",
  "predicted_token": 53,
  "expected_token": 57,
  "prior_expected_token": 53
}
```

### Turning Experimental Side Effects into Continuous Regressors
Because different representation layouts and sequence lengths naturally vary the token distance between the rule statement (`[AXIOM]`) and the query position, logging `rule_to_query_distance` and `entry_count_between` transforms layout choices into continuous regressors. Researchers can fit logistic compliance curves:
$$P(\text{Compliance}) = \sigma(\beta_0 + \beta_1 \cdot \text{Distance} + \beta_2 \cdot p_{\text{prior}} + \beta_3 \cdot \text{Radix})$$
answering questions about rule decay over context distance without requiring new training runs.

---

## 6. Evaluation Budget & Hardware Optimization

1. **Evaluation Schedule**:
   - Checkpoint evaluations (every 5,000 steps): 100 instances per cell.
   - Terminal evaluation (step 100,000): 10,000 instances per cell.
2. **BF16 FlashAttention-2 Routing**:
   - All evaluation passes execute under `torch.amp.autocast(dtype=torch.bfloat16)` to force PyTorch SDPA into FlashAttention-2.
3. **Bounded Chunking**:
   - Evaluation runs process instances in bounded sub-chunks of 32 rows to guarantee zero `CUDA OOM` on long sequence lengths.
