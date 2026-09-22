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

In algorithmic rule-following, models frequently fail without researchers knowing *where* the breakdown occurred. Did the self-attention mechanism fail to bind the rule parameters, or did it fail during multi-step procedural calculation?

To resolve this ambiguity, models with tied copy heads (`C-Sequence`) can be evaluated on a **Single-Token Successor Binding Probe**:

```
                              THE BINDING PROBE PROTOCOL
                              
  Task Prompt:       [BOS] [AXIOM] s_0 s_1 s_2 ... [A] a [B] b ... [S]
                       │
  Binding Probe:     [BOS] [AXIOM] s_0 s_1 s_2 ... [A] a [B] b ... [PROBE] s_i
                       │
  Target Prediction: s_{i+1}   (Evaluated via tied copy head E^T)
```

### 3.1 Probe Mechanics & Architecture Scope
1. **Query Construction**: Given the exact same problem prompt (containing the active alphabet $\mathcal{A}$ and operands), the harness appends the probe trigger `[PROBE]` followed by an alphabet symbol $s_i \in \mathcal{A}$. The model must emit its immediate successor $s_{i+1} = \mathcal{A}[(\text{val}(s_i) + 1) \pmod B]$.
2. **Architectural Scope**: The probe relies on parameter-free attention retrieval via the tied head ($W_{\text{head}} = E^T$) and is natively supported for all `C-Sequence` architectures. For `C-Weight` architectures, the probe is undefined unless a dedicated linear probe projection head $W_{\text{probe}}$ is added.
3. **Probe Training Protocol & Methodological Confound**:
   - Because models are trained from scratch, an untrained model has never encountered `[PROBE]`. Therefore, probe queries must be included during training at an explicit **probe training rate** ($p_{\text{probe}} \in [0.0, 1.0]$, default: $p_{\text{probe}} = 0.05$).
   - A control condition with $p_{\text{probe}} = 0.0$ measures the baseline without probe supervision.
   - **Crucial Methodological Note**: Training on probe queries is an auxiliary multi-task objective. Exposing the model to explicit successor queries during training may itself act as an inductive scaffold that improves in-context binding and prior suppression. This auxiliary effect must be treated as a potential confound when evaluating hypotheses H7 and H8.

### 3.2 The Retrieval-Execution Gap ($\Delta_{\text{exec}}$)
Comparing probe accuracy against whole-sequence exact match is uninformative because sequence EM degrades with length by construction. Instead, the informative comparison evaluates probe accuracy against **per-step prediction accuracy ($\text{Acc}_{\text{step}}$) or per-digit output accuracy ($\text{Acc}_{\text{digit}}$)** on the exact same problem instances:
$$\Delta_{\text{exec}} = \text{Accuracy}_{\text{probe}} - \text{Acc}_{\text{step}}$$
- If $\text{Accuracy}_{\text{probe}} \approx 100\%$ while $\text{Acc}_{\text{step}}$ drops significantly on out-of-distribution instances, the failure is definitively isolated to procedural carry tracking, column alignment, or multi-step execution rather than an inability to bind the alphabet rule.

---

## 4. Fine-Grained Error Classification Taxonomy

Every failed problem instance ($EM_i = 0$) is inspected, and its **first failed output token** is classified into one of four mutually exclusive error types:

```
                           THE ERROR CLASSIFICATION TAXONOMY
                           
  ┌─────────────────────────┬────────────────────────────────────────────────────────┐
  │ Error Class             │ Operational Definition (Evaluated at First Error)      │
  ├─────────────────────────┼────────────────────────────────────────────────────────┤
  │ 1. prior_intrusion      │ Emitted digit is WRONG under the active bound rule,    │
  │                         │ but matches the prediction under canonical order.      │
  │                         │ (DEFINED STRICTLY UNDER canonical_permuted).           │
  │ 2. out_of_alphabet      │ Emitted token does NOT belong to the active alphabet A. │
  │ 3. procedural_error     │ Emitted token is in A, but represents an incorrect     │
  │                         │ arithmetic calculation.                                │
  │ 4. syntax_malformation  │ Emitted structural control token is misplaced or       │
  │                         │ corrupted (e.g. premature [FINISHED] or broken tag).   │
  └─────────────────────────┴────────────────────────────────────────────────────────┘
```

### 4.1 Chance-Adjusted Prior Intrusion Rate
Under the `canonical_permuted` condition, a model that makes a random wrong guess among the other $B - 1$ alphabet digits will accidentally match the canonical digit with probability:
$$\mathcal{R}_{\text{prior}}^{\text{chance}} = \frac{1}{B - 1}$$
For Base 10, random guessing produces an apparent prior intrusion rate of $1/9 \approx 11.1\%$; for Base 3, chance is $1/2 = 50.0\%$.

To isolate genuine prior regression from random error, the **Chance-Adjusted Prior Intrusion Rate** is reported:
$$\mathcal{R}_{\text{prior}}^{\text{adj}} = \frac{\mathcal{R}_{\text{prior}} - \frac{1}{B - 1}}{1 - \frac{1}{B - 1}}$$
where $\mathcal{R}_{\text{prior}} = N_{\text{prior\_intrusion}} / N_{\text{errors}}$. Under `symbols_novel` or `canonical_order`, `prior_intrusion` is undefined and logged as `null`.

---

## 5. Per-Instance Telemetry & Regressor Logging

The evaluation harness records **per-instance JSONL records** (`instances.jsonl`) for every evaluated problem, capturing fine-grained continuous covariates alongside outcomes:

```json
{
  "instance_id": "cell5_b10_00042",
  "rule_family": "addition",
  "base": 10,
  "alphabet_condition": "canonical_permuted",
  "p_prior": 0.5,
  "p_probe": 0.05,
  "len_a": 28,
  "len_b": 31,
  "prompt_token_length": 348,
  "rule_to_query_distance": 294,
  "entry_count_between": 59,
  "outcome_em": 0,
  "per_step_accuracy": 0.742,
  "probe_accuracy": 1.0,
  "first_error_step_k": 14,
  "first_error_type": "prior_intrusion",
  "first_predicted_token": 53,
  "first_expected_token": 57,
  "canonical_expected_token": 53
}
```

*Note on Distance Regressors*:
- For representation layouts with an explicit axiom preamble (`sinusoidal_pe` through `addressed_kv`), `rule_to_query_distance` records the exact token count from the end of the axiom header to the query trigger.
- For `zero_preamble_kv`, which contains no axiom header, `rule_to_query_distance` is structurally undefined and recorded as `null`.

### Turning Experimental Side Effects into Continuous Regressors
Because different representation layouts and sequence lengths naturally vary the token distance between the rule statement and the query position, logging `rule_to_query_distance` and `entry_count_between` transforms layout choices into continuous regressors for post-hoc logistic regression:
$$P(\text{Compliance}) = \sigma(\beta_0 + \beta_1 \cdot \text{Distance} + \beta_2 \cdot p_{\text{prior}} + \beta_3 \cdot \text{Radix})$$

## 6. Evaluation Budget & Hardware Optimization

1. **Evaluation Schedule**:
   - Checkpoint evaluations (every 5,000 steps): 100 instances per cell.
   - Terminal evaluation (step 100,000): 10,000 instances per cell.
2. **BF16 FlashAttention-2 Routing**:
   - All evaluation passes execute under `torch.amp.autocast(dtype=torch.bfloat16)` to force PyTorch SDPA into FlashAttention-2.
3. **Bounded Chunking**:
   - Evaluation runs process instances in bounded sub-chunks of 32 rows to guarantee zero `CUDA OOM` on long sequence lengths.
