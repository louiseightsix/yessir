# Evaluation, Metrics, Binding Probe and Telemetry

**Specification Document 05**

---

## 1. Zero-Shot Protocol

* **Frozen weights**: no parameter updates.
* **No demonstrations**: prompts contain only the axiom and the query problem.
* **Fixed evaluation seeds**: evaluation data is procedurally generated from seeds disjoint from training.
* **Free-running vs. teacher-forced**: free-running metrics let the model consume its own outputs through the harness; teacher-forced metrics pose each step from the reference state, isolating single-step fidelity from error accumulation.

---

## 2. Evaluation Cells

```
  Cell 1  In-domain               B in train bases,   L 1..20
  Cell 2  Length extrapolation    B in train bases,   L 21..40   (2x training length)
  Cell 3  Unseen base, interp.    B = 3,              L 1..20
  Cell 4  Unseen base, extrap.    B = 12,             L 1..20
  Cell 5  Prior conflict          B in train bases,   canonical_permuted
  Cell 6  Combined stress         B = 12,             L 21..40,  canonical_permuted
```

Each cell is evaluated on the alphabet regimes of [`03`](03_representations.md) §2 (`--eval-axioms`: random, canonical, permuted). Unseen bases at extrapolation length (Base-3 and Base-12 at $L = 21..40$) are also reported.

Metric keys in `metrics.jsonl` follow the cell: `id_*` (cell 1), `extrap_len_*` (cell 2), `ood_base3_*` (cell 3), `ood_base12_*` (cell 4), `extrap_len_base3_*` / `extrap_len_base12_*`; canonical-axiom variants are prefixed `canon_`, and newer runs also write explicit `random_` prefixes.

---

## 3. Metrics

1. **Exact match (EM, `*_em`)**: the harness finished and the full output equals the reference $OUT$ (same digits and length, hence correct halting). Primary metric.
2. **Digit accuracy (`*_digit_acc`)**: fraction of sum digits correct, over reference columns.
3. **Carry-chain accuracy (`*_carry_acc`)**: digit accuracy on columns with non-zero carry in or out.
4. **Step-transition accuracy (`*_step_acc`)**: teacher-forced; all of $o_k$, $c_{k+1}$, $\text{FLAG}_k$ correct. For `AR`, the reference trace is teacher-forced up to column $k$.
5. **Halting accuracy (`*_halt_acc`)**: teacher-forced $\text{FLAG}_k$ correct; also split into `_terminal` ($k = K$) and `_nonterminal`.
6. **Failure counts**: `*_max_steps_exceeded`, `*_max_tokens_exceeded`, `*_malformed`, and `*_out_of_axiom` (emitted numerals not in $\mathcal{A}$).

One `metrics.jsonl` line is written per evaluation, e.g.:
```json
{"step": 5000, "loss": 0.042, "id_em": 0.992, "ood_base3_em": 0.945, "ood_base12_em": 0.910, "extrap_len_em": 0.885, "ood_base12_step_acc": 0.991, "ood_base12_halt_acc": 0.998}
```
The final evaluation is also written to `summary.json`.

---

## 4. Binding Probe: Reading the Axiom vs. Using It

When a model fails, did it fail to read the axiom, or fail to execute the procedure? The **successor probe** asks a one-token question from the same prompt:

```
  Task prompt:   [BOS] [AXIOM] s_0 s_1 s_2 ... [A] a [B] b ... [S]
  Probe prompt:  [BOS] [AXIOM] s_0 s_1 s_2 ... [A] a [B] b ... [PROBE] s_i
  Target:        s_{i+1} = A[(val(s_i) + 1) mod B]    (via tied head E^T)
```

### 4.1 Mechanics and scope
1. **Construction**: append `[PROBE]` and a symbol $s_i \in \mathcal{A}$ to the problem prompt; the model must emit its successor.
2. **Scope**: native to all `C-Sequence` models (tied head, no new parameters). For `C-Weight` models the probe is undefined unless a dedicated probe head is added.
3. **Training**: an untrained model has never seen `[PROBE]`, so probe queries are mixed into training at rate $p_{\text{probe}}$ (`--p-probe`; default 0, suggested 0.05). $p_{\text{probe}} = 0$ is the control.
4. **Confound**: probe training is an auxiliary task and may itself improve binding and prior suppression. Treat this as a confound for H7 and H8.

### 4.2 Retrieval-execution gap
Whole-sequence EM falls with length by construction, so compare the probe with **per-step** or **per-digit** accuracy on the same instances:
$$\Delta_{\text{exec}} = \text{Acc}_{\text{probe}} - \text{Acc}_{\text{step}}$$
Probe accuracy near 100% while step accuracy drops out of distribution means the failure is in execution, not binding.

---

## 5. Error Taxonomy

The **first failed output token** of each failed instance is classified:

| Class | Definition at first error |
| :--- | :--- |
| `prior_intrusion` | Wrong under the axiom, but equals the canonical-order answer. Defined only under `canonical_permuted`. |
| `out_of_alphabet` | Emitted token is not in $\mathcal{A}$. |
| `procedural_error` | Token is in $\mathcal{A}$ but arithmetically wrong. |
| `syntax_malformation` | Misplaced or corrupted control token (e.g. premature `[FINISHED]`). |

### 5.1 Chance-adjusted prior intrusion
A random wrong digit matches the canonical answer with probability $1/(B-1)$ (11.1% in Base 10, 50% in Base 3). Report
$$\mathcal{R}_{\text{prior}}^{\text{adj}} = \frac{\mathcal{R}_{\text{prior}} - \frac{1}{B-1}}{1 - \frac{1}{B-1}}, \qquad \mathcal{R}_{\text{prior}} = N_{\text{prior\_intrusion}} / N_{\text{errors}}$$
Under `symbols_random` and `canonical_order`, prior intrusion is undefined and logged as `null`.

---

## 6. Per-Instance Telemetry

Each evaluated problem is appended to `instances.jsonl`:

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

* `rule_to_query_distance`: tokens from the end of the axiom block to the query trigger; `null` for `zero_preamble_kv` (no axiom).
* Because layouts and lengths vary this distance naturally, it can be used as a continuous regressor, e.g. $P(\text{correct}) = \sigma(\beta_0 + \beta_1 \cdot \text{distance} + \beta_2 \cdot p_{\text{prior}} + \beta_3 \cdot B)$.

---

## 7. Evaluation Budget

* **During training**: every `--eval-interval` steps (default 5,000) on `--eval-problems` per cell (default 100).
* **Final**: `--final-eval-problems` per cell (default 250 for runs of ≥ 50k steps). The success criteria of [`01`](01_research_goal.md) §5 call for 10,000 per cell, to be run on frozen checkpoints of candidate models.
* **Precision**: evaluation runs under bfloat16 autocast so SDPA can use FlashAttention.
* **Chunking**: evaluation processes instances in bounded chunks (32 rows on long-sequence tiers) to avoid running out of GPU memory at extrapolation lengths.
