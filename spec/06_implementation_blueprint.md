# PyTorch Implementation Blueprint, Generation Contracts, and Engineering Guidelines

**Specification Document 06**  
**TransformAdder Project**

---

## 1. Universal Transformer Architecture

All primary models in the TransformAdder project share an identical, tightly controlled architectural footprint to ensure that empirical differences stem strictly from representational and topological inductive biases rather than parameter capacity.

```
                           UNIVERSAL TRANSFORMER ARCHITECTURE
                           
                 Token IDs -> Input Embedding Matrix E in R^(256 x 256)
                                      │
                 [Optional: Role Table E_role + Continuous PE]
                                      │
                         Pre-LN Transformer Block x 6
                         ├── RMSNorm / LayerNorm
                         ├── Multi-Head Causal Self-Attention (4 Heads, d_k = 64)
                         │   └── PyTorch scaled_dot_product_attention (FlashAttention-2)
                         ├── Residual Connection
                         ├── RMSNorm / LayerNorm
                         ├── Feed-Forward Network (SwiGLU / GELU, d_ff = 1024)
                         └── Residual Connection
                                      │
                                 RMSNorm / LayerNorm
                                      │
                ┌─────────────────────┴─────────────────────┐
                ▼                                           ▼
      Weight-Space Readout (C-Weight)             Sequence-Space Readout (C-Sequence)
      ├── Linear Head W_sum   (256 -> 256)        └── Parameter-Free Tied Head (W = E^T)
      ├── Linear Head W_carry (256 -> 256)            Single projection (256 -> 256)
      └── Linear Head W_flag  (256 -> 2)              Conditioned on [S], [C], [F] triggers
```

### 1.1 Hyperparameter Specifications
* **Layers ($L$)**: 6 pre-layer-normalised transformer blocks.
* **Model Dimension ($d_{\text{model}}$)**: 256.
* **Attention Heads ($n_{\text{heads}}$)**: 4 (head dimension $d_{\text{head}} = 64$).
* **Feed-Forward Dimension ($d_{\text{ff}}$)**: 1024 (expansion factor 4.0).
* **Activation Function**: SwiGLU (gated linear unit with SiLU) or standard GELU.
* **Attention Mask**: Strictly lower-triangular causal attention mask ($M_{ij} = -\infty$ for $j > i$).
* **Dropout**: 0.0 (deterministic inference; pure algorithmic reasoning).
* **Parameter Count**: ~3.15M parameters under tied sequence heads (`C-Sequence`); ~3.28M under multi-head projections (`C-Weight`).

---

## 2. Deterministic Procedural Arithmetic Generator

To ensure absolute reproducibility with zero disk I/O, all training and evaluation samples are generated procedurally from the oracle reference implementation (`core/reference.py`).

### 2.1 The Data Generation Contract
Data generation is a deterministic pure function of the random seed:
$$\text{Sample} = \text{Generator}(\text{seed}, \text{step}, \text{tier}, \text{curriculum})$$
1. **Radix Selection**: Base $B$ is sampled uniformly from the active regime (e.g. $\{2, 4, 8, 10\}$ for multimodal training; $B=3$ or $12$ for evaluation).
2. **Alphabet Sampling**: The alphabet $\mathcal{A} = [s_0, \dots, s_{B-1}]$ is sampled from the 240 candidate byte pool $\Sigma$ (or assigned canonical ASCII digits `0`..`9`).
3. **Operand Length Sampling**:
   - Under `D-Standard`: $L_a, L_b \sim \mathcal{U}(L_{\min}, L_{\max})$ with $L_{\min} = 1, L_{\max} = 20$.
   - Under `D-FullSpan`: $L_a, L_b \sim \mathcal{U}(\max(L_{\min}, B), \max(L_{\max}, \max(L_{\min}, B)))$.
4. **Digit Uniformity**: Operand digits are sampled uniformly from $[0, B-1)$.

### 2.2 Carry Chain Injection ($p_{\text{chain}} = 0.5$)
Under purely uniform random digits, the probability that a carry propagates across $L$ consecutive columns decays geometrically as $(1/B)^L$. In Base 10, a carry chain of length 5 occurs with probability $10^{-5}$, rendering deep carry propagation virtually absent from random data.

To force the model to acquire a robust carry-lookahead circuit:
* With probability $p_{\text{chain}} = 0.5$, an instance is modified to contain an artificial carry cascade:
  1. A generating column $i$ is chosen where $a_i + b_i \ge B$.
  2. A chain length $L \in [1, \min(|a|, |b|) - 1 - i]$ is selected.
  3. For all columns $j \in [i+1, i+L]$, digits are constrained such that $a_j + b_j = B - 1$.
* This guarantees rich exposure to multi-column carry propagation cascades.

### 2.3 Terminal Step Oversampling ($p_{\text{terminal}} = 0.2$)
In harnessed state machine training (`SM-MultiHead`, `SM-Fork`, `SM-Seq`, `SM-SeqRev`), an operand of length $L$ produces $L$ continuation steps ($\text{FLAG} = \texttt{FLAG\_CONTINUE}$) and exactly 1 halting step ($\text{FLAG} = \texttt{FLAG\_FINISHED}$). Randomly sampled step transitions would exhibit a severe class imbalance ($10:1$ to $20:1$).

* To counteract this imbalance, the step sampler forces the terminal step $k = K$ with probability $p_{\text{terminal}} = 0.2$, stabilising the convergence of the halting prediction circuit.

---

## 3. Training Protocol & Optimization

1. **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-8}$, weight decay $\lambda = 0.01$).
2. **Learning Rate Schedule**:
   - Peak learning rate: $\eta_{\max} = 3 \times 10^{-4}$.
   - Linear warmup: 2,000 steps from $\eta_{\text{init}} = 10^{-7}$.
   - Cosine decay: Decays from step 2,000 to 100,000 down to $\eta_{\min} = 3 \times 10^{-5}$.
3. **Effective Batch Size & Gradient Accumulation**:
   - Canonical effective batch size: 256 problem instances per optimization step.
   - For memory-intensive configurations (e.g. `SM-Fork` under long sequence addressing), gradient accumulation is dynamically configured:
     $$\text{Batch Size per Sub-Step} = 32, \quad \text{Gradient Accumulation Steps} = 8$$
     $$\text{Effective Batch Size} = 32 \times 8 = 256$$
4. **Gradient Clipping**: Norm clipping at $\|\mathbf{g}\|_2 \le 1.0$ to prevent optimization instability during early warmup.

---

## 4. Hardware Optimization & Memory Engineering

Long-sequence addressing prompts (e.g. `addressed_kv` and `zero_preamble_kv`) reach sequence lengths of up to ~920 tokens, creating high memory pressure on mid-tier accelerators (e.g. NVIDIA L4 24GB GPUs). The following engineering guidelines are mandatory:

1. **FlashAttention-2 Kernel Routing**:
   - All attention operations must use `torch.nn.functional.scaled_dot_product_attention`.
   - In both training and evaluation, operations must be wrapped in `torch.amp.autocast(dtype=torch.bfloat16)` to guarantee that PyTorch routes execution to the FlashAttention-2 or memory-efficient attention CUDA kernel rather than falling back to standard fp32 cubic attention.
2. **PyTorch Expandable Memory Segments**:
   - The CUDA memory allocator must be initialized with:
     ```bash
     export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     ```
     This prevents virtual memory fragmentation from triggering premature `CUBLAS_STATUS_ALLOC_FAILED` or `CUDA OOM` errors during large matrix multiplications.
3. **Evaluation Chunking**:
   - Evaluation metrics under `teacher_forced_metrics` must process evaluation sets in bounded sub-chunks of 32 rows, preventing memory allocation spikes while evaluating out-of-distribution double-length instances.

---

## 5. Checkpoint Schema & Architectural Compatibility

All saved checkpoints store their configuration metadata using the orthogonal 5-axis schema:

```json
{
  "agency": "harnessed",
  "topology": "carry_first",
  "channel": "sequence",
  "representation": "addressed_kv",
  "curriculum": "standard",
  "n_layers": 6,
  "d_model": 256,
  "n_heads": 4,
  "d_ff": 1024,
  "step": 100000
}
```

### 5.1 Backward-Compatible Checkpoint Resolver
To ensure that existing experimental checkpoints produced during earlier phases can be evaluated without modification, model loaders must implement a compact, self-contained configuration resolver:

```python
def resolve_legacy_config(cfg: dict) -> dict:
    """Maps legacy paradigm/tier metadata into canonical orthogonal coordinates."""
    if "agency" in cfg:
        return cfg  # Already canonical
    
    paradigm = cfg.get("paradigm", "sm_mh")
    tier = int(cfg.get("tier", 1))
    
    agency = "autonomous" if paradigm == "ar" else "harnessed"
    topology = "carry_first" if paradigm == "sm_seq_rev" else ("factored" if paradigm in ("sm_mh", "sm_fork") else "sum_first")
    channel = "weight" if paradigm == "sm_mh" else "sequence"
    
    repr_map = {
        1: "sinusoidal_pe", 2: "semantic_nope", 3: "generic_nope",
        4: "role_embed_nope", 5: "semantic_origin_nope",
        6: "addressed_kv", 7: "zero_preamble_kv"
    }
    representation = repr_map.get(tier, "sinusoidal_pe")
    curriculum = "full_span" if tier == 7 else "standard"
    
    return {**cfg, "agency": agency, "topology": topology, "channel": channel,
            "representation": representation, "curriculum": curriculum}
```
This isolates historical compatibility entirely to checkpoint loading, keeping all core training and inference pathways pure and orthogonal.
