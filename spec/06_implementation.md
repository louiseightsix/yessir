# Engineering Implementation & Technical Blueprints

This document specifies the technical implementation blueprints, tensor shapes, vector layouts, software architecture, and runtime execution harness code for **TransformAdder**. It operationalises the mechanisms and variants defined across [`02_mechanics.md`](02_mechanics.md), [`03_levers.md`](03_levers.md), and [`04_variants.md`](04_variants.md). Where this document and `02`/`03` disagree, `02`/`03` are authoritative and this document is in error.

---

## 1. Codebase Architecture & Directory Layout

To ensure auditability, minimal dependency rot, and reproducibility, the implementation adheres to two structural invariants:
1. **Zero External Framework Dependencies**: Built strictly with standard Python 3.10+ and core PyTorch (`torch`, `torch.nn`, `torch.nn.functional`). No HuggingFace, Wandb, or external orchestrators. In particular there is no library `generate()`; greedy decoding is implemented in-repo (Section 5).
2. **Decoupled Common Core & Explicit Experiment Runners**: Shared mathematical components, layers, and data generators reside in a modular core library (`transformadder/core/`, `transformadder/harnesses/`, `transformadder/utils/`). Each variant in the model matrix is instantiated in its own standalone, self-contained experiment runner (`experiments/`) that can be executed, inspected, and verified independently.

**Reproducibility Contract**: Data generation is a pure function of the seed (bitwise identical on any platform). Training is run with `torch.use_deterministic_algorithms(True, warn_only=True)` (a few CUDA kernels have no deterministic implementation) and fixed seeds, giving run-to-run reproducibility on a fixed hardware/software stack. Bitwise identity *across* GPU models, driver versions, or PyTorch versions is not guaranteed (attention and matmul kernels differ) and is not claimed.

```text
<repository root>/
├── run.sh                   # Environment bootstrap; delegates to scripts/configure_experiment.py
├── scripts/
│   └── configure_experiment.py  # Interactive / scripted configurator; generates experiment bundles
├── transformadder/          # importable core library
│   ├── cli.py               # `python -m transformadder.cli`: flag-driven runner and Vertex AI entrypoint
│   ├── core/
│   │   ├── constants.py     # Byte constants (mirrors 03_levers.md Section 2.1), role ids, dimensions
│   │   ├── blocks.py        # RMSNorm (optional gain), SwiGLU, MultiHeadAttention via SDPA, prefix_lm_mask
│   │   ├── embedding.py     # Fixed one-hot basis + unit-norm 2D Split PE / NoPE padding
│   │   ├── model.py         # Modular Transformer trunk, initialisation, batch-of-one helpers
│   │   ├── heads.py         # Multi-Head projections vs Tied Sliced Identity Head
│   │   ├── reference.py     # Reference adder (ground-truth oracle)
│   │   ├── sequences.py     # Seq container; SM / AR prompt builders for Tiers 1-5; collator
│   │   ├── data.py          # Deterministic procedural generator, sample encoders, batch builders
│   │   └── training.py      # Seeding, AdamW parameter groups, warm-up + cosine schedule
│   ├── harnesses/
│   │   ├── state_machine.py # run_state_machine_mh, run_state_machine_seq
│   │   └── autoregressive.py# run_autoregressive_trace, parse_trace
│   └── utils/
│       ├── evaluation.py    # Free-running and teacher-forced metrics over the evaluation cells
│       ├── runner.py        # Shared CLI, training loop and evaluation cadence for all variants
│       ├── logging.py       # Local JSONL logger & console formatter
│       └── stepper.py       # Shared interactive educational stepper (--interactive)
├── experiments/             # Generated, untracked: one bundle per run, experiments/<name>/{run.py, run.sh, submit_vertex.sh, config.json}
└── tests/
    ├── test_mechanics.py    # Reference adder, generator, prompt layouts, encoders, parser, vocab-vs-spec
    ├── test_shapes.py       # Tensor shapes, unit-norm embedding, padding invariance, masks, training
    ├── test_harnesses.py    # End-to-end harness tests against a scripted oracle model
    └── test_cli.py          # CLI variant builder and experiment-bundle generation
```

Each generated `run.py` declares only its `Variant` (paradigm, tier, depth, FFN, regime, seed, and run-length settings) and delegates to `utils/runner.py`, so that the training loop, optimiser settings (Section 7.3) and evaluation are identical across the model matrix by construction.

Bundle names follow the variant tags of [`04_variants.md`](04_variants.md), suffixed with the training regime and seed (e.g. `SM-MH-T1-L6-SwiGLU-Causal-multimodal-seed42`).

---

## 2. Tensor Shapes, Coordinate Grids & Memory Contracts

### 2.1 Universal Vocabulary & Byte Identity Space
The vocabulary ($|\mathcal{V}| = 256$), the numeral pool $\Sigma$ (`0x00`–`0xEF`), and all control byte assignments are defined in [`03_levers.md`](03_levers.md) Section 2.1, which is the single source of truth. `core/constants.py` mirrors that table as constants (`BOS`, `PAD`, `START`, `TOKEN_S`, `TOKEN_C`, `TOKEN_FLAG`, `FLAG_CONTINUE`, `FLAG_FINISHED`, `DELIM_*`) and a unit test asserts the two agree. The table is deliberately **not** restated here.

### 2.2 Canonical Subspace Partitioning ($d_{\text{model}} = 512$)
Every input token is mapped to a vector $\mathbf{x}_t \in \mathbb{R}^{512}$ split into two contiguous 256-dimensional blocks:
$$\mathbf{x}_t = [ \, \mathbf{e}_{b_t} \;\parallel\; \mathbf{p}_t \, ] \in \mathbb{R}^{512}$$

* **Symbol Identity Subspace ($\text{dims } 0 \dots 255$)**:
  * Fixed, non-trainable one-hot vector $\mathbf{e}_{b_t} \in \{0, 1\}^{256}$ corresponding to the byte integer $b_t \in \{0, \dots, 255\}$.
  * Constructed on the fly via `F.one_hot(token_ids, num_classes=256).to(dtype)`.
* **Positional & Algorithmic State Subspace ($\text{dims } 256 \dots 511$)**:
  * **Tier 1 (2D Split PE)**:
    $$\mathbf{p}_t = [ \, e_{\text{role}}(r_t) \;\parallel\; \tfrac{1}{\sqrt{96}}\, e_{\text{sig}}(p_t) \, ] \in \mathbb{R}^{256}$$
    * $e_{\text{role}}(r_t) \in \mathbb{R}^{64}$: Learned embedding lookup table of shape $(8, 64)$, rows initialised to unit norm, for the roles $r \in \{0, \dots, 7\}$ defined in [`03_levers.md`](03_levers.md) Section 1.1 (`ROLE_AXIOM`, `ROLE_OPERAND_A`, `ROLE_OPERAND_B`, `ROLE_CARRY`, `ROLE_OUT`, `ROLE_FLAG`, `ROLE_START`, `ROLE_CTRL`).
    * $e_{\text{sig}}(p_t) \in \mathbb{R}^{192}$: Uncapped sinusoidal functional encoding for significance coordinate $p \ge 0$, scaled by $1/\sqrt{96}$ to unit norm:
      $$e_{\text{sig}}(p)_{2i} = \sin(p \cdot \omega_i), \quad e_{\text{sig}}(p)_{2i+1} = \cos(p \cdot \omega_i), \quad \text{where } \omega_i = 10000^{-2i/192}, \; i \in \{0, \dots, 95\}$$
  * **Tier 4 (Role only)**:
    $$\mathbf{p}_t = [ \, e_{\text{role}}(r_t) \;\parallel\; \mathbf{0}_{192} \, ]$$
  * **Tiers 2, 3 and 5 (NoPE)**:
    $$\mathbf{p}_t = \mathbf{0}_{256}$$
    Implemented via zero padding `torch.nn.functional.pad(one_hot, (0, 256))`.

### 2.3 Sequence Container & Model Interface
All prompt builders return a `Seq`, which carries the Tier-1 coordinates alongside the tokens so that harness code is tier-agnostic:

```python
from dataclasses import dataclass

@dataclass
class Seq:
    tokens: list[int]
    roles: list[int] | None = None   # Tier 1 only; None under NoPE
    sigs: list[int] | None = None    # Tier 1 only; None under NoPE

    def push(self, token: int, role: int, sig: int) -> "Seq":
        """Returns a new Seq with one token appended (coordinates ignored under NoPE)."""
        if self.roles is None:
            return Seq(self.tokens + [token])
        return Seq(self.tokens + [token], self.roles + [role], self.sigs + [sig])
```

Model entry points (batched tensors; `role_ids`/`sig_ids` are `None` for Tiers 2–3):

| Call | Returns |
| :--- | :--- |
| `model(token_ids, role_ids, sig_ids)` | Tied-slice logits `[B, T, 256]` (`SM-Seq`, `SM-SeqRev`, `SM-Fork`, `AR`) |
| `model.forward_sm_mh(token_ids, role_ids, sig_ids, lengths)` | `(logits_sum [B,256], logits_carry [B,256], logits_flag [B,2])`, read at index `lengths - 1` |
| `model.next_logits(seq)` | Convenience for harnesses: batch-of-one forward, logits at the final position, `[256]` |

### 2.4 Batching & Padding Contract
* Batches are **right-padded** with `PAD` (`ROLE_CTRL`, $p = 0$ in Tier 1) to the longest sequence in the batch.
* Under a causal mask, no real token can attend to a later `PAD`, so right-padding requires **no padding mask** and cannot perturb NoPE position counting. Left-padding is forbidden (it would displace `[BOS]` from index 0 and create fully-masked attention rows).
* `SM-MH` reads the hidden state at the true tail, index `lengths - 1` (the `[S]` query), **never** at `h[:, -1]`, which in a padded batch is a `PAD` position.
* Labels at `PAD` positions are `-100`.

---

## 3. Core PyTorch Modules & Technology Stack

### 3.1 Pre-LN RMSNorm
Implemented following standard Pre-LN formulation with a small epsilon. The learnable gain is optional: transformer blocks use `gain=True`; the final norm feeding the tied copy slice uses `gain=False` ($\text{RMSNorm}_0$), keeping the output path parameter-free.
$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \mathbf{g}, \qquad \text{RMSNorm}_0: \; \mathbf{g} \equiv \mathbf{1} \text{ (fixed)}$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, gain: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if gain else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x if self.weight is None else x * self.weight
```

### 3.2 SwiGLU Gated Feed-Forward Network
Gated bilinear feed-forward layer implementing carry and modulo non-linearities:
$$\text{FFN}(\mathbf{x}) = \left(\text{Swish}(\mathbf{x} W_{\text{gate}}) \odot (\mathbf{x} W_{\text{up}})\right) W_{\text{down}}$$

```python
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ffn: int = 1376):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

### 3.3 Multi-Head Attention via PyTorch SDPA
Uses `F.scaled_dot_product_attention` for fast, memory-efficient self-attention without external dependencies.

**Mask contract**: with `attn_mask=None` the layer is strictly causal, which is sufficient for every right-padded batch (Section 2.4) and is the default for all tiers. A supplied `attn_mask` **replaces** the causal mask entirely and must therefore itself encode causality; it is used only for the Tier-1 Prefix-LM ablation ([`03_levers.md`](03_levers.md) Section 6), built by `prefix_lm_mask`.

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # attn_mask: optional bool [B, 1, T, T], True = may attend. Must be a COMPLETE
        # mask (including causal structure); None selects the built-in causal mask.
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None)
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


def prefix_lm_mask(prefix_lens: torch.Tensor, T: int) -> torch.Tensor:
    """Complete Prefix-LM mask [B, 1, T, T]: causal everywhere, plus full
    bidirectional visibility inside each sequence's static prefix ([BOS], A, a, b).
    Tier 1 only."""
    idx = torch.arange(T, device=prefix_lens.device)
    causal = idx[None, :, None] >= idx[None, None, :]                 # [1, T, T]
    in_prefix = idx[None, :] < prefix_lens[:, None]                   # [B, T]
    bidir = in_prefix[:, :, None] & in_prefix[:, None, :]             # [B, T, T]
    return (causal | bidir)[:, None, :, :]
```

### 3.4 Output Head Architectures

#### Head Topology 1: Tied Identity Copy Head (Used in `SM-Seq`, `SM-SeqRev`, `SM-Fork` and `AR`)
* Slices the residual stream directly onto the first 256 dimensions after the **gain-free** final norm: $\mathbf{y}_t = \text{RMSNorm}_0(\mathbf{h}_t)[:256]$ ([`03_levers.md`](03_levers.md) Section 4.1).
* Contains **zero trainable weights**:

```python
class TiedIdentityHead(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.norm = RMSNorm(d_model, gain=False)   # no parameters

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, 512] -> normalised -> sliced to [B, T, 256]
        return self.norm(h)[:, :, :256]
```

#### Head Topology 2: Multi-Head Parallel Projections (Used in `SM-MH`)
* Emits sum, carry, and flag logits in parallel from the hidden state at the `[S]` query position (the true sequence tail, Section 2.4):

```python
class MultiHeadOutput(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.head_sum = nn.Linear(d_model, 256, bias=False)
        self.head_carry = nn.Linear(d_model, 256, bias=False)
        self.head_flag = nn.Linear(d_model, 2, bias=False)   # index 0: CONTINUE, 1: FINISHED

    def forward(self, h: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # h: [B, T, 512]; lengths: [B] true (unpadded) sequence lengths
        batch = torch.arange(h.size(0), device=h.device)
        h_tail = self.norm(h[batch, lengths - 1, :])   # [B, 512], the [S] position
        logits_sum = self.head_sum(h_tail)     # [B, 256]
        logits_carry = self.head_carry(h_tail) # [B, 256]
        logits_flag = self.head_flag(h_tail)   # [B, 2]
        return logits_sum, logits_carry, logits_flag
```

---

## 4. Deterministic Procedural Data Pipeline

### 4.1 Seeded Procedural Generation Contract
Dataset instances are generated on the fly, without disk I/O, by a deterministic generator seeded by a global seed. The generator (unlike the harness) is free to perform arithmetic: it is the oracle side of the experiment.

* **Axiom**: $B$ distinct bytes sampled uniformly from $\Sigma$, in random order (`canonical=True` instead yields the fixed ASCII axiom `0`–`9`, `A`, `B` truncated to $B$, used only for the canonical evaluation set of [`01_research_goal.md`](01_research_goal.md) Section 4.1).
* **Operands**: lengths independent and uniform on `[min_len, max_len]`; digits uniform over the axiom. Most-significant zeros are permitted (operands need not be canonical numerals).
* **Carry-Chain Enrichment**: with probability `p_chain` (default 0.25), a carry chain is constructed deliberately: one column is forced to *generate* a carry ($a_i + b_i \ge B$) and the following $L$ columns are forced to *propagate* it ($a_j + b_j = B - 1$). Uniformly random digits produce a propagating column with probability only $1/B$, so long chains would otherwise be essentially absent in Base-10 and Base-12.
* **Regime**: `bases` must be passed explicitly (`[2, 10]` bimodal or `[2, 4, 8, 10]` multimodal); there is no default, to prevent silently training the wrong regime.

```python
import random

CANONICAL_AXIOM = list(b"0123456789AB")

class ProceduralArithmeticGenerator:
    def __init__(self, bases: list[int], min_len: int = 1, max_len: int = 20,
                 p_chain: float = 0.25, p_terminal: float = 0.2, canonical: bool = False, seed: int = 42):
        self.bases = list(bases)
        self.min_len = min_len
        self.max_len = max_len
        self.p_chain = p_chain
        self.p_terminal = p_terminal       # SM terminal-step sampling (Section 4.3)
        self.canonical = canonical
        self.rng = random.Random(seed)
        self.sigma_pool = list(range(0x00, 0xF0)) # 240 candidate symbols

    def sample_instance(self) -> dict:
        base = self.rng.choice(self.bases)
        axiom = CANONICAL_AXIOM[:base] if self.canonical else self.rng.sample(self.sigma_pool, base)
        len_a = self.rng.randint(self.min_len, self.max_len)
        len_b = self.rng.randint(self.min_len, self.max_len)
        a_vals = [self.rng.randrange(base) for _ in range(len_a)]
        b_vals = [self.rng.randrange(base) for _ in range(len_b)]
        if self.rng.random() < self.p_chain:
            self._inject_carry_chain(a_vals, b_vals, base)
        return {"base": base, "axiom": axiom,
                "a": [axiom[v] for v in a_vals], "b": [axiom[v] for v in b_vals]}

    def _inject_carry_chain(self, a_vals: list[int], b_vals: list[int], base: int) -> None:
        n = min(len(a_vals), len(b_vals))
        if n < 2:
            return
        L = self.rng.randint(1, n - 1)                 # number of propagating columns
        i = self.rng.randint(0, n - 1 - L)             # generating column
        a_vals[i] = self.rng.randint(1, base - 1)
        b_vals[i] = self.rng.randint(base - a_vals[i], base - 1)   # a_i + b_i >= base
        for j in range(i + 1, i + 1 + L):
            a_vals[j] = self.rng.randrange(base)
            b_vals[j] = base - 1 - a_vals[j]           # a_j + b_j == base - 1
```

### 4.2 Reference Adder (Ground-Truth Oracle)
`core/reference.py` is the single source of ground truth for training targets, evaluation scoring, and the stepper's explanatory output. It implements exactly the transition and halting invariant of [`02_mechanics.md`](02_mechanics.md) Sections 2 and 3.4:

```python
def reference_add(axiom: list[int], a: list[int], b: list[int]) -> list[dict]:
    """Returns the K+1 ground-truth column transitions for a + b under `axiom`."""
    val = {s: i for i, s in enumerate(axiom)}
    base, n = len(axiom), max(len(a), len(b))
    steps, carry, k = [], 0, 0
    while True:
        v_a = val[a[k]] if k < len(a) else 0
        v_b = val[b[k]] if k < len(b) else 0
        total = v_a + v_b + carry
        o, c_next = total % base, total // base
        finished = (k >= n - 1) and (c_next == 0)
        steps.append({
            "k": k, "c_in": axiom[carry], "o": axiom[o], "c_out": axiom[c_next],
            "flag": FLAG_FINISHED if finished else FLAG_CONTINUE,
        })
        if finished:
            return steps
        carry, k = c_next, k + 1
```

The number of transitions is $K + 1$, where $K + 1 = \max(|a|,|b|)$ if the final operand column produces no carry and $\max(|a|,|b|) + 1$ otherwise.

### 4.3 Step-Transition Samples (`SM-MH`, `SM-Seq`, `SM-SeqRev` and `SM-Fork`)
For state machines, each training sample is **one column transition** of one instance. Per sample, a fresh instance is drawn and a single step $k$ is selected: the terminal step $k = K$ with probability `p_terminal` (default 0.2), otherwise uniformly from $\{0, \dots, K\}$. Sampling one step per instance decorrelates the batch; up-weighting the terminal step counteracts the $\approx 1{:}20$ imbalance between `FLAG_CONTINUE` and `FLAG_FINISHED`. The prompt is built from the oracle state $(c_k, OUT_k)$ (teacher forcing).

* **`SM-MH`**: input is the base prompt `[BOS] A a b c_k OUT_k [S]`; targets are the triple $(o_k, c_{k+1}, \text{FLAG}_k)$ with `FLAG` encoded as class index 0 (`CONTINUE`) / 1 (`FINISHED`).
* **`SM-Seq`**: input is the full Pass-3 sequence plus the emitted flag,
  $$\texttt{[BOS]} \; \mathcal{A} \; a \; b \; c_k \; OUT_k \; \texttt{[S]} \; o_k \; \texttt{[C]} \; c_{k+1} \; \texttt{[F]} \; \text{FLAG}_k$$
  trained with next-token prediction. **Exactly three positions carry loss**: $o_k$ (predicted at `[S]`), $c_{k+1}$ (predicted at `[C]`), and $\text{FLAG}_k$ (predicted at `[F]`). The trigger tokens `[C]` and `[F]` are inserted by the harness at inference, never generated by the model, so their labels are `-100`, as are all base-prompt tokens.
* **`SM-SeqRev`** (`encode_sm_seq_rev`): as `SM-Seq` in carry-first order,
  $$\texttt{[BOS]} \; \mathcal{A} \; a \; b \; c_k \; OUT_k \; \texttt{[C]} \; c_{k+1} \; \texttt{[S]} \; o_k \; \texttt{[F]} \; \text{FLAG}_k$$
  with loss on $c_{k+1}$ (predicted at `[C]`), $o_k$ (at `[S]`) and $\text{FLAG}_k$ (at `[F]`).
* **`SM-Fork`** (`encode_sm_fork`): each sampled transition yields **three rows**, the base prompt ending on `[S]`, `[C]` and `[F]`, each with exactly one loss position (the tail query) targeting $o_k$, $c_{k+1}$ and $\text{FLAG}_k$ respectively. `sm_fork_batch(batch_size)` therefore holds $\lfloor \texttt{batch\_size} / 3 \rfloor$ transitions ($3\lfloor \texttt{batch\_size} / 3 \rfloor$ rows), i.e. one third as many transitions per optimisation step as the other `SM` sub-variants at equal `batch_size`.

### 4.4 Autoregressive Full-Trace Batches (`AR`)
For `AR`, each training sample is one full calculation sequence:
$$\text{Sequence} = \texttt{[BOS]} \parallel \mathcal{A} \parallel a \parallel b \parallel \texttt{[START]} \parallel \texttt{[S]} \, o_0 \, \texttt{[C]} \, c_1 \dots \texttt{[S]} \, o_K \, \texttt{[C]} \, c_{K+1} \parallel \texttt{[FINISHED]}$$
* Labels for the static prompt prefix (`[BOS]`, $\mathcal{A}$, $a$, $b$, `[START]`) and for padding are `-100`.
* **Every trace token carries loss**, including the markers `[S]` and `[C]` and the terminator `[FINISHED]`: in `AR` the model generates them itself.
* Tier-1 trace coordinates follow the slot rule of [`02_mechanics.md`](02_mechanics.md) Section 4.2.

---

## 5. Runtime Execution Harnesses

The harnesses implement Algorithms 1–2 and the generation contract of [`02_mechanics.md`](02_mechanics.md) Sections 3.3 and 4.3. They contain no numeral arithmetic, decode by unconstrained `argmax`, and return a `(result, status)` pair; any status other than `"finished"` is scored as a failure. Each also accepts an optional `carries` list to which emitted carries are appended (logging and out-of-axiom scoring only; omitted from the listings below). They recompute the full sequence on every forward pass (sequences are short), avoiding KV-cache invalidation logic entirely.

Coordinate helpers used below (Tier 1; ignored under NoPE): sum-slot tokens take `(ROLE_OUT, k)`, carry-slot tokens `(ROLE_CARRY, k + 1)`, the flag trigger `(ROLE_FLAG, k)`.

### 5.1 Multi-Head Stateful Driver (`SM-MH`)
```python
def run_state_machine_mh(model, axiom, a, b, tier_format, stepper=None):
    """Multi-Head Stateful Driver (Paradigm A1: SM-MH). One forward pass per column."""
    max_steps = max(len(a), len(b)) + 1            # largest legal step count
    carry, out_digits = axiom[0], []

    for k in range(max_steps):
        prompt = build_sm_prompt(axiom, a, b, carry, out_digits, tier_format)  # ... [S]
        logits_sum, logits_carry, logits_flag = model.forward_sm_mh_single(prompt)

        o_k = int(torch.argmax(logits_sum))
        c_next = int(torch.argmax(logits_carry))
        flag = FLAG_FINISHED if int(torch.argmax(logits_flag)) == 1 else FLAG_CONTINUE

        out_digits.append(o_k)
        carry = c_next

        if stepper:
            stepper.step(k, prompt, o_k, c_next, flag, [logits_sum, logits_carry, logits_flag])
        if flag == FLAG_FINISHED:
            return out_digits, "finished"

    return out_digits, "max_steps_exceeded"
```
(`forward_sm_mh_single` is the batch-of-one wrapper around `forward_sm_mh`.)

### 5.2 Stateless Sequential Micro-AR Driver (`SM-Seq`)
```python
def run_state_machine_seq(model, axiom, a, b, tier_format, stepper=None):
    """Stateless Micro-AR Driver (Paradigm A2: SM-Seq). Three forward passes per column."""
    max_steps = max(len(a), len(b)) + 1            # largest legal step count
    carry, out_digits = axiom[0], []

    for k in range(max_steps):
        # Micro-Pass 1: base prompt already terminates on [S]; emits o_k
        p1 = build_sm_prompt(axiom, a, b, carry, out_digits, tier_format)
        logits1 = model.next_logits(p1)
        o_k = int(torch.argmax(logits1))

        # Micro-Pass 2: emits outgoing carry c_{k+1}
        p2 = p1.push(o_k, ROLE_OUT, k).push(TOKEN_C, ROLE_CARRY, k + 1)
        logits2 = model.next_logits(p2)
        c_next = int(torch.argmax(logits2))

        # Micro-Pass 3: emits control flag
        p3 = p2.push(c_next, ROLE_CARRY, k + 1).push(TOKEN_FLAG, ROLE_FLAG, k)
        logits3 = model.next_logits(p3)
        flag = int(torch.argmax(logits3))

        out_digits.append(o_k)
        carry = c_next                              # transients p1..p3 are discarded

        if stepper:
            stepper.step(k, p1, o_k, c_next, flag, [logits1, logits2, logits3])
        if flag == FLAG_FINISHED:
            return out_digits, "finished"

    return out_digits, "max_steps_exceeded"
```

`SM-SeqRev` uses `run_state_machine_seq_rev`: the same driver with the base prompt built with `query_token=TOKEN_C`, Pass 1 emitting $c_{k+1}$ and Pass 2 (triggered by `[S]`) emitting $o_k$.

`SM-Fork` uses `run_state_machine_fork` ([`02_mechanics.md`](02_mechanics.md) Algorithm 3): the three query prompts are equal-length, so `model.next_logits_batch` evaluates them in one unpadded batch of 3.

### 5.3 Autoregressive Generative Driver (`AR`)
```python
def run_autoregressive_trace(model, axiom, a, b, tier_format, stepper=None):
    """Generative Driver (Paradigm B: AR). Plain greedy decoding, no state re-injection."""
    prompt = build_ar_prompt(axiom, a, b, tier_format)      # [BOS] ... [START]
    budget = 4 * (max(len(a), len(b)) + 1) + 1              # longest well-formed trace
    seq, trace = prompt, []

    for i in range(budget):
        token = int(torch.argmax(model.next_logits(seq)))
        trace.append(token)
        if token == FLAG_FINISHED:
            break
        # Tier 1 coordinates depend only on the slot index i, never on `token`
        k = i // 4
        role, sig = (ROLE_OUT, k) if i % 4 < 2 else (ROLE_CARRY, k + 1)
        seq = seq.push(token, role, sig)

    out_digits, carries, status = parse_trace(trace)        # 02_mechanics.md Section 4.3

    if stepper:
        stepper.show_trace(prompt, trace, out_digits, status)
    return out_digits, status
```

---

## 6. Shared Interactive Educational Stepper (`utils/stepper.py`)

The interactive stepper provides an explanatory terminal interface for both training and evaluation harnesses when `--interactive` is enabled.

The stepper sits **outside** the zero-arithmetic harness boundary: it calls `reference_add` (Section 4.2) to show what the correct transition *is*, alongside what the model emitted. It has no access to, and makes no claims about, the model's internal mechanism; its arithmetic narration describes the reference computation, not the network.

### 6.1 Stepper Control Loop & Keyboard Shortcuts
At each step pause, the stepper halts execution and awaits user input:
* `[Enter]`: Advance by a single step.
* `c`: Continue execution without pausing for the remainder of the problem/epoch.
* `q`: Abort execution or exit interactive mode.
* `v`: Toggle verbose inspection (shows top-5 logits and probability distributions).

### 6.2 Terminal Visualisation Dashboard
```text
================================================================================
TransformAdder Interactive Stepper | Step k = 3 | Base B = 3
================================================================================
Axiom:      [0x30, 0x31, 0x32] -> Symbols: '0' (0), '1' (1), '2' (2)
Operand A:  Column 3 Digit = '2' (value 2)
Operand B:  Column 3 Digit = '1' (value 1)
Carry-In:   c_3 = '1' (value 1)
--------------------------------------------------------------------------------
Model Predictions:
  Sum Output:       o_3 = '1' (confidence: 99.4%)   [matches reference]
  Outgoing Carry:   c_4 = '1' (confidence: 98.9%)   [matches reference]
  Control Flag:     FLAG_CONTINUE (confidence: 99.8%) [matches reference]
--------------------------------------------------------------------------------
Reference Computation (oracle, not a readout of the model's internals):
  * Correct transition for this column:
      val(A_3) + val(B_3) + val(c_3) = 2 + 1 + 1 = 4
      Sum digit:    4 mod 3 = 1  -> Symbol '1'
      Carry-out:    floor(4/3) = 1 -> Symbol '1'
  * What Happens Next:
    Harness appends the model's '1' to accumulated OUT and updates active carry
    to c_4 = '1'. Loop continues to column k = 4.
================================================================================
[Enter: Step | c: Continue | v: Verbose Logits | q: Quit]: 
```

---

## 7. Training Loss Objectives & Metrics

### 7.1 Loss Formulations
* **Multi-Head Composite Loss (`SM-MH`)**:
  $$\mathcal{L}_{\text{MH}} = 1.0 \cdot \mathcal{L}_{\text{CE}}(\hat{y}_{\text{sum}}, o_k) + 1.0 \cdot \mathcal{L}_{\text{CE}}(\hat{y}_{\text{carry}}, c_{k+1}) + 0.5 \cdot \mathcal{L}_{\text{CE}}(\hat{y}_{\text{flag}}, \text{FLAG}_k)$$
* **Masked Causal Cross-Entropy Loss (`SM-Seq`, `SM-SeqRev`, `SM-Fork`, `AR`)**:
  $$\mathcal{L}_{\text{Seq}} = -\frac{1}{|\mathcal{T}_{\text{target}}|} \sum_{t \in \mathcal{T}_{\text{target}}} \log p(y_t \mid y_{<t})$$
  $\mathcal{T}_{\text{target}}$ is the set of **model-emitted** positions: for `SM-Fork` the single tail-query position of each row; for `SM-Seq` and `SM-SeqRev` the three positions $o_k, c_{k+1}, \text{FLAG}_k$ (Section 4.3); for `AR` every trace token after `[START]` (Section 4.4). All other positions have label `-100`.

### 7.2 Evaluation Metrics & Local JSONL Logging
Evaluation is zero-shot, without gradient updates, scored against `reference_add`, and stored locally. Metrics 1–3 are **free-running** (the model consumes its own outputs through the harness); metrics 4–5 are **teacher-forced** (each step is posed from the oracle state, isolating single-transition fidelity from error accumulation).
1. **Sequence Exact Match (EM)**: Proportion of test problems where harness status is `"finished"` and the complete output sequence equals the reference $OUT$ exactly (same digits, same length, hence correct halting).
2. **Column-Level Digit Accuracy**: Proportion of individual sum digits correctly predicted, over reference columns.
3. **Carry-Chain Accuracy**: Digit accuracy restricted to columns with non-zero carry-in or non-zero carry-out.
4. **Step-Transition Accuracy**: Proportion of teacher-forced transitions for which all three of $(o_k, c_{k+1}, \text{FLAG}_k)$ are correct. For `AR`, the transition for column $k$ is posed by teacher-forcing the reference trace up to column $k$; the predicted `[S]`-or-`[FINISHED]` choice after $c_{k+1}$ plays the role of $\text{FLAG}_k$.
5. **Halting Accuracy**: Proportion of teacher-forced transitions with correct $\text{FLAG}_k$, reported separately for terminal ($k = K$) and non-terminal steps.
6. **Failure Status Counts**: Counts of `max_steps_exceeded`, `max_tokens_exceeded`, and `malformed` runs, and of emitted numerals (sum digits and carries) lying outside the active axiom.
7. **Local Log Format (`metrics.jsonl`)**:
   ```json
   {"step": 5000, "loss": 0.042, "id_em": 0.992, "ood_base3_em": 0.945, "ood_base12_em": 0.910, "extrap_len_em": 0.885, "ood_base12_step_acc": 0.991, "ood_base12_halt_acc": 0.998}
   ```

### 7.3 Baseline Training Hyperparameters (Provisional)
Defaults for all primary variants; provisional until validated by the first in-domain runs, after which they are frozen across the matrix.

| Hyperparameter | Value |
| :--- | :--- |
| Optimiser | AdamW, $\beta = (0.9, 0.95)$, $\epsilon = 10^{-8}$ |
| Peak learning rate | $3 \times 10^{-4}$ |
| Schedule | 1,000-step linear warm-up, cosine decay to 10% of peak |
| Weight decay | 0.1 on weight matrices; 0 on norm gains and role embeddings |
| Gradient clipping | Global norm 1.0 |
| Batch size | 256 samples (`SM`: step transitions; `AR`: full traces) |
| Training steps | 100,000 |
| Precision | bfloat16 autocast, float32 master weights and optimiser state |
| Initialisation | $\mathcal{N}(0, 0.02^2)$ for all linear weights; residual output projections (`out_proj`, `w_down`) scaled by $1/\sqrt{2 N_{\text{layers}}}$ |
| Seeds | 3 per configuration ($\{0, 1, 2\}$), seeding both initialisation and the data generator |
| Evaluation cadence | Every 5,000 steps on 1,000-problem cells; final evaluation on the full 10,000-problem cells of [`01_research_goal.md`](01_research_goal.md) Section 4.1 |
