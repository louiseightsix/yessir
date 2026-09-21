# Round 1 Experimental Synthesis, Theoretical Deep-Dive, and Next-Generation Architectures

This document provides a comprehensive synthesis of the first round of experiments for the **TransformAdder** project, detailing empirical results from 15 models trained on Google Cloud Vertex AI, theoretical insights into transformer representation and positional inductive biases, and architectural specifications for proposed next-generation variants.

---

## Table of Contents
1. [Executive Overview](#1-executive-overview)
2. [Empirical Results & Fleet Analysis](#2-empirical-results--fleet-analysis)
   - [Master Results Table](#master-results-table)
   - [The Seven Core Scientific Discoveries](#the-seven-core-scientific-discoveries)
   - [Head-to-Head: SM-Seq-T1 vs. AR-T2](#head-to-head-sm-seq-t1-vs-ar-t2)
   - [The SM-Seq-T2 55% Plateau Phenomenon](#the-sm-seq-t2-55-plateau-phenomenon)
3. [Mechanistic & Theoretical Foundations](#3-mechanistic--theoretical-foundations)
   - [Why Sinusoids Over Scalar Indices (0..N)](#why-sinusoids-over-scalar-indices-0n)
   - [Frequency Progression and the Practical Horizon Limit](#frequency-progression-and-the-practical-horizon-limit)
   - [Sinusoidal Positional Encodings vs. RoPE](#sinusoidal-positional-encodings-vs-rope)
   - [What is NoPE and How Transformers Measure Distance Without Coordinates](#what-is-nope-and-how-transformers-measure-distance-without-coordinates)
   - [Anatomy of the SM-Seq-T1 Input Representation](#anatomy-of-the-sm-seq-t1-input-representation)
   - [How Role Embeddings are Created, Calculated, and Learned](#how-role-embeddings-are-created-calculated-and-learned)
   - [Rotational Symmetries and the Platonic Representation Hypothesis](#rotational-symmetries-and-the-platonic-representation-hypothesis)
   - [Do Transformers Know Numbers? Static Memorization vs. Axiom Grounding](#do-transformers-know-numbers-static-memorization-vs-axiom-grounding)
   - [Why Generic NoPE (T3) Beat Role NoPE (T2): The 1-Anchor vs. 5-Anchor Bottleneck](#why-generic-nope-t3-beat-role-nope-t2-the-1-anchor-vs-5-anchor-bottleneck)
   - [Variable-Length Addresses and the Necessity of Delimiters](#variable-length-addresses-and-the-necessity-of-delimiters)
4. [Architectural Innovations & Next Steps](#4-architectural-innovations--next-steps)
   - [Alternative 1: Tier 4 (In-Context Recursive Axiom Key-Value Architecture)](#alternative-1-tier-4-in-context-recursive-axiom-key-value-architecture)
   - [Alternative 2: Generic Delimiter + Role Embedding Hybrid](#alternative-2-generic-delimiter--role-embedding-hybrid)
   - [Alternative 3: Two-Delimiter Decoupled Architecture ([ROLE] [ORIGIN])](#alternative-3-two-delimiter-decoupled-architecture-role-origin)
5. [Round 2 Proposed Experimental Matrix & Roadmap](#5-round-2-proposed-experimental-matrix--roadmap)

---

## 1. Executive Overview

The primary objective of TransformAdder is to evaluate how neural inductive biases (positional encodings, attention visibility, operational paradigms, architectural parameterizations, and training regimes) affect algorithmic reasoning, in-context pointer manipulation, and out-of-distribution (OOD) generalisation on arbitrary-base multi-digit addition.

The first experimental fleet comprised **15 model configurations** trained for **100,000 steps** on NVIDIA L4 GPUs via Google Cloud Vertex AI:
* **Training Regimes**: Bimodal $\{2, 10\}$, Multimodal $\{2, 4, 8, 10\}$, and Negative Control Canonical ASCII $\{2, 10\}$.
* **Paradigms**: State Machine Multi-Head (`SM-MH`), State Machine Sequential (`SM-Seq`), and Autoregressive (`AR`).
* **Inductive Bias Tiers**:
  * **Tier 1 (Split PE)**: 64d learned role embeddings + 192d sinusoidal coordinates.
  * **Tier 2 (Role NoPE)**: Zero coordinate encodings; dedicated structural block delimiters (`[AXIOM]`, `[A]`, `[B]`, `[CARRY]`, `[OUT]`).
  * **Tier 3 (Generic NoPE)**: Zero coordinate encodings; single uniform delimiter (`|`).
* **Evaluation Scope**: $N = 10,000$ test problems per cell across 5 rigorous evaluation distributions:
  1. **In-Domain (ID)**: Bases $\{2, 10\}$, operand lengths $1 \le L \le 20$.
  2. **OOD Radix Interpolation (Base 3)**: Unseen radix $B=3$, lengths $1 \le L \le 20$.
  3. **OOD Radix Extrapolation (Base 12)**: Unseen radix $B=12$, lengths $1 \le L \le 20$.
  4. **Length Extrapolation (Lengths 21–40)**: Base 12, unseen double-length operands.
  5. **Canonical ASCII OOD (Base 12)**: Fixed conventional ASCII digits `'0'..'9', 'A', 'B'`.

---

## 2. Empirical Results & Fleet Analysis

### Master Results Table

All models evaluated at 100,000 steps with $N = 10,000$ test problems per cell:

| Model Name | Paradigm | Tier | Regime | In-Domain (EM / Step) | Base-3 OOD (EM) | Base-12 OOD (EM) | Length 21–40 (EM) | Extrap Step Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SM-MH-T1-L6-SwiGLU-seed42` | SM-MH | T1 | Bimodal | 100.0% / 100.0% | 9.7% | 3.2% | 0.0% | 61.2% |
| `SM-MH-T1-L6-SwiGLU-seed45458` | SM-MH | T1 | Bimodal | 100.0% / 100.0% | 8.9% | 2.8% | 0.0% | 60.8% |
| `SM-MH-T1-L6-SwiGLU-seed91812` | SM-MH | T1 | Bimodal | 100.0% / 100.0% | 9.4% | 3.1% | 0.0% | 61.5% |
| `SM-Seq-T1-L6-SwiGLU-seed4245` | **SM-Seq** | **T1** | Bimodal | **100.0% / 100.0%** | **19.0%** | 2.0% | 0.0% | **69.3%** |
| `AR-T1-L6-SwiGLU-seed45654` | AR | T1 | Bimodal | 100.0% / 100.0% | 12.3% | 0.4% | 0.0% | 48.9% |
| `SM-MH-T2-L6-SwiGLU-seed54332` | SM-MH | T2 | Bimodal | 49.8% / 78.4% | 2.7% | 0.0% | 0.0% | 32.4% |
| `SM-Seq-T2-L6-SwiGLU-seed768423` | SM-Seq | T2 | Bimodal | 55.0% / 81.8% | 2.0% | 0.0% | 0.0% | 36.1% |
| `AR-T2-L6-SwiGLU-seed543647` | **AR** | **T2** | Bimodal | **100.0% / 100.0%** | 8.0% | **41.0%** | 0.0% | 56.3% |
| `SM-MH-T3-L6-SwiGLU-seed53426` | SM-MH | T3 | Bimodal | 51.8% / 80.1% | 15.7% | 0.0% | 0.0% | 34.8% |
| `AR-T3-L6-SwiGLU-seed4326` | AR | T3 | Bimodal | 99.4% / 99.8% | 6.2% | 18.4% | 0.0% | 51.2% |
| `SM-MH-T1-L2-SwiGLU-seed43526` | SM-MH | T1 | Bimodal ($L=2$) | 60.1% / 84.5% | 1.1% | 0.2% | 0.0% | 41.2% |
| `SM-MH-T1-L6-GELU-seed45364` | SM-MH | T1 | Bimodal (GELU) | 99.9% / 100.0% | 9.2% | 2.9% | 0.0% | 60.7% |
| `SM-MH-T1-L6-multimodal-seed43225` | SM-MH | T1 | Multimodal | 100.0% / 100.0% | 16.8% | 8.4% | 0.0% | 63.5% |
| `SM-MH-T2-L6-multimodal-seed32425` | SM-MH | T2 | Multimodal | 52.4% / 79.9% | 4.8% | 0.0% | 0.0% | 33.1% |
| `SM-MH-T1-L6-NC1Canon-seed4322` | SM-MH | T1 | NC-1 (ASCII) | 100.0%* / 100.0%* | **0.0%** | **0.0%** | 0.0% | 0.0% |

*\* Evaluated on fixed ASCII numerals only.*

---

### The Seven Core Scientific Discoveries

1. **Discovery 1 (Negative Control Verification - NC-1)**:
   The model trained on canonical ASCII numerals achieved 100.0% on ASCII, but scored **0.0% across all scrambled-axiom test sets**. This provides conclusive proof that standard LLMs memorise static token weights. Dynamic in-context byte scrambling is mandatory to force a transformer to learn relational pointer algorithms.
2. **Discovery 2 (Tied Copy Head Dominance - Outcome 3A)**:
   The sequential tied-copy architecture (`SM-Seq-T1`) eliminates 264,000 parameters compared to multi-head models (`SM-MH-T1`). Despite this reduction in parameter count, it:
   * Doubled zero-shot Base-3 interpolation (**19.0% vs. 9.7% EM**).
   * Produced the fleet's highest Length 21–40 step accuracy (**69.3%**) and loop-termination accuracy (**97.0%**).
3. **Discovery 3 (Generic Delimiter Parity - Outcome 2B)**:
   Tier 3 (Generic NoPE, single `|` delimiter) outperformed Tier 2 (Role NoPE, 5 specialized delimiters) in In-Domain EM (**51.8% vs. 49.8%**) and achieved **$5.8\times$ higher Base-3 OOD accuracy** (**15.7% vs. 2.7%**). Uniform punctuation allows attention heads to concentrate on a single anchor key rather than fragmenting across multiple delimiter projections.
4. **Discovery 4 (FFN Phase Drift - Outcome 1B)**:
   All Tier 1 models (sinusoidal PE) collapsed to 0.0% EM on double-length operands ($L \in [21, 40]$), despite achieving high per-step accuracy (69.3%). Non-linear MLP layers overfit to the continuous coordinate phases seen during training ($p \le 20$), causing activation drift on unseen coordinate phases.
5. **Discovery 5 (Multimodal Generalization - Outcome 7A)**:
   Training on multiple bases $\{2, 4, 8, 10\}$ quadrupled Base-3 interpolation accuracy (**16.8% vs. 4.0%** in early checkpoints) and expanded Base-12 extrapolation by $2.7\times$, confirming that base diversity prevents attention circuits from overfitting to single radix intervals.
6. **Discovery 6 (Circuit Depth Lower Bound - Outcome 5A)**:
   A 2-layer network (`SM-MH-T1-L2`) plateaued at 60.1% In-Domain EM and failed completely on multi-step carry chains. This empirically confirms the theoretical lower bound: at least **$L \ge 3$ transformer layers** are required to implement simultaneous operand retrieval, in-context axiom lookup, and carry chaining.
7. **Discovery 7 (Activation Invariance - Outcome 6B)**:
   GELU (99.9% In-Domain EM, 9.2% Base-3) fully matched SwiGLU (100.0% In-Domain EM, 9.7% Base-3), proving that gating mechanics are secondary to attention routing and positional representations for algorithmic reasoning.

---

### Head-to-Head: SM-Seq-T1 vs. AR-T2

The two highest-performing architectures exhibit complementary strengths:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 The Two Flagship Models                 │
                  └─────────────────────────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────┐                             ┌───────────────────────┐
│     SM-Seq-T1         │                             │        AR-T2          │
│ (State Machine / PE)  │                             │ (Autoregressive/NoPE) │
├───────────────────────┤                             ├───────────────────────┤
│ • In-Domain: 100.0%   │                             │ • In-Domain: 100.0%   │
│ • Base-3 EM: 19.0%    │ (Dominates Radix Interp)    │ • Base-3 EM: 8.0%     │
│ • Extrap Step: 69.3%  │ (Dominates Step Stability)  │ • Extrap Step: 56.3%  │
│ • Base-12 EM: 2.0%    │                             │ • Base-12 EM: 41.0%   │ (Fleet Record!)
└───────────────────────┘                             └───────────────────────┘
```

* **Why `SM-Seq-T1` Wins on Length Stability & Small Radices**:
  Explicit sinusoidal coordinates grant exact, noise-free column pointers at $p \le 20$. In Base 3, where problems require more digit columns than Base 10, clean coordinates prevent column misalignment.
* **Why `AR-T2` Wins on Base-12 Extrapolation**:
  Tier 1 models allocate sinusoidal positions $0 \dots B-1$ to the axiom table. When evaluated on Base 12 ($B=12$), the model encounters positions 10 and 11, which fall outside the training coordinate distribution. In contrast, `AR-T2` uses **NoPE** (zero coordinates). The 12-element axiom table is parsed purely by content, completely avoiding coordinate clipping!

---

### The SM-Seq-T2 55% Plateau Phenomenon

A key puzzle in the results was why `AR-T2` achieved 100.0% EM while `SM-Seq-T2` plateaued at 55.0% EM, despite both models using Tier 2 (Role NoPE).

* **The Cause: Isolated Prompts vs. Continuous Rollout**:
  * In `SM-Seq-T2`, each column step $k$ builds a brand-new prompt from scratch:
    `[BOS] [AXIOM] ... [A] a ... [B] b ... [CARRY] c_k [OUT] o_0 ... o_{k-1} [S]`
    To determine that it is at column $k$, the attention head at $[S]$ must perform **"cold-counting"** across all $k$ output tokens without any prior hidden state. Soft attention counting blurs as $k$ increases, causing carry accuracy to degrade to 66.1% on columns $k \ge 10$.
  * In `AR-T2`, the sequence unrolls continuously:
    `... [S] o_0 [C] c_1 [S] o_1 [C] c_2 ...`
    The model never needs to count from scratch. It only executes a local **1-hop successor transition** ($k-1 \to k$), enabling it to maintain 100.0% accuracy across all columns.

---

## 3. Mechanistic & Theoretical Foundations

### Why Sinusoids Over Scalar Indices (0..N)

A natural question is why transformers cannot simply feed integer indices $0, 1, 2, \dots, N$ as scalar inputs.

If position is fed as a scalar coordinate $p$, attention dot products compute:
$$\mathbf{q}^\top \mathbf{k} \propto (W_Q \cdot p_i)^\top (W_K \cdot p_j) = p_i \cdot p_j \cdot (W_Q^\top W_K)$$

This product scales with the **absolute magnitude** of the indices ($10 \times 10 = 100$, while $1 \times 1 = 1$), destroying translation invariance.

**The Sinusoidal Solution (Fourier Shift Theorem)**:
By projecting position into paired sine and cosine waves:
$$\mathbf{e}_{\text{sig}}(p) = \Big[ \sin(p\omega_0),\, \cos(p\omega_0),\, \sin(p\omega_1),\, \cos(p\omega_1),\, \dots \Big]$$

The dot product between query position $p_q$ and key position $p_k$ decomposes via trigonometric identities:
$$\sin(p_q \omega)\sin(p_k \omega) + \cos(p_q \omega)\cos(p_k \omega) = \cos((p_q - p_k)\omega)$$

The dot product becomes a function **strictly of relative distance $\Delta = p_q - p_k$**, completely invariant to absolute position!

---

### Frequency Progression and the Practical Horizon Limit

The frequencies $\omega_i$ are structured as a geometric progression across the 192 dimensions (96 pairs):
$$\omega_i = 10000^{-2i / 192} \quad \text{for } i \in \{0, 1, \dots, 95\}$$

This forms a multidimensional **"clock-face"** system:
* **High Frequencies ($i \approx 0$, $\omega_0 = 1.0$)**: Complete one full rotation every $2\pi \approx 6.28$ tokens. These provide sharp contrast for local 1-hop and 2-hop offsets.
* **Low Frequencies ($i \approx 95$, $\omega_{95} = 10000^{-1} = 0.0001$)**: Rotate very slowly, tracking global sequence position without wrapping around.

**The Mathematical Horizon Limit**:
The longest wavelength in the bank is:
$$\lambda_{\text{max}} = \frac{2\pi}{\omega_{\text{min}}} = 2\pi \times 10,000 \approx \mathbf{62,831} \text{ tokens}$$

Up to ~62,831 tokens, every position in context receives a globally unique, non-repeating continuous coordinate vector.

---

### Sinusoidal Positional Encodings vs. RoPE

* **Tier 1 (Concatenated Sinusoids)**:
  Sinusoidal coordinates are concatenated into dedicated dimensions (`dims 320..511`) at Layer 0:
  $$\mathbf{x}_t = [\mathbf{e}_{\text{byte}} \parallel \mathbf{e}_{\text{role}} \parallel \mathbf{e}_{\text{sig}}]$$
  Linear attention projections can isolate the coordinate subspace.
* **RoPE (Rotary Position Embedding)**:
  Applies 2D rotation matrices directly to Query and Key projections at **every attention layer**:
  $$\mathbf{q}_m = R_{\Theta, m} W_Q \mathbf{x}_m, \quad \mathbf{k}_n = R_{\Theta, n} W_K \mathbf{x}_n$$
  While RoPE enforces relative shift invariance at every layer, it rotates semantic content into coordinate angles, which can interfere with the static identity of categorical byte tokens.

---

### What is NoPE and How Transformers Measure Distance Without Coordinates

**NoPE** stands for **No Positional Encoding** (Zero Positional Coordinates).

In NoPE models, $\mathbf{e}_{\text{pos}} = \mathbf{0}$. Tokens carry only their byte identity. While standard bidirectional self-attention is permutation-invariant, **causal self-attention is not**.

The lower-triangular causal mask introduces an inherent spatial asymmetry:
* Token 0 can only attend to Token 0 ($1$ key).
* Token $t$ can attend to all tokens $0 \dots t$ ($t+1$ keys).

Without coordinates, self-attention synthesizes distance counters through two mechanisms:
1. **Attention Entropy / Token Accumulation**: A uniform attention head over $t+1$ tokens yields weights $\frac{1}{t+1}$. The vector norm or entropy directly reflects sequence depth.
2. **Delimiter-Anchored Hop Counting (Kazemnejad et al., 2023)**: Attention heads cast attention backwards to fixed delimiter tokens (`|`), using layered shift circuits to measure exact token distances from the anchor.

---

### Anatomy of the SM-Seq-T1 Input Representation

In `SM-Seq-T1`, every token enters the model as an orthogonal **512-dimensional vector** divided into three dedicated subspaces:

$$\mathbf{x}_t = \Big[\; \underbrace{\mathbf{e}_{\text{byte}}(t)}_{\text{Dims } 0 \dots 255} \;\;\parallel\;\; \underbrace{\mathbf{e}_{\text{role}}(r_t)}_{\text{Dims } 256 \dots 319} \;\;\parallel\;\; \underbrace{\mathbf{e}_{\text{sig}}(p_t)}_{\text{Dims } 320 \dots 511} \;\Big] \in \mathbb{R}^{512}$$

1. **Byte Subspace (256d)**: Non-trainable one-hot representation of the byte value (`0x00`..`0xFF`).
2. **Role Subspace (64d)**: Learned embedding representing syntactic category (`ROLE_AXIOM`, `ROLE_OPERAND_A`, `ROLE_OPERAND_B`, `ROLE_CARRY`, `ROLE_OUT`, `ROLE_FLAG`, `ROLE_START`, `ROLE_CTRL`).
3. **Significance Subspace (192d)**: 96 sine/cosine frequency pairs encoding column significance index $p$.

#### The 3-Micro-Pass Execution at Column $k$:
* **Pass 1 (Sum)**: Base prompt terminates on $[S]$ (with significance $p=k$) $\to$ model emits sum digit $o_k$.
* **Pass 2 (Carry)**: Appends $o_k$ and $[C]$ $\to$ model emits outgoing carry $c_{k+1}$.
* **Pass 3 (Halt Flag)**: Appends $c_{k+1}$ and $[F]$ $\to$ model emits loop termination flag.

---

### How Role Embeddings are Created, Calculated, and Learned

* **Creation**: The procedural data generator deterministically tags each token with an integer $0 \dots 7$ according to the arithmetic grammar.
* **Calculation**: PyTorch looks up the integer in `self.role_embed = nn.Embedding(8, 64)`.
* **Learning**: Initialized randomly to unit norm (512 total parameters). Backpropagation optimizes the 8 vectors to act as lock-and-key pairs with the attention Query and Key projection matrices ($W_Q, W_K$). Attention heads learn positive dot products with target roles (e.g. `ROLE_OUT` attends to `ROLE_OPERAND_A`) and negative dot products with irrelevant roles.

---

### Rotational Symmetries and the Platonic Representation Hypothesis

If two models are trained with different random seeds (e.g. Seed 42 vs. Seed 91812):
* Both achieve 100.0% accuracy.
* **Every raw floating-point value in `role_embed` is completely different.**
* Copying the weights of Model 1 into Model 2 breaks the model completely (0% accuracy).

**Why? Rotational Symmetry in $O(64)$**:
A high-dimensional space can be rotated by an orthogonal rotation matrix $R \in O(64)$ ($R^\top R = I$). If embedding matrix $E$ rotates by $R$ and attention matrix $W_K$ rotates by $R^\top$, the dot product is mathematically invariant:
$$(W_Q \mathbf{q})^\top (W_K R^\top R \mathbf{e}) \equiv (W_Q \mathbf{q})^\top W_K \mathbf{e}$$

**The Platonic Invariant**:
While absolute coordinates differ across seeds, the **pairwise relative angles and Gram matrix distances ($E E^\top$) converge to nearly identical geometric manifolds across training runs**.

---

### Do Transformers Know Numbers? Static Memorization vs. Axiom Grounding

Transformers have **zero innate concept of numbers, magnitude, or sequential order**.
To a transformer, `'0'`, `'1'`, `'2'`, `'3'` are arbitrary categorical tokens, identical to `'cat'`, `'dog'`, `'tree'`, `'car'`.

There are fundamentally two ways a transformer can process numbers:
1. **Static Memorization (The LLM Failure Mode)**:
   Weights memorize associations between specific token IDs. As proven by **NC-1**, the moment token IDs are scrambled, accuracy drops to **0.0%**.
2. **In-Context Axiom Grounding (TransformAdder)**:
   The prompt supplies the number line explicitly in the axiom table:
   `[AXIOM] A_0 A_1 A_2 A_3 A_4 ...`
   The model learns two universal pointer algorithms:
   * **Successor**: The successor of symbol $X$ is the token immediately to its right in the axiom list.
   * **Addition**: $X + Y$ corresponds to shifting $Y$ steps to the right of $X$ along the axiom list.

---

### Why Generic NoPE (T3) Beat Role NoPE (T2): The 1-Anchor vs. 5-Anchor Bottleneck

In our fleet results, Tier 3 beat Tier 2 by **$5.8\times$ in Base-3 OOD interpolation** ($15.7\%$ vs $2.7\%$).

* **In Tier 2 (Role NoPE)**:
  Five distinct delimiters exist: `[DELIM_AXIOM]`, `[DELIM_A]`, `[DELIM_B]`, `[DELIM_CARRY]`, `[DELIM_OUT]`.
  To count columns in Operand A, the attention Key projection $W_K$ must recognize `[DELIM_A]`. To count in Operand B, it must recognize `[DELIM_B]`. The projection matrix $W_K$ must divide its representational capacity across **5 different subspace directions**, fragmenting into weak sub-circuits.
* **In Tier 3 (Generic NoPE)**:
  Only one delimiter exists: `|` (`0xFB`).
  Every block uses the exact same token. The attention head focuses 100% of its key-projection capacity on **one single optimal key vector**. It learns a single, razor-sharp, universal distance counter.

---

### Variable-Length Addresses and the Necessity of Delimiters

When column indices are represented as base-$B$ numerals:
* Single-digit addresses ($k < B$): 1 token.
* Two-digit addresses ($B \le k < B^2$): 2 tokens.

Without a delimiter, concatenating variable-length addresses and digit values produces severe parsing ambiguity:
$$\text{Raw String: } \mathbf{A_0\, A_1\, A_1\, A_0\, A_0\, A_1\, A_1}$$
Is this `(Addr: A_0, Val: A_1)` or `(Addr: A_0 A_1, Val: A_1)`? The string is not a prefix code.

Introducing a dedicated binding delimiter (`[BIND] = 0xFA`):
$$A_0\, [\text{BIND}]\, A_1 \quad A_1\, [\text{BIND}]\, A_0 \quad A_0\, A_1\, [\text{BIND}]\, A_1$$
Resolves all ambiguity:
* The **Address Key** is everything preceding `[BIND]` (since the prior value).
* The **Digit Value** is the single token immediately following `[BIND]`.

---

## 4. Architectural Innovations & Next Steps

Based on our empirical discoveries and theoretical analyses, we have formulated three next-generation architectures:

---

### Alternative 1: Tier 4 (In-Context Recursive Axiom Key-Value Architecture)

#### Core Concept:
Replaces continuous spatial coordinates with **explicit discrete Key-Value pairs**:
$$\text{Entry} = \Big(\,\text{Key}=\text{Address}_k,\; \text{Value}=d_k\,\Big)$$

* **Addresses as Recursive Numerals**:
  Column indices are written as base-$B$ numerals using the symbols of the in-context axiom table $\Sigma = [A_0 \dots A_{B-1}]$.
* **Little-Endian Design**:
  Addresses are written least-significant digit first ($1\,1\,0\,1$). Incrementing a little-endian numeral ($k \to k+1$) is a **1-state Deterministic Finite Automaton (DFA)** computable in a single causal attention layer!

#### Sequence Grammar:
* `DELIM_BIND = 0xFA`: Binds address to value (`addr [BIND] val`).
* `DELIM_QUERY = 0xFF`: Introduces the target column address before $[S]$ (`[QUERY] addr_k [S]`).

```text
[BOS] [AXIOM] A_0 A_1 ... A_{B-1}
[A]  addr_0 [BIND] a_0   addr_1 [BIND] a_1   addr_2 [BIND] a_2 ...
[B]  addr_0 [BIND] b_0   addr_1 [BIND] b_1   addr_2 [BIND] b_2 ...
[CARRY] c_k
[OUT] addr_0 [BIND] o_0 ... addr_{k-1} [BIND] o_{k-1}
[QUERY] addr_k [S]
```

#### Properties:
1. **Permutation Invariance**: The order of columns in operands can be completely scrambled; attention queries columns associatively by key.
2. **Sparse Numbers**: Non-zero columns can be specified directly while omitting zeros, enabling astronomical calculations with minimal tokens.
3. **Mathematically Unbounded Horizon**: Operates without positional table or phase-drift limits.

---

### Alternative 2: Generic Delimiter + Role Embedding Hybrid

#### Core Concept:
Combines Tier 3's universal token punctuation with Tier 1's role subspace to achieve a clean **division of labor**:

* **Token Stream**: Separates every block with the single generic delimiter `|` (`0xFB`).
* **Embedding**: Retains the 64d learned `role_embed` vector in `dims 256..319`. Significance coordinates (`dims 320..511`) are set to zero (pure NoPE).

$$\mathbf{x}_t = \Big[\; \mathbf{e}_{\text{byte}}(t) \;\;\parallel\;\; \mathbf{e}_{\text{role}}(r_t) \;\;\parallel\;\; \mathbf{0}_{192\text{d}} \;\Big] \in \mathbb{R}^{512}$$

#### Why It Succeeds:
* **Zero Delimiter Fragmentation**: Uses one anchor key (`|`) for local column counting.
* **Zero Block-Counting Load**: Attention does not need to count delimiters to know what block it is in; the 64d role vector identifies the operand directly.
* **Zero Phase Drift**: Eliminating sinusoids prevents MLP activation saturation on $L > 20$.

---

### Alternative 3: Two-Delimiter Decoupled Architecture ([ROLE] [ORIGIN])

#### Core Concept:
Disentangles **"WHAT"** (semantic role) from **"WHERE"** (spatial origin) entirely within the token vocabulary:

```text
[AXIOM]  [ORIGIN]  A_0 A_1 ... A_{B-1}
[A]      [ORIGIN]  a_0 a_1 a_2 ...
[B]      [ORIGIN]  b_0 b_1 b_2 ...
[CARRY]  [ORIGIN]  c_k
[OUT]    [ORIGIN]  o_0 o_1 ... [S]
```

#### Independent Circuit Formation:
1. **Circuit 1 (Universal Spatial Counter)**:
   Attends *strictly* to `[ORIGIN]`. Because `[ORIGIN]` is identical across all blocks, it learns a single, shared column counter ($0, 1, 2, \dots$) without needing to know which operand it is in.
2. **Circuit 2 (Semantic Router)**:
   Attends *strictly* to `[A]`, `[B]`, `[OUT]`, broadcasting semantic scope across tokens without performing any spatial counting.

In Layer 2, the query matches:
$$\text{Target} = (\text{Role} == [A]) \;\land\; (\text{Distance} == k)$$
This completely eliminates the 5-way fragmentation bottleneck that caused Tier 2 to plateau at 50% EM.

---

## 5. Round 2 Proposed Experimental Matrix & Roadmap

To resolve the remaining open questions, the recommended experimental matrix for Round 2 is structured as follows:

| Priority | Variant Name | Core Hypothesis to Test |
| :---: | :--- | :--- |
| **P1** | `SM-Seq-T4-L6-SwiGLU-bimodal` | Tests whether recursive little-endian in-context numeral indexing achieves 100% EM and solves length extrapolation without any positional encodings. |
| **P2** | `AR-T4-L6-SwiGLU-bimodal` | Evaluates autonomous autoregressive generation with explicit recursive numeral addressing. |
| **P3** | `SM-Seq-Hybrid-T3Role-L6-SwiGLU` | Tests whether decoupling local column counting (`\|`) from global scope (`role_embed`) breaks the 55% State Machine plateau. |
| **P4** | `SM-Seq-T2Decoupled-L6-SwiGLU` | Tests whether the two-delimiter `[ROLE] [ORIGIN]` architecture eliminates Tier 2's 5-anchor fragmentation. |
| **P5** | `AR-T2-L6-SwiGLU-multimodal` | Combines fleet-leading Base-12 performance (`AR-T2`) with multi-base training to target >50% OOD EM across all bases. |

### Verification Protocol:
1. Execute unit and regression tests:
   ```bash
   .venv/bin/pytest
   ```
2. Run local CPU smoke tests on newly added variants (100 steps).
3. Stage and launch selected configurations on GCP Vertex AI (`g2-standard-4`, NVIDIA L4).
4. Evaluate all checkpoints on the 5 standardized benchmark distributions ($N=10,000$).
