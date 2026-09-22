# Execution Mechanics, Agency Contracts, and Causal Topologies

**Specification Document 02**  
**TransformAdder Project**

---

## 1. Overview & Formal Decomposition

In conventional sequence modeling, runtime execution harnesses and decoding strategies are frequently bundled together as monolithic "paradigms". This specification strictly decouples the execution mechanics into three orthogonal architectural dimensions:

1. **Supervisory Agency ($\mathcal{S}_{\text{agency}}$)**: The locus of state management and loop control flow.
2. **Intra-Step Causal Topology ($\mathcal{S}_{\text{topology}}$)**: The joint probability factorisation over column outputs ($o_k, c_{k+1}, \text{FLAG}_k$).
3. **Readout Decoding Channel ($\mathcal{S}_{\text{channel}}$)**: The parameterisation used to project hidden representations into token predictions.

```
                               THE EXECUTION MECHANICS TENSOR
                               
  Axis I: Supervisory Agency (Who maintains state memory and loop control?)
  ├── A-Harnessed:    External Python runtime maintains registers; model evaluates single transitions.
  └── A-Autonomous:   Zero runtime intervention; model unrolls full trace in its own KV-cache.

  Axis II: Intra-Step Causal Topology (What is the conditional dependency graph?)
  ├── T-Factored:     Order-free conditional independence: P(o|s) * P(c|s) * P(FLAG|s)
  ├── T-SumFirst:     Sequential chain: s -> o_k -> c_{k+1} -> FLAG_k
  └── T-CarryFirst:   Sequential chain: s -> c_{k+1} -> o_k -> FLAG_k

  Axis III: Readout Decoding Channel (How are predictions extracted?)
  ├── C-Weight:       Multi-head linear projections (W_sum, W_carry, W_flag) from tail vector.
  └── C-Sequence:     Tied parameter-free copy head (W_head = E^T) via sequence query triggers.
```

---

## 2. Axis I: Supervisory Agency ($\mathcal{S}_{\text{agency}}$)

The Supervisory Agency axis defines whether the model operates as a modular transition operator managed by an external program or as an autonomous agent maintaining its own scratchpad.

### 2.1 `A-Harnessed` (Supervised State Machine)
In the harnessed regime, the neural network acts strictly as a stateless column transition operator:
$$\Delta s_k = f_\theta(\text{Prompt}(a, b, \mathcal{A}, c_k, OUT_k))$$
An external supervisory program (the Python runtime harness) manages all memory registers and iteration logic:
1. **Register Maintenance**: The harness stores the current incoming carry $c_k \in \{s_0, s_1\}$ and the accumulated output sequence $OUT_k = [o_0, \dots, o_{k-1}]$ in external RAM.
2. **Prompt Virtualization**: At step $k$, the harness constructs a fresh prompt slice containing the problem definition, operands, current carry $c_k$, and historical output prefix $OUT_k$.
3. **Execution Control**: The harness queries the model, extracts predictions, updates $c_{k+1}$ and $OUT_{k+1}$, checks whether the predicted $\text{FLAG}_k == \texttt{FLAG\_FINISHED}$, and terminates the loop when complete.
4. **Error Isolation**: Because the harness formats the prompt at each step, an erroneous digit prediction at column $k$ does not corrupt the structural token alignment of column $k+1$.

### 2.2 `A-Autonomous` (Autoregressive Rollout)
In the autonomous regime, the transformer receives zero runtime assistance after step 0:
1. **Single Prompt Ingestion**: The model is presented with the problem prefix exactly once:
   $$\text{Prompt} = [\texttt{BOS}] \circ \mathcal{A} \circ a \circ b \circ [\texttt{START}]$$
2. **Autonomous Scratchpad Generation**: The model autoregressively generates its own reasoning trace into its unified KV-cache:
   $$x_t \sim P_\theta(x_t \mid x_{<t})$$
3. **Internal State Maintenance**: The model must track column indices, retain historical carries, avoid context drift, and autonomously decide when to emit $\texttt{FLAG\_FINISHED}$.
4. **Error Compounding**: Any mispredicted structural delimiter or erroneous carry permanently enters the attention history, providing a rigorous test of autonomous error recovery.

---

## 3. Axis II: Intra-Step Causal Topology ($\mathcal{S}_{\text{topology}}$)

Given the problem state $s_k$ at column $k$, the intra-step causal topology dictates the probabilistic factorisation of the step outputs: the sum digit $o_k$, the outgoing carry $c_{k+1}$, and the halting signal $\text{FLAG}_k$.

### 3.1 `T-Factored` (Order-Free Conditional Independence)
Under factored topology, all three outputs are conditionally independent given state $s_k$:
$$P(o_k, c_{k+1}, \text{FLAG}_k \mid s_k) = P(o_k \mid s_k) \cdot P(c_{k+1} \mid s_k) \cdot P(\text{FLAG}_k \mid s_k)$$
* **Mechanics**: Neither prediction conditions on the other; the model cannot use the value of the computed carry to assist in computing the sum, nor vice versa. Both must be resolved in parallel from the latent representation of state $s_k$.

### 3.2 `T-SumFirst` (Sum-First Causal Chain)
Under sum-first topology, the joint distribution is factorised by emitting the sum digit first:
$$P(o_k, c_{k+1}, \text{FLAG}_k \mid s_k) = P(o_k \mid s_k) \cdot P(c_{k+1} \mid s_k, o_k) \cdot P(\text{FLAG}_k \mid s_k, o_k, c_{k+1})$$
* **Computational Profile**: Emitting $o_k$ first requires the network to compute the modular residue:
  $$o_k = (a_k + b_k + c_k) \pmod B$$
  directly from $s_k$ without any intermediate scratchpad tokens. The outgoing carry $c_{k+1}$ is then predicted conditioning on both $s_k$ and the emitted $o_k$.

### 3.3 `T-CarryFirst` (Carry-First Causal Chain)
Under carry-first topology, the joint distribution is factorised by emitting the carry digit first:
$$P(o_k, c_{k+1}, \text{FLAG}_k \mid s_k) = P(c_{k+1} \mid s_k) \cdot P(o_k \mid s_k, c_{k+1}) \cdot P(\text{FLAG}_k \mid s_k, c_{k+1}, o_k)$$
* **Computational Profile**: Emitting $c_{k+1}$ first requires computing the monotonic threshold inequality:
  $$c_{k+1} = \begin{cases} s_1 & \text{if } a_k + b_k + c_k \ge B \\ s_0 & \text{otherwise} \end{cases}$$
  Linear self-attention can readily separate this monotonic half-space. Once $c_{k+1}$ is emitted into the sequence context, self-attention can condition on it directly, reducing the sum computation to a conditional linear subtraction:
  $$o_k = (a_k + b_k + c_k) - \text{val}(c_{k+1}) \cdot B$$
  This eliminates the non-linear modular addition bottleneck.

---

## 4. Axis III: Readout Decoding Channel ($\mathcal{S}_{\text{channel}}$)

The readout channel specifies how latent hidden vectors $\mathbf{h} \in \mathbb{R}^{d_{\text{model}}}$ are transformed into categorical probability distributions over the vocabulary.

### 4.1 `C-Weight` (Multi-Head Linear Projections)
* **Architecture**: The transformer evaluates the sequence forward pass up to the tail position $T$, obtaining a single hidden vector $\mathbf{h}_T \in \mathbb{R}^{d_{\text{model}}}$. Three separate parameterised linear heads project $\mathbf{h}_T$ into logits:
  $$\mathbf{z}_{\text{sum}} = W_{\text{sum}} \mathbf{h}_T \in \mathbb{R}^{|\mathcal{V}|}$$
  $$\mathbf{z}_{\text{carry}} = W_{\text{carry}} \mathbf{h}_T \in \mathbb{R}^{|\mathcal{V}|}$$
  $$\mathbf{z}_{\text{flag}} = W_{\text{flag}} \mathbf{h}_T \in \mathbb{R}^2$$
* **Constraint**: Static weight heads learn fixed spatial projections in the activation space. If continuous positional encodings are removed, $\mathbf{h}_T$ must encode multiple distinct categorical targets simultaneously without geometric coordinates to partition the subspace.

### 4.2 `C-Sequence` (Tied Copy Head via Query Triggers)
* **Architecture**: Logits are extracted exclusively through the shared input embedding matrix transposed as a parameter-free tied projection:
  $$W_{\text{head}} = E^T \in \mathbb{R}^{|\mathcal{V}| \times d_{\text{model}}}$$
  $$\mathbf{z}_t = E^T \mathbf{h}_t$$
* **Mechanics**: Instead of using dedicated linear projection matrices, the model conditions on distinct sequence query trigger tokens:
  - Query token `[S]` (`0xF9`) prompts the copy head to attend to and retrieve the sum digit $o_k$.
  - Query token `[C]` (`0xFA`) prompts the copy head to attend to and retrieve the carry digit $c_{k+1}$.
  - Query token `[F]` (`0xFB`) prompts the copy head to emit the halting flag.
* **Property**: Decouples prediction from static weights; the model solves column addition via dynamic in-context routing and pointer-style attention lookup.

---

## 5. Sequence Grammars & Training Contracts

The combination of Agency, Topology, and Channel defines the exact sequence grammars ingested and emitted by the models.

### 5.1 Harness-Driven Multi-Head Grammar (`SM-MultiHead`)
* **Tuple**: `⟨A-Harnessed, T-Factored, C-Weight⟩`
* **Input Sequence at Column $k$**:
  $$[\texttt{BOS}] \circ \text{Prefix}(\mathcal{A}, a, b) \circ c_k \circ OUT_k \circ [\texttt{S}]$$
* **Prediction**: Evaluated from hidden vector $\mathbf{h}_{\text{tail}}$ at position $[\texttt{S}]$ across $W_{\text{sum}}, W_{\text{carry}}, W_{\text{flag}}$.
* **Loss**:
  $$\mathcal{L} = \mathcal{L}_{\text{CE}}(\mathbf{z}_{\text{sum}}, o_k) + \mathcal{L}_{\text{CE}}(\mathbf{z}_{\text{carry}}, c_{k+1}) + \mathcal{L}_{\text{CE}}(\mathbf{z}_{\text{flag}}, \text{FLAG}_k)$$

### 5.2 Harness-Driven Branching Query Grammar (`SM-Fork`)
* **Tuple**: `⟨A-Harnessed, T-Factored, C-Sequence⟩`
* **Input Sequences at Column $k$**: Three parallel prompts differing only in the trailing query trigger:
  $$\text{Prompt}_{\text{sum}} = [\texttt{BOS}] \circ \text{Prefix}(\mathcal{A}, a, b) \circ c_k \circ OUT_k \circ [\texttt{S}]$$
  $$\text{Prompt}_{\text{carry}} = [\texttt{BOS}] \circ \text{Prefix}(\mathcal{A}, a, b) \circ c_k \circ OUT_k \circ [\texttt{C}]$$
  $$\text{Prompt}_{\text{flag}} = [\texttt{BOS}] \circ \text{Prefix}(\mathcal{A}, a, b) \circ c_k \circ OUT_k \circ [\texttt{F}]$$
* **Prediction**: Next-token prediction at the trailing trigger position via tied head $W_{\text{head}} = E^T$.
* **Loss**:
  $$\mathcal{L} = \mathcal{L}_{\text{CE}}(\mathbf{z}_{[\texttt{S}]}, o_k) + \mathcal{L}_{\text{CE}}(\mathbf{z}_{[\texttt{C}]}, c_{k+1}) + \mathcal{L}_{\text{CE}}(\mathbf{z}_{[\texttt{F}]}, \text{FLAG}_k)$$

### 5.3 Harness-Driven Micro-Sequential Grammars (`SM-Seq` & `SM-SeqRev`)
* **`SM-Seq` (`⟨A-Harnessed, T-SumFirst, C-Sequence⟩`)**:
  - Sequence: $[\texttt{BOS}] \circ \text{Prefix} \circ c_k \circ OUT_k \circ [\texttt{S}] \circ \mathbf{o_k} \circ [\texttt{C}] \circ \mathbf{c_{k+1}} \circ [\texttt{F}]$
  - Target labels aligned to: $o_k$ at $[\texttt{S}]$, $c_{k+1}$ at $[\texttt{C}]$, and $\text{FLAG}_k$ at $[\texttt{F}]$.
* **`SM-SeqRev` (`⟨A-Harnessed, T-CarryFirst, C-Sequence⟩`)**:
  - Sequence: $[\texttt{BOS}] \circ \text{Prefix} \circ c_k \circ OUT_k \circ [\texttt{C}] \circ \mathbf{c_{k+1}} \circ [\texttt{S}] \circ \mathbf{o_k} \circ [\texttt{F}]$
  - Target labels aligned to: $c_{k+1}$ at $[\texttt{C}]$, $o_k$ at $[\texttt{S}]$, and $\text{FLAG}_k$ at $[\texttt{F}]$.

### 5.4 Autonomous Scratchpad Grammars (`AR-SumFirst` & `AR-CarryFirst`)
* **Prompt (Single Ingestion)**:
  $$\text{Prompt} = [\texttt{BOS}] \circ \text{Prefix}(\mathcal{A}, a, b) \circ [\texttt{START}]$$
* **Canonical `AR-SumFirst` Trace**:
  $$\text{Trace} = \left( [\texttt{S}] \circ o_k \circ [\texttt{C}] \circ c_{k+1} \right)_{k=0}^K \circ [\texttt{FINISHED}]$$
* **Proposed `AR-CarryFirst` Trace**:
  $$\text{Trace} = \left( [\texttt{C}] \circ c_{k+1} \circ [\texttt{S}] \circ o_k \right)_{k=0}^K \circ [\texttt{FINISHED}]$$
* **Teacher Forcing Contract**: Every token in $\text{Trace}$ carries cross-entropy loss, including the structural markers `[S]`, `[C]`, and `[FINISHED]`. The prompt tokens carry zero loss (`IGNORE_LABEL = -100`).
