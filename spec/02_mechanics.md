# Generic Rule Execution Mechanics, State Machines, and Causal Topologies

**Specification Document 02**  
**TransformAdder Project: Foundational Framework**

---

## 1. Generalized State Machine Formulation

To ensure that the execution harness, agency contracts, and causal topologies defined here serve as a general foundation for sequential rule following, we formulate the execution engine as an abstract **parameterized discrete-state transition system**:

$$\mathcal{M}_{\text{rule}} = \langle \mathcal{S}, \mathcal{I}, \mathcal{O}, \Theta_{\text{rule}}, \delta, \mathcal{H}, \mathcal{O}_{\text{ref}} \rangle$$

where:
1. $\mathcal{S}$: The internal execution state space (e.g. step counters, active registers, memory cells).
2. $\mathcal{I}$: The input stream or environment configuration.
3. $\mathcal{O}$: The output symbol alphabet emitted by the rule.
4. $\Theta_{\text{rule}}$: The rule parameter binding supplied in context (e.g. symbol successor tables, transition graphs, grid dimensions).
5. $\delta: \mathcal{S} \times \mathcal{I} \times \Theta_{\text{rule}} \to \mathcal{S} \times \mathcal{O}$: The deterministic single-step transition operator.
6. $\mathcal{H}: \mathcal{S} \to \{\texttt{FLAG\_CONTINUE}, \texttt{FLAG\_FINISHED}\}$: The halting predicate determining termination.
7. $\mathcal{O}_{\text{ref}}$: The reference oracle providing ground-truth execution trajectories for supervision and evaluation.

```
                         GENERIC STATE TRANSITION OPERATOR
                         
                         In-Context Binding Theta_rule
                                      │
                                      ▼
             Input i_k ───────►┌─────────────┐───────► Output o_k
                               │  Delta(s_k) │
             State s_k ───────►└─────────────┘───────► State s_{k+1}
                                      │
                                      ▼
                        Halting Predicate H(s_{k+1})
```

### 1.1 Concrete Instantiation: Column Addition
Base-conditioned addition is a specific instantiation of this generic automaton:
* **State**: $s_k = \langle c_k, k \rangle$, where $c_k \in \{s_0, s_1\}$ is the carry register and $k \in \mathbb{N}$ is the active column index.
* **Input**: $i_k = \langle a_k, b_k \rangle$, the pair of operand digits at column $k$.
* **Parameter Binding**: $\Theta_{\text{rule}} = \mathcal{A} = [s_0, \dots, s_{B-1}]$, the ordered alphabet defining the radix $B$ and digit values $\text{val}(s_j) = j$.
* **Transition Operator ($\delta$)**:
  $$\text{val}_{\text{sum}} = \text{val}(a_k) + \text{val}(b_k) + \text{val}(c_k)$$
  $$c_{k+1} = \mathcal{A}[\lfloor \text{val}_{\text{sum}} / B \rfloor]$$
  $$o_k = \mathcal{A}[\text{val}_{\text{sum}} \pmod B]$$
  $$\delta(s_k, i_k; \mathcal{A}) = (\langle c_{k+1}, k+1 \rangle, o_k)$$
* **Halting Predicate ($\mathcal{H}$)**: Returns $\texttt{FLAG\_FINISHED}$ iff $k \ge \max(|a|, |b|) - 1$ and $c_{k+1} == s_0$.

Any future rule family (e.g. 1D string rewrites, grid flood-fill, ARC object translation) implements this exact same interface $\delta$, inheriting the execution mechanics detailed below without architectural modification.

---

## 2. Axis I: Supervisory Agency ($\mathcal{S}_{\text{agency}}$)

The Supervisory Agency axis defines whether the loop iteration and state registers are maintained externally by a program or internally by the model's self-attention.

### 2.1 `A-Harnessed` (Supervised State Machine)
In the harnessed regime, the neural network acts strictly as a local transition operator approximating $\delta$:
$$(\hat{s}_{k+1}, \hat{o}_k, \widehat{\text{FLAG}}_k) = f_\theta(\text{Prompt}(i_k, s_k, \Theta_{\text{rule}}, OUT_k))$$
An external supervisor (the Python runtime harness) manages the execution loop:
1. **Register Maintenance**: The harness stores state $s_k$ and output history $OUT_k$ in external memory.
2. **Context Virtualization**: At step $k$, the harness constructs a fresh prompt slice presenting the input context, the active rule binding $\Theta_{\text{rule}}$, and the current state $s_k$.
3. **Execution Control**: The harness evaluates predictions, checks the halting predicate $\mathcal{H}$, and terminates the loop when complete.
4. **Fault Isolation**: An execution error at step $k$ does not desynchronize the structural token alignment of step $k+1$.

### 2.2 `A-Autonomous` (Autoregressive Rollout)
In the autonomous regime, the transformer operates with zero external assistance after step 0:
1. **Single Prompt Ingestion**: The model ingests the initial configuration once:
   $$\text{Prompt} = [\texttt{BOS}] \circ \Theta_{\text{rule}} \circ \mathcal{I} \circ [\texttt{START}]$$
2. **Autonomous Trajectory Unrolling**: The model unrolls its own execution trajectory autoregressively into its KV-cache:
   $$x_t \sim P_\theta(x_t \mid x_{<t})$$
3. **Internal State Maintenance**: The model must track its own step index, update internal registers, and decide when to emit $\texttt{FLAG\_FINISHED}$.
4. **Compounding Dynamics**: Any execution error or structural hallucination permanently enters the attention history, providing a true test of autonomous procedural stability.

---

## 3. Axis II: Intra-Step Causal Topology ($\mathcal{S}_{\text{topology}}$)

When a transition operator produces multiple coupled variables (e.g. an auxiliary state update $c_{k+1}$ and a primary output $o_k$), the causal topology specifies the probabilistic factorisation across those variables.

```
                           INTRA-STEP CAUSAL TOPOLOGIES
                           
  T-Factored (Order-Free)     T-SumFirst (Output First)    T-CarryFirst (State Update First)
        s_k                             s_k                            s_k
       ┌─┴─┐                             │                              │
       ▼   ▼                             ▼                              ▼
      o_k c_{k+1}                       o_k                          c_{k+1}
                                         │                              │
                                         ▼                              ▼
                                      c_{k+1}                          o_k
```

### 3.1 `T-Factored` (Order-Free Conditional Independence)
All step outputs are conditionally independent given the current state $s_k$:
$$P(o_k, c_{k+1}, \text{FLAG}_k \mid s_k) = P(o_k \mid s_k) \cdot P(c_{k+1} \mid s_k) \cdot P(\text{FLAG}_k \mid s_k)$$
Neither variable conditions on the other; the network must resolve both the primary output and the auxiliary state update simultaneously from its latent representation of $s_k$.

### 3.2 `T-SumFirst` (Primary Output First)
Emits the primary output first, then conditions on it to update the internal state:
$$P(o_k, c_{k+1} \mid s_k) = P(o_k \mid s_k) \cdot P(c_{k+1} \mid s_k, o_k)$$
* **Computational Profile**: Emitting $o_k$ first requires computing the non-linear modular residue directly from $s_k$ with zero scratchpad tokens.

### 3.3 `T-CarryFirst` (State Update First)
Emits the auxiliary state update first, then conditions on it to emit the primary output:
$$P(o_k, c_{k+1} \mid s_k) = P(c_{k+1} \mid s_k) \cdot P(o_k \mid s_k, c_{k+1})$$
* **Computational Profile**: Updating state $c_{k+1}$ first requires only a monotonic threshold decision ($a_k + b_k + c_k \ge B$), readily linearly separated by self-attention. Conditioned on $c_{k+1}$, computing the primary output $o_k$ simplifies to conditional linear subtraction.

---

## 4. Axis III: Readout Decoding Channel ($\mathcal{S}_{\text{channel}}$)

The readout channel specifies how hidden vectors $\mathbf{h} \in \mathbb{R}^{d_{\text{model}}}$ are converted into categorical predictions.

### 4.1 `C-Weight` (Multi-Head Linear Projections)
* Evaluates a single hidden vector $\mathbf{h}_T$ at the sequence tail through dedicated linear projection matrices:
  $$\mathbf{z}_{\text{out}} = W_{\text{out}} \mathbf{h}_T, \quad \mathbf{z}_{\text{state}} = W_{\text{state}} \mathbf{h}_T, \quad \mathbf{z}_{\text{flag}} = W_{\text{flag}} \mathbf{h}_T$$
* **Constraint**: Static weight heads learn fixed spatial projections. Without continuous positional coordinates to structure the hidden space, static linear projections struggle to resolve multiple distinct categorical outputs simultaneously.

### 4.2 `C-Sequence` (Tied Copy Head via Query Triggers)
* Logits are generated exclusively via the transposed input embedding matrix:
  $$W_{\text{head}} = E^T \in \mathbb{R}^{|\mathcal{V}| \times d_{\text{model}}}$$
* Dedicated query trigger tokens (`[S]`, `[C]`, `[F]`, `[PROBE]`) are appended to the sequence. Conditioned on the trigger token, the model routes attention dynamically to retrieve the appropriate value from context.
* **Property**: Decouples prediction from static weights, enabling dynamic in-context pointer routing over bound parameters. Because decoding is driven by sequence triggers, `C-Sequence` architectures natively support arbitrary auxiliary diagnostic queries (such as the single-token binding probe `[PROBE]`), whereas `C-Weight` architectures require adding separate parameterised linear heads for every new query type.

---

## 5. Sequence Grammars & Training Contracts

### 5.1 Harness-Driven Grammars
* **`SM-MultiHead` (`⟨A-Harnessed, T-Factored, C-Weight⟩`)**:
  - Input: $[\texttt{BOS}] \circ \Theta_{\text{rule}} \circ \mathcal{I} \circ s_k \circ OUT_k \circ [\texttt{S}]$
  - Target labels: Evaluated from hidden state at $[\texttt{S}]$ across $W_{\text{sum}}, W_{\text{carry}}, W_{\text{flag}}$.
* **`SM-Fork` (`⟨A-Harnessed, T-Factored, C-Sequence⟩`)**:
  - Prompts: Three parallel sequences terminating in `[S]`, `[C]`, and `[F]` respectively.
  - Target labels: Aligned to the trailing trigger token position via tied head $W_{\text{head}} = E^T$.
* **`SM-Seq` (`⟨A-Harnessed, T-SumFirst, C-Sequence⟩`)**:
  - Sequence: $[\texttt{BOS}] \circ \Theta_{\text{rule}} \circ \mathcal{I} \circ s_k \circ OUT_k \circ [\texttt{S}] \circ \mathbf{o_k} \circ [\texttt{C}] \circ \mathbf{c_{k+1}} \circ [\texttt{F}]$
* **`SM-SeqRev` (`⟨A-Harnessed, T-CarryFirst, C-Sequence⟩`)**:
  - Sequence: $[\texttt{BOS}] \circ \Theta_{\text{rule}} \circ \mathcal{I} \circ s_k \circ OUT_k \circ [\texttt{C}] \circ \mathbf{c_{k+1}} \circ [\texttt{S}] \circ \mathbf{o_k} \circ [\texttt{F}]$

### 5.2 Autonomous Scratchpad Grammars
* **Prompt**: $[\texttt{BOS}] \circ \Theta_{\text{rule}} \circ \mathcal{I} \circ [\texttt{START}]$
* **Canonical `AR-SumFirst` Trace**:
  $$\text{Trace} = \left( [\texttt{S}] \circ o_k \circ [\texttt{C}] \circ c_{k+1} \right)_{k=0}^K \circ [\texttt{FINISHED}]$$
* **Proposed `AR-CarryFirst` Trace**:
  $$\text{Trace} = \left( [\texttt{C}] \circ c_{k+1} \circ [\texttt{S}] \circ o_k \right)_{k=0}^K \circ [\texttt{FINISHED}]$$
* **Loss Contract**: All tokens in $\text{Trace}$ carry cross-entropy loss. Prompt tokens are ignored (`IGNORE_LABEL = -100`).
