# Execution Mechanics: State Machine, Agency, Topology and Readout

**Specification Document 02**

---

## 1. Column Addition as a State Machine

One column of addition is a deterministic transition. With the axiom $\mathcal{A}$ as the in-context parameter:

* **State**: $s_k = \langle c_k, k \rangle$, the incoming carry $c_k \in \{s_0, s_1\}$ and the column index $k$.
* **Input**: $i_k = \langle a_k, b_k \rangle$, the operand digits at column $k$.
* **Transition**:
  $$v = \text{val}(a_k) + \text{val}(b_k) + \text{val}(c_k), \qquad c_{k+1} = \mathcal{A}[\lfloor v / B \rfloor], \qquad o_k = \mathcal{A}[v \bmod B]$$
  Since $v \le 2B - 1$, the carry is a single threshold ($v \ge B$) and the sum digit a single conditional subtraction ($v - B \cdot [v \ge B]$).
* **Halting**: see §3.

```
                         In-context axiom A
                                │
                                ▼
             i_k = (a_k, b_k) ─►┌───────────┐─► o_k
                                │   delta   │
             s_k = (c_k, k)  ──►└───────────┘─► s_{k+1} = (c_{k+1}, k+1)
                                │
                                ▼
                         FLAG_k (halting)
```

The reference oracle (`core/reference.py`) computes the ground-truth trajectory for supervision and scoring.

---

## 2. Shared Invariants

All mechanics, in every tier, obey:

1. **Axiom-defined numerals**: $\text{val}(s_i) = i$ for $s_i \in \mathcal{A}$; nothing else assigns value to a byte.
2. **LSD-first order**: $a$, $b$ and $OUT$ are written least-significant digit first, so column $k$ is always at offset $+k$ from the start of its block, independent of operand length. (In MSD-first order, locating column $k$ would require first measuring the operand's length, a difficulty unrelated to addition.)
3. **Missing operand digits are zero**: operands are unpadded; for $k \ge |a|$, $a_k = s_0$ (likewise $b$). This is *learned*: attention cannot attend to "nothing", so the model must detect absence and substitute 0, using `[BOS]` as a null key.
4. **Initial carry**: $c_0 = s_0$.
5. **`[BOS]` origin**: every sequence starts with `[BOS]`. It is a constant attention sink / null key, and the fixed first token that NoPE models count from under causal attention.
6. **Unconstrained decoding**: decoding is plain greedy $\arg\max$ over all 256 logits (2 for the `SM-MH` flag head). Logits are never masked to the axiom or to the grammatically expected token class; emitting an out-of-axiom byte or a misplaced control token is a model error.

---

## 3. Halting Contract

$$\text{FLAG}_k = \begin{cases} \texttt{FLAG\_FINISHED} & \text{if } k \ge \max(|a|, |b|) - 1 \text{ and } c_{k+1} = s_0 \\ \texttt{FLAG\_CONTINUE} & \text{otherwise} \end{cases}$$

* **Terminal step**: column $K$ emits $o_K$ with $c_{K+1} = s_0$ and `FLAG_FINISHED`.
* **Final carry column**: if the last operand column carries out, one more step runs with $a_k = b_k = s_0$, emitting $o_K = s_1$, $c_{K+1} = s_0$, `FLAG_FINISHED`.
* **No dummy step**: the harness appends $o_K$ and stops.

---

## 4. Axis I: Supervisory Agency ($\mathcal{S}_{\text{agency}}$)

### 4.1 `A-Harnessed` (state machine, `SM-*`)
The network approximates one transition:
$$(\hat{o}_k, \hat{c}_{k+1}, \widehat{\text{FLAG}}_k) = f_\theta(\text{Prompt}(\mathcal{A}, a, b, c_k, OUT_k))$$
An external Python harness runs the loop:
1. **Registers**: it stores $c_k$ and $OUT_k$ outside the model.
2. **Fresh prompt per step**: at step $k$ it builds a new prompt containing the axiom, operands, current carry and output so far.
3. **Control**: it reads the model's predictions, appends $o_k$, re-injects $c_{k+1}$, and stops on `FLAG_FINISHED` (or a step limit).
4. **Zero arithmetic**: the harness never computes anything itself; it only copies the model's outputs back in.
5. **Fault isolation**: an error at step $k$ does not misalign the token structure of step $k+1$.

### 4.2 `A-Autonomous` (autoregressive trace, `AR-*`)
The model sees one prompt, $[\texttt{BOS}] \circ \mathcal{A} \circ a \circ b \circ [\texttt{START}]$, then generates the whole trace greedily into its own KV-cache. It must track the column index, carry and halting itself. Errors stay in context and compound. The harness only extracts the $o_k$ tokens from the finished trace; it does no arithmetic and no re-injection. Because $c_{k+1}$ is emitted right before column $k+1$ begins, the carry is always one step back in context.

---

## 5. Axis II: Intra-Step Causal Topology ($\mathcal{S}_{\text{topology}}$)

How the coupled outputs of one column ($o_k$, $c_{k+1}$, $\text{FLAG}_k$) are factorised.

```
  T-Factored (order-free)      T-SumFirst                 T-CarryFirst
         s_k                      s_k                         s_k
        ┌─┴─┐                      │                           │
        ▼   ▼                      ▼                           ▼
      o_k  c_{k+1}                o_k                       c_{k+1}
                                   │                           │
                                   ▼                           ▼
                                c_{k+1}                       o_k
```

* **`T-Factored`**: $P(o_k, c_{k+1}, \text{FLAG}_k \mid s_k) = P(o_k \mid s_k)\,P(c_{k+1} \mid s_k)\,P(\text{FLAG}_k \mid s_k)$. Both values are resolved at once from the representation of $s_k$.
* **`T-SumFirst`**: $P(o_k \mid s_k)\,P(c_{k+1} \mid s_k, o_k)$. The modular residue must be computed directly, with no intermediate token.
* **`T-CarryFirst`**: $P(c_{k+1} \mid s_k)\,P(o_k \mid s_k, c_{k+1})$. The carry is a monotone threshold ($v \ge B$); given the carry, the sum digit is a conditional subtraction.

---

## 6. Axis III: Readout Channel ($\mathcal{S}_{\text{channel}}$)

* **`C-Weight`** (multi-head linear projections): one hidden vector $\mathbf{h}_T$ at the sequence tail feeds separate heads $W_{\text{sum}}$, $W_{\text{carry}}$, $W_{\text{flag}}$. Static heads learn fixed directions in hidden space; without positional coordinates to organise that space, resolving several categorical outputs from one vector may be hard. A new query type (e.g. the probe) needs a new head.
* **`C-Sequence`** (tied copy head with query triggers): logits come from the transposed input embedding, $W_{\text{head}} = E^\top$ (in practice the symbol slice of a gain-free final RMSNorm; [`06`](06_implementation.md) §2). Trigger tokens `[S]`, `[C]`, `[F]`, `[PROBE]` in the sequence say what to retrieve; attention then copies the right symbol from context. New queries need no new parameters.

---

## 7. Sequence Grammars and Loss Contracts

In grammars below, $\Theta = $ the axiom block and $\mathcal{I} = $ the operand blocks; their exact token layout depends on the representation tier ([`03`](03_representations.md) §3). $\mathbf{bold}$ tokens are model-emitted targets.

### 7.1 Harnessed grammars (one prompt per column)
* **`SM-MultiHead`** ⟨Harnessed, Factored, Weight⟩: $[\texttt{BOS}] \circ \Theta \circ \mathcal{I} \circ c_k \circ OUT_k \circ [\texttt{S}]$; targets read from the hidden state at `[S]` through $W_{\text{sum}}$, $W_{\text{carry}}$, $W_{\text{flag}}$. One forward pass per column.
* **`SM-Fork`** ⟨Harnessed, Factored, Sequence⟩: three parallel prompts from the same base, ending in `[S]`, `[C]` and `[F]`; each target is predicted at its trailing trigger through the tied head. No emission order.
* **`SM-Seq`** ⟨Harnessed, SumFirst, Sequence⟩: $[\texttt{BOS}] \circ \Theta \circ \mathcal{I} \circ c_k \circ OUT_k \circ [\texttt{S}] \circ \mathbf{o_k} \circ [\texttt{C}] \circ \mathbf{c_{k+1}} \circ [\texttt{F}] \circ \mathbf{FLAG_k}$. Three micro-steps per column; triggers are harness-inserted.
* **`SM-SeqRev`** ⟨Harnessed, CarryFirst, Sequence⟩: as `SM-Seq` with $[\texttt{C}] \circ \mathbf{c_{k+1}}$ before $[\texttt{S}] \circ \mathbf{o_k}$.

### 7.2 Autonomous grammars (one prompt per problem)
* **Prompt**: $[\texttt{BOS}] \circ \Theta \circ \mathcal{I} \circ [\texttt{START}]$
* **`AR-SumFirst` trace**: $\left( [\texttt{S}] \circ o_k \circ [\texttt{C}] \circ c_{k+1} \right)_{k=0}^{K} \circ [\texttt{FINISHED}]$
* **`AR-CarryFirst` trace**: $\left( [\texttt{C}] \circ c_{k+1} \circ [\texttt{S}] \circ o_k \right)_{k=0}^{K} \circ [\texttt{FINISHED}]$
* The choice of `[S]`/`[C]` vs. `[FINISHED]` after each column plays the role of $\text{FLAG}_k$.

### 7.3 Loss contract
Only model-emitted positions carry loss: the tail trigger for `SM-Fork`; $o_k$, $c_{k+1}$, $\text{FLAG}_k$ for `SM-Seq` / `SM-SeqRev`; every trace token after `[START]` for `AR`. Prompt tokens, harness-inserted triggers and padding are labelled `IGNORE_LABEL = -100`. Loss formulas are in [`06`](06_implementation.md) §4.
