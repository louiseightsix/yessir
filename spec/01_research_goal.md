# In-Context Rule Following, Parameter Binding, and Prior Suppression

**Specification Document 01**  
**TransformAdder Project: Foundational Framework**

---

## 1. Primary Research Goal: Neural Rule Execution

Modern artificial intelligence benchmarks—culminating in abstract reasoning challenges such as **ARC-AGI**—increasingly demand that neural networks behave not merely as associative pattern matchers, but as **faithful, in-context rule execution engines**. 

In conventional language pre-training, tasks are solved by retrieving statistical priors baked into the stationary weights of the model. In contrast, general intelligence requires:
1. **Dynamic Parameter Binding**: Reading explicit rules, definitions, or parameter mappings supplied in-context.
2. **Prior Suppression**: Faithfully adhering to the active in-context rule even when it directly contradicts strong statistical priors learned during training.
3. **Deterministic Execution**: Carrying out multi-step procedural transformations without error compounding, context drift, or spatial degradation.

This project investigates **the foundational mechanics of in-context rule following in transformers**. Rather than treating arithmetic as an end in itself, we treat base-conditioned addition as the **first concrete instance of a generalized rule-following and procedural induction harness**.

---

## 2. A Task-Independent Formalism for Rule Following

To ensure that the theoretical constructs, evaluation protocols, and code infrastructure developed here carry over directly to broader reasoning domains (such as 2D cellular automata, grid transformations, and ARC-AGI primitives), we formalise four universal concepts:

```
                            THE RULE-FOLLOWING ONTOLOGY
                            
  [RULE]         A parameterised computational procedure: P(input; Theta_rule) -> output
                   │
  [BINDING]      Supplying the parameters Theta_rule dynamically in the sequence context
                   │
  [PRIOR]        Competing default procedures or associations learned in model weights
                   │
  [EXECUTION]    Evaluating the bound procedure to produce the output trajectory
```

### 2.1 Universal Definitions
1. **Rule ($\mathcal{P}$)**: An algorithmic procedure that maps an input configuration to an output configuration, parameterised by a parameter tuple $\Theta_{\text{rule}}$:
   $$\mathcal{P}_{\Theta}: \mathcal{X} \to \mathcal{Y}$$
2. **Binding**: The process by which the sequence prompt supplies the concrete values of $\Theta_{\text{rule}}$ to the model at inference time. The model must bind these parameters dynamically within its attention activations rather than relying on static weights.
3. **Execution**: The step-by-step unrolling of the procedure $\mathcal{P}_{\Theta}$ over the inputs to generate the valid result or reasoning trace.
4. **Prior ($\pi_0$)**: Any default procedure, ordering, or association encoded within the model's weights through training frequency (e.g. the canonical sequence of decimal digits `'0' \to '1' \to '2' \dots`). In models trained strictly from scratch, there is zero pre-training; the only prior that exists is the empirical frequency prior induced by the training parameter $p_{\text{prior}}$. A prior competes with an in-context binding whenever the active rule contradicts this learned expectation.

### 2.2 The Progression of Rule-Following Complexity
Within this ontology, algorithmic tasks form a natural continuum toward ARC-AGI:
* **Level 1 (Current Investigation)**: The procedure $\mathcal{P}$ (column addition with carry) is fixed in the neural network weights; exactly **one parameter**—the alphabet successor relation $\mathcal{A} = [s_0, s_1, \dots, s_{B-1}]$—is dynamically bound in context per instance.
* **Level 2 (Multi-Parameter Binding)**: The procedure remains fixed in weights, but multiple parameters are bound in context (e.g. modular base, operation selector, carry direction, grid boundaries).
* **Level 3 (Symbolic Procedure Ingestion)**: The procedure itself is described symbolically in the prompt (e.g. explicit state-transition tables, cellular automaton update rules).
* **Level 4 (Rules with Stated Exceptions)**: The procedure includes explicit conditional overrides ("follow rule $X$ except when predicate $Y$ holds").
* **Level 5 (ARC-AGI Induction)**: The procedure and its parameters are not stated explicitly; they must be induced entirely from input-output demonstration pairs and then executed on a query input.

---

## 3. Central Research Questions

Framed as an inquiry into rule following, the experimental program investigates five core scientific questions:

### 3.1 Question 1: Supervisory Agency & Autonomous State Tracking
*Can a causal self-attention mechanism autonomously maintain rule execution state without external supervisory scaffolding?*
* **Inquiry**: In a harnessed State Machine (`A-Harnessed`), an external supervisor manages memory registers, advancing execution step-by-step. In an autonomous regime (`A-Autonomous`), the model must maintain its own execution pointer, carry memory, and termination condition in a single KV-cache. We ask whether autonomous execution is prone to context drift and whether self-attention can sustain an internal step counter across unseen sequence lengths.

### 3.2 Question 2: Intra-Step Causal Topology & Directional Asymmetry
*Does the factorization of multi-variable rule transitions determine computational feasibility?*
* **Inquiry**: Executing a column transition requires determining both a primary output (sum digit $o_k$) and an auxiliary state update (outgoing carry $c_{k+1}$). We investigate whether emitting the auxiliary state update first (**Carry-First**: monotonic thresholding) before the primary output (**Sum**: conditional subtraction) resolves the non-linear modular addition bottleneck across both harnessed micro-steps and autonomous scratchpads.

### 3.3 Question 3: Decoding Channels & Coordinate Independence
*Do static linear projection heads bind rules to absolute coordinates, and can tied sequence-space query triggers restore coordinate-free rule binding?*
* **Inquiry**: Static linear heads ($W_{\text{head}} \mathbf{h}$) project fixed spatial directions. In coordinate-free (NoPE) spaces, we test whether sequence-space query triggers (`[S]`, `[C]`, `[F]`) paired with a parameter-free tied copy head ($W_{\text{head}} = E^T$) allow the model to execute dynamic in-context pointer routing over bound parameters.

### 3.4 Question 4: Prior Suppression & The Rule-Prior Conflict Dial
*What determines whether a neural model obeys an in-context rule binding when it directly contradicts a deeply learned prior?*
* **Inquiry**: By varying the training frequency $p_{\text{prior}}$ of canonical numerical order vs. permuted orderings, we measure the **Prior-Override Compliance Curve**. When a model fails under prior conflict, does it fail randomly, or does it suffer from **Prior Intrusion** (emitting the digit expected under the canonical prior rather than the active binding)?

### 3.5 Question 5: Binding Retrieval vs. Procedural Execution
*When a model fails an algorithmic task, did it fail to bind the rule parameters, or did it fail the multi-step procedural execution?*
* **Inquiry**: By introducing a single-token **Successor Probe** ($s_i \to s_{i+1}$) answered from the exact same prompt context, we decouple parameter retrieval from multi-step calculation. Comparing probe accuracy directly against **per-step or per-digit prediction accuracy** on the same instances isolates parameter retrieval from procedural execution.

---

## 4. Operational Invariants

1. **Universal 1-Byte Vocabulary ($\mathcal{V}$)**: Strictly single-byte tokens ($|\mathcal{V}|=256$, `0x00`–`0xFF`). Sub-word tokenisation is forbidden; every rule parameter, delimiter, and numeral occupies exactly one token slot.
2. **Procedural Determinism (Zero Disk I/O)**: Data is synthesized on-the-fly as a pure mathematical function of the seed and step index.
3. **Strict Zero-Shot Evaluation**: Out-of-distribution evaluation is conducted with frozen weights, zero fine-tuning, and zero few-shot demonstrations.
4. **Full-Sequence Exact Match (EM) as Primary Metric**: A problem is correct if and only if every single step output and the halting token match the ground-truth reference trajectory.
5. **Stratified Prior Control**: Alphabets are explicitly categorised into neutral random symbols (`symbols_random`), canonical orders (`canonical_order`), and permuted canonical orders (`canonical_permuted`) to control prior strength.

---

## 5. Scope Boundaries: What This Investigation Cannot Answer

To ensure scientific clarity, we explicitly document the conceptual limits of this investigation:

1. **Binary Compliance Only**: Rules in this study are exact, formal, and unambiguous. This investigation does not address fuzzy, subjective, or conversational rule compliance (e.g. "be helpful and harmless").
2. **No Applicability Judgment**: The model is explicitly supplied with the active rule binding in its prompt; it is not required to infer *which* of several competing rules applies to a given input.
3. **Structural Computational Load vs. Semantic Nuance**: Failures observed in this study arise strictly from attention capacity limits, context distance, coordinate degradation, or prior intrusion—not from lexical ambiguity or natural language comprehension failures.
