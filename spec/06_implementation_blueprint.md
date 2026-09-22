# PyTorch Implementation Blueprint, Rule Protocols, and Engineering Guidelines

**Specification Document 06**  
**TransformAdder Project: Foundational Framework**

---

## 1. The Generic Rule Protocol

To ensure that the codebase scales seamlessly from addition to broader algorithmic and ARC-AGI rule-following tasks, data generation and oracle verification are structured around a generic Python `Rule` protocol:

```python
from typing import Protocol, Any, TypedDict

class RuleInstance(TypedDict):
    instance_id: str
    rule_family: str
    rule_params: dict[str, Any]   # e.g. {"base": 10, "axiom": [...], "alphabet_condition": "canonical_permuted"}
    inputs: dict[str, Any]        # e.g. {"a": [...], "b": [...]}
    steps: list[dict[str, Any]]   # Ground truth oracle transition steps
    expected_out: list[int]       # Final expected output tokens

class Rule(Protocol):
    """Generic protocol for procedural rule execution engines."""
    
    @property
    def rule_family(self) -> str:
        """Unique identifier for the rule family (e.g. 'addition', 'arc_grid')."""
        ...

    def sample_instance(self, seed: int, step_idx: int, p_prior: float = 0.5) -> RuleInstance:
        """Deterministically generates a problem instance with oracle ground truth."""
        ...

    def reference_step(self, state: Any, input_slice: Any, rule_params: dict[str, Any]) -> tuple[Any, int, int]:
        """Executes a single ground-truth transition delta: (state, input) -> (next_state, out, flag)."""
        ...

    def build_prompt(self, instance: RuleInstance, representation: str) -> Any:
        """Serializes the problem instance into input token sequences according to representation layout."""
        ...
```

### 1.1 Concrete Implementation: `AdditionRule`
The arithmetic engine is implemented as `AdditionRule(Rule)`:
* `rule_family = "addition"`.
* `rule_params` contains `base`, `axiom`, and `alphabet_condition`.
* `reference_step` computes integer addition with carry.
* Future tasks (e.g. `CellularAutomataRule`, `GridTransformRule`) implement this protocol directly, reusing the training loop, evaluation logging, and model architectures without modification.

---

## 2. Universal Transformer Architecture

All models share an identical, tightly controlled architectural footprint:

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
      └── Linear Head W_flag  (256 -> 2)              Conditioned on [S], [C], [F], [PROBE]
```

* **Layers ($L$)**: 6.
* **Model Dimension ($d_{\text{model}}$)**: 256.
* **Attention Heads ($n_{\text{heads}}$)**: 4 (head dimension $d_{\text{head}} = 64$).
* **Feed-Forward Dimension ($d_{\text{ff}}$)**: 1024 (SwiGLU expansion factor 4.0).
* **Dropout**: 0.0.
* **Parameter Count**: ~3.15M parameters under tied copy head; ~3.28M under multi-head projections.

---

## 3. Stratified Procedural Generator Contracts

```python
class ProceduralRuleGenerator:
    """Deterministic procedural generator with controlled prior dialing."""
    
    def __init__(self, seed: int, rule: Rule, p_prior: float = 0.5, curriculum: str = "standard"):
        self.seed = seed
        self.rule = rule
        self.p_prior = p_prior
        self.curriculum = curriculum

    def sample_alphabet(self, base: int, condition: str, rng: random.Random) -> list[int]:
        if condition == "canonical_order":
            return [ord('0') + i for i in range(base)]
        elif condition == "canonical_permuted":
            canonical = [ord('0') + i for i in range(base)]
            perm = canonical[:]
            while perm == canonical and base > 1:
                rng.shuffle(perm)
            return perm
        elif condition == "symbols_novel":
            return rng.sample(SIGMA_NOVEL_POOL, base)
        raise ValueError(f"Unknown alphabet condition: {condition}")
```

1. **Prior Dialing in Training**: During training, condition is sampled as `canonical_order` with probability $p_{\text{prior}}$, and `symbols_novel` with probability $1 - p_{\text{prior}}$.
2. **Carry Cascade Injection ($p_{\text{chain}} = 0.5$)**: Forces generating columns ($a_i + b_i \ge B$) followed by propagating columns ($a_j + b_j = B - 1$).
3. **Terminal Step Oversampling ($p_{\text{terminal}} = 0.2$)**: Alleviates halting flag class imbalance in harnessed training.

---

## 4. Evaluation Telemetry & Per-Instance Result Writer

In addition to logging summary metrics (`summary.json`), the evaluation suite writes fine-grained **per-instance records** to a JSONL stream (`instances.jsonl`):

```python
def write_instance_record(
    log_dir: str,
    record: dict[str, Any]
) -> None:
    """Appends an instance record containing continuous regressors and error classification."""
    path = os.path.join(log_dir, "instances.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
```

This captures:
* Token distance from rule definition to query.
* Intermediate entry counts.
* Exact error classification (`prior_intrusion`, `out_of_alphabet`, `procedural_error`, `syntax_malformation`).
* In-context probe accuracy.

---

## 5. Checkpoint Schema & Architectural Compatibility

All checkpoints store metadata according to the 7-dimensional design tensor:

```json
{
  "rule_family": "addition",
  "agency": "harnessed",
  "topology": "carry_first",
  "channel": "sequence",
  "representation": "addressed_kv",
  "curriculum": "standard",
  "p_prior": 0.5,
  "n_layers": 6,
  "d_model": 256,
  "n_heads": 4,
  "d_ff": 1024,
  "step": 100000
}
```

### 5.1 Backward-Compatible Checkpoint Resolver
To ensure existing model checkpoints can be evaluated without modification, model loaders implement a compact resolver:

```python
def resolve_legacy_config(cfg: dict) -> dict:
    """Maps legacy configuration metadata into canonical orthogonal coordinates."""
    if "agency" in cfg and "rule_family" in cfg:
        return cfg
    
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
    
    return {
        **cfg,
        "rule_family": cfg.get("rule_family", "addition"),
        "agency": agency,
        "topology": topology,
        "channel": channel,
        "representation": representation,
        "curriculum": curriculum,
        "p_prior": cfg.get("p_prior", 0.0),
    }
```
