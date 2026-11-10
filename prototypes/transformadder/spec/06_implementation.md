# Implementation

**Specification Document 06**

Pure PyTorch; the only runtime dependency is `torch`.

---

## 1. Code Layout

```
prototypes/transformadder/       the prototype (Python package prototypes.transformadder)
  cli.py                        CLI and Vertex AI entrypoint (python -m prototypes.transformadder.cli)
  run.sh                        sets up the venv and launches the configurator wizard
  core/
    constants.py                byte vocabulary, dimensions (mirrors spec/README.md §2)
    blocks.py                   RMSNorm, SwiGLU / GELU FFN, SDPA attention, Pre-LN TransformerBlock
    training.py                 seeding, AdamW parameter groups, warm-up + cosine schedule
    rule.py                     Rule protocol, AdditionRule, alphabet sampling
    reference.py                ground-truth oracle (reference_add, traces)
    sequences.py                prompt / trace builders for every paradigm and tier
    data.py                     ProceduralArithmeticGenerator
    embedding.py                tier-dependent input embedding
    heads.py                    multi-head projections and tied copy head
    model.py                    TransformAdder model (+ from_axes)
  harnesses/
    state_machine.py            SM-MH / SM-Seq / SM-SeqRev / SM-Fork drivers
    autoregressive.py           AR greedy rollout
  utils/
    runner.py                   Variant, training loop, losses
    evaluation.py               evaluation cells and metrics
    telemetry.py                error taxonomy, per-instance records
    logging.py                  JSONL MetricsLogger
    stepper.py                  interactive step-through viewer (--interactive)
  scripts/
    configure_experiment.py     interactive wizard; writes results/<run>/
    summarize_results.py        leaderboard over results/
  tests/
  spec/                         this specification
  notes/                        dated analysis write-ups from training rounds
  results/<run>/                config.json, run.py, run.sh, submit_vertex.sh,
                                metrics.jsonl, summary.json, instances.jsonl
```

---

## 2. Model Architecture

```
  token ids ─► embedding (symbol one-hot slice + tier-dependent role / position subspace)
                 │
          Pre-LN TransformerBlock × N_layers
            ├─ RMSNorm ─► multi-head causal self-attention (SDPA) ─► residual
            └─ RMSNorm ─► SwiGLU (or GELU) FFN ─────────────────────► residual
                 │
          final RMSNorm
                 │
     ┌───────────┴─────────────────┐
     ▼                             ▼
  C-Weight (SM-MH)              C-Sequence (SM-Seq / SeqRev / Fork, AR)
  W_sum, W_carry (→256),        tied copy: gain-free RMSNorm_0(h)[:256]
  W_flag (→2)                   read at [S] / [C] / [F] / [PROBE]
```

| Hyperparameter | Value | Notes |
| :--- | :--- | :--- |
| $d_{\text{model}}$ | 512 | 256 symbol subspace + 256 positional / state subspace |
| Heads | 8 | $d_{\text{head}} = 64$ |
| $d_{\text{ffn}}$ | 1376 (SwiGLU) / 2048 (GELU) | parameter-matched |
| Layers | 6 (default) | 2, 3 for depth ablations |
| Normalisation | RMSNorm, Pre-LN | learnable gain in blocks; gain-free final norm before the tied slice |
| Role / significance | 64-d role, 192-d sinusoidal significance | T1 (and role table in T4) |
| Dropout | 0 | |

---

## 3. Rule Protocol and Data Generation

`core/rule.py` defines a small `Rule` protocol (`sample_instance`, `reference_step`, `build_prompt`) returning `RuleInstance` records (`rule_params`: base, axiom, alphabet condition; `inputs`; oracle `steps`; `expected_out`). `AdditionRule` implements it. Alphabets are drawn by condition: `canonical_order` (`'0'`..`'9'`, `'A'`, `'B'`), `canonical_permuted` (a non-identity shuffle of those), or `symbols_random` (`rng.sample(SIGMA_POOL, B)`).

`ProceduralArithmeticGenerator` (`core/data.py`) produces training batches deterministically from (seed, step):
1. **Prior dial**: `canonical_order` with probability $p_{\text{prior}}$, else `symbols_random` from all of $\Sigma$.
2. **Probe queries**: with probability $p_{\text{probe}}$, `[PROBE] s_i → s_{i+1}` rows are added (sequence-channel models).
3. **Carry-chain injection** ($p_{\text{chain}} = 0.25$): builds a generating column ($a_i + b_i \ge B$) followed by propagating columns ($a_j + b_j = B - 1$), since random digits rarely produce long chains.
4. **Terminal-step oversampling** ($p_{\text{terminal}} = 0.2$): for SM training, samples the final column more often, to counter the CONTINUE / FINISHED imbalance.
5. **Training rows**: SM paradigms train on single step transitions; AR trains on full traces.

---

## 4. Losses

* **`SM-MH`**: $\mathcal{L} = \mathcal{L}_{\text{CE}}(\hat{y}_{\text{sum}}, o_k) + \mathcal{L}_{\text{CE}}(\hat{y}_{\text{carry}}, c_{k+1}) + 0.5\,\mathcal{L}_{\text{CE}}(\hat{y}_{\text{flag}}, \text{FLAG}_k)$
* **Sequence models**: masked causal cross-entropy over model-emitted positions only ([`02`](02_mechanics.md) §7.3), $\mathcal{L} = -\frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \log p(y_t \mid y_{<t})$; all other labels are `-100`.

---

## 5. Training Hyperparameters

| Setting | Value |
| :--- | :--- |
| Optimiser | AdamW, $\beta = (0.9, 0.95)$, $\epsilon = 10^{-8}$ |
| Peak LR | $3 \times 10^{-4}$ |
| Schedule | 1,000-step linear warm-up, cosine decay to 10% of peak |
| Weight decay | 0.1 on weight matrices; 0 on norm gains and role embeddings |
| Gradient clipping | global norm 1.0 |
| Batch size | 256 (SM: step transitions; AR: full traces) |
| Gradient accumulation | 1 by default; 2 for T6/T7, 8 for `SM-Fork` on T6/T7 (memory) |
| Steps | 100,000 (some runs extended to 500,000) |
| Precision | bfloat16 autocast; float32 weights and optimiser state |
| Initialisation | $\mathcal{N}(0, 0.02^2)$; residual output projections scaled by $1/\sqrt{2 N_{\text{layers}}}$; role embedding rows unit norm |
| Seeds | one seed drives both initialisation and the data generator; same seed ⇒ identical data, and repeatable training on a fixed hardware / software stack |

---

## 6. Run Artefacts

Each run directory (`prototypes/transformadder/results/<run>/`, or a GCS path on Vertex AI) contains:
* `config.json`, `run.py`, `run.sh`, `submit_vertex.sh`: written by the configurator; `run.sh` reproduces the run locally, `submit_vertex.sh` builds an sdist of the repo and submits it as a Vertex AI custom job running `prototypes.transformadder.cli`.
* `metrics.jsonl`: one line per evaluation ([`05`](05_evaluation_and_metrics.md) §3).
* `summary.json`: final evaluation.
* `instances.jsonl`: per-instance telemetry ([`05`](05_evaluation_and_metrics.md) §6).

No model checkpoints are saved.

*Historical note*: run directories created before the yessir restructure contain `run.py` / `submit_vertex.sh` files that import the old top-level `transformadder` package. They are kept as a record of how each run was launched; to re-run one, use the configurator with the same flags.

---

## 7. Known Gaps Between Spec and Code

* **Axis flags**: the CLI's `--agency autonomous|harnessed`, `--topology unified|pipelined|branching`, `--channel sequence|weight` and `--representation` use older names, and `REPRESENTATION_TO_TIER` (`core/embedding.py`) maps only four representation names to tiers 1, 2, 3, 6, which does not match the layout table in [`03`](03_representations.md) §3.1 (e.g. it maps `role_embed` to tier 1). Use `--paradigm` and `--tier`.
* **Config metadata**: runs record `paradigm` and `tier`, not axis coordinates. The intended translation is:

```python
def resolve_legacy_config(cfg: dict) -> dict:
    """Maps paradigm/tier run config onto the axes of 04_experimental_matrix.md (not yet in code)."""
    paradigm, tier = cfg.get("paradigm", "sm_mh"), int(cfg.get("tier", 1))
    agency = "autonomous" if paradigm == "ar" else "harnessed"
    if paradigm == "ar":
        topology = "carry_first" if cfg.get("trace_order") == "carry_first" else "sum_first"
    else:
        topology = {"sm_mh": "factored", "sm_fork": "factored",
                    "sm_seq": "sum_first", "sm_seq_rev": "carry_first"}[paradigm]
    channel = "weight" if paradigm == "sm_mh" else "sequence"
    representation = {1: "sinusoidal_pe", 2: "semantic_nope", 3: "generic_nope", 4: "role_embed_nope",
                      5: "semantic_origin_nope", 6: "addressed_kv", 7: "zero_preamble_kv"}[tier]
    curriculum = "full_span" if tier == 7 else "standard"
    return {**cfg, "agency": agency, "topology": topology, "channel": channel,
            "representation": representation, "curriculum": curriculum,
            "p_prior": cfg.get("p_prior", 0.0)}
```
