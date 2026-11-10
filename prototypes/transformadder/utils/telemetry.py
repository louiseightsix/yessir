"""Instance telemetry, 4-way error taxonomy classification, and prior suppression metrics.
Conforms to spec/05_evaluation_and_metrics.md Sections 4, 5, and 6.
"""

from pathlib import Path
from typing import Any
import json
import math

import torch

from prototypes.transformadder.core.constants import (
    FLAG_FINISHED,
    TOKEN_PROBE,
    TOKEN_S,
    TOKEN_C,
    TOKEN_FLAG,
)
from prototypes.transformadder.core.rule import get_canonical_alphabet
from prototypes.transformadder.core.sequences import build_probe_prompt, Seq


def classify_first_error(
    instance: dict[str, Any],
    out: list[int],
    carries: list[int] | None = None,
    status: str = "finished",
) -> dict[str, Any]:
    """Inspects a rollout and classifies the first failure token into the 4-way error taxonomy:
    1. prior_intrusion: Emitted digit matches canonical order prediction under canonical_permuted.
    2. out_of_alphabet: Emitted token not in the active alphabet A.
    3. procedural_error: In A, but incorrect calculation.
    4. syntax_malformation: Misplaced control token, unparseable sequence, or abnormal halting.

    Returns dict matching spec/05 Section 6 JSON schema.
    """
    axiom = instance["axiom"]
    base = instance.get("base", len(axiom))
    steps = instance["steps"]
    expected_out = instance["expected_out"]

    alphabet_cond = instance.get("alphabet_condition")
    if alphabet_cond is None and "rule_params" in instance:
        alphabet_cond = instance["rule_params"].get("alphabet_condition")
    is_permuted = alphabet_cond == "canonical_permuted"

    canonical_alphabet = get_canonical_alphabet(base)

    # Check for perfect execution
    if status == "finished" and out == expected_out:
        return {
            "first_error_step_k": None,
            "first_error_type": None,
            "first_predicted_token": None,
            "first_expected_token": None,
            "canonical_expected_token": None,
        }

    # Step-by-step comparison to locate the earliest deviation
    for k, step in enumerate(steps):
        exp_o = step["o"]
        v_exp = step["total"] % base
        canon_exp = canonical_alphabet[v_exp] if v_exp < len(canonical_alphabet) else None

        if k >= len(out):
            # Model halted prematurely
            return {
                "first_error_step_k": k,
                "first_error_type": "syntax_malformation",
                "first_predicted_token": None,
                "first_expected_token": exp_o,
                "canonical_expected_token": canon_exp,
            }

        pred_o = out[k]
        if pred_o != exp_o:
            if pred_o not in axiom:
                err_type = "out_of_alphabet"
            elif is_permuted and canon_exp is not None and pred_o == canon_exp:
                err_type = "prior_intrusion"
            else:
                err_type = "procedural_error"

            return {
                "first_error_step_k": k,
                "first_error_type": err_type,
                "first_predicted_token": pred_o,
                "first_expected_token": exp_o,
                "canonical_expected_token": canon_exp,
            }

    # If all sum digits matched but status indicates error or extra tokens emitted
    if len(out) > len(steps) or status != "finished":
        return {
            "first_error_step_k": len(steps),
            "first_error_type": "syntax_malformation",
            "first_predicted_token": out[len(steps)] if len(out) > len(steps) else None,
            "first_expected_token": FLAG_FINISHED,
            "canonical_expected_token": None,
        }

    return {
        "first_error_step_k": None,
        "first_error_type": None,
        "first_predicted_token": None,
        "first_expected_token": None,
        "canonical_expected_token": None,
    }


def compute_chance_adjusted_prior_intrusion(
    prior_intrusion_count: int,
    total_errors: int,
    base: int,
) -> float | None:
    """Computes R_prior_adj = (R_prior - 1/(B - 1)) / (1 - 1/(B - 1)) as specified in spec/05 Section 5.1."""
    if total_errors == 0 or base <= 1:
        return None
    r_prior = prior_intrusion_count / total_errors
    r_chance = 1.0 / (base - 1)
    if r_chance >= 1.0:
        return 0.0
    return (r_prior - r_chance) / (1.0 - r_chance)


def calculate_distance_regressors(
    prompt_len: int,
    axiom_len: int,
    tier: int,
    len_a: int,
    len_b: int,
) -> tuple[int | None, int]:
    """Calculates continuous layout regressors:
    - rule_to_query_distance: Token distance from end of axiom header to query trigger (None for Tier 7).
    - entry_count_between: Count of operand elements intervening between axiom and query.
    """
    entry_count_between = len_a + len_b
    if tier == 7:
        # Tier 7 has no axiom preamble
        return None, entry_count_between

    if tier in (1, 3, 4):
        # [BOS] + axiom (len 1 + axiom_len)
        axiom_end = 1 + axiom_len
    elif tier in (2, 6):
        # [BOS, DELIM_AXIOM] + axiom (len 2 + axiom_len)
        axiom_end = 2 + axiom_len
    elif tier == 5:
        # [BOS, DELIM_AXIOM, DELIM_ORIGIN] + axiom (len 3 + axiom_len)
        axiom_end = 3 + axiom_len
    else:
        axiom_end = 1 + axiom_len

    query_pos = max(0, prompt_len - 1)
    rule_to_query_distance = max(0, query_pos - axiom_end)
    return rule_to_query_distance, entry_count_between


@torch.no_grad()
def eval_binding_probe(
    model,
    tier: int,
    instances: list[dict[str, Any]],
    device: torch.device | str | None = None,
) -> float:
    """Evaluates the Single-Token Successor Binding Probe ([PROBE] s_i -> s_{i+1})
    across instances for sequence-space models. Returns mean accuracy in [0.0, 1.0].
    Returns nan for weight-channel models (SM-MH).
    """
    if getattr(model, "head_type", None) == "multi_head":
        return float("nan")

    if device is None:
        device = next(model.parameters()).device

    correct = 0
    total = 0

    for inst in instances:
        axiom = inst["axiom"]
        base = len(axiom)
        # Probe a random symbol from the instance's axiom
        query_sym = axiom[0] if len(axiom) == 1 else axiom[len(axiom) // 2]
        idx = axiom.index(query_sym)
        target = axiom[(idx + 1) % base]

        prompt = build_probe_prompt(axiom, inst["a"], inst["b"], query_sym, tier)
        logits = model.next_logits(prompt)
        pred = int(logits.argmax(-1).item())
        if pred == target:
            correct += 1
        total += 1

    return correct / total if total > 0 else float("nan")


class InstanceTelemetryLogger:
    """Thread-safe JSONL per-instance regression logger writing instances.jsonl."""

    def __init__(self, log_dir: str | Path):
        self.path = Path(log_dir) / "instances.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_instance(self, record: dict[str, Any]) -> None:
        """Sanitizes floats (NaN/inf -> null) and appends a record as a JSON line."""
        clean_record = {}
        for k, v in record.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_record[k] = None
            else:
                clean_record[k] = v
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean_record) + "\n")
