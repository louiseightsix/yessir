"""Zero-shot evaluation (spec/06_implementation.md Section 7.2), scored against the reference
adder. Free-running metrics drive the model through its harness; teacher-forced metrics pose
each transition from the oracle state, isolating single-step fidelity from error accumulation.
"""

import math
import torch

from transformadder.core.constants import (
    FLAG_CLASS_FINISHED,
    FLAG_FINISHED,
    IGNORE_LABEL,
    TOKEN_C,
    TOKEN_FLAG,
    TOKEN_S,
)
from transformadder.core.data import (
    ProceduralArithmeticGenerator,
    collate_labelled,
    collate_sm_mh,
    encode_ar,
    encode_sm_fork,
    encode_sm_mh,
    encode_sm_seq,
    encode_sm_seq_rev,
)
from transformadder.core.sequences import build_sm_prompt, build_ar_prompt, build_probe_prompt
from transformadder.harnesses.autoregressive import run_autoregressive_trace
from transformadder.harnesses.state_machine import (
    run_state_machine_fork,
    run_state_machine_mh,
    run_state_machine_seq,
    run_state_machine_seq_rev,
)
from transformadder.utils.telemetry import (
    InstanceTelemetryLogger,
    calculate_distance_regressors,
    classify_first_error,
    compute_chance_adjusted_prior_intrusion,
    eval_binding_probe,
)

HARNESSES = {
    "sm_mh": run_state_machine_mh,
    "sm_seq": run_state_machine_seq,
    "sm_seq_rev": run_state_machine_seq_rev,
    "sm_fork": run_state_machine_fork,
    "ar": run_autoregressive_trace,
}

STANDARD_LEN = (1, 20)
EXTRAP_LEN = (21, 40)


def eval_cells(train_bases: list[int]) -> list[tuple[str, list[int], tuple[int, int]]]:
    """Evaluation cells of spec/05_evaluation_and_metrics.md Section 2."""
    return [
        ("id", list(train_bases), STANDARD_LEN),
        ("ood_base3", [3], STANDARD_LEN),
        ("ood_base12", [12], STANDARD_LEN),
        ("extrap_len", list(train_bases), EXTRAP_LEN),
        ("extrap_len_base3", [3], EXTRAP_LEN),
        ("extrap_len_base12", [12], EXTRAP_LEN),
    ]


def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if v is not None else None for k, v in batch.items()}


@torch.no_grad()
def free_running_metrics(
    model,
    paradigm: str,
    tier: int,
    instances: list[dict],
    stepper=None,
    trace_order: str = "sum_first",
    telemetry_logger: InstanceTelemetryLogger | None = None,
    p_prior: float | None = None,
    p_probe: float | None = None,
) -> dict:
    """Exact match, digit accuracy, carry-chain accuracy, 4-way error counts, and telemetry logging."""
    run = HARNESSES[paradigm]
    n_em = n_digits = n_digits_ok = n_carry = n_carry_ok = n_out_of_axiom = 0
    statuses: dict[str, int] = {}
    error_type_counts = {
        "prior_intrusion": 0,
        "out_of_alphabet": 0,
        "procedural_error": 0,
        "syntax_malformation": 0,
    }

    for inst in instances:
        carries: list[int] = []
        if paradigm == "ar":
            out, status = run(
                model, inst["axiom"], inst["a"], inst["b"], tier, stepper=stepper, carries=carries, order=trace_order
            )
        else:
            out, status = run(model, inst["axiom"], inst["a"], inst["b"], tier, stepper=stepper, carries=carries)
        statuses[status] = statuses.get(status, 0) + 1
        is_em = int(status == "finished" and out == inst["expected_out"])
        n_em += is_em
        n_out_of_axiom += sum(1 for t in out + carries if t not in inst["axiom"])

        inst_digits = 0
        inst_digits_ok = 0
        s_0 = inst["axiom"][0]
        for j, step in enumerate(inst["steps"]):
            ok = int(j < len(out) and out[j] == step["o"])
            n_digits += 1
            n_digits_ok += ok
            inst_digits += 1
            inst_digits_ok += ok
            if step["c_in"] != s_0 or step["c_out"] != s_0:
                n_carry += 1
                n_carry_ok += ok

        per_step_acc = inst_digits_ok / inst_digits if inst_digits > 0 else 0.0

        # Classify first error token
        err_info = classify_first_error(inst, out, carries, status)
        if err_info["first_error_type"] is not None:
            error_type_counts[err_info["first_error_type"]] += 1

        # Instance telemetry logging
        if telemetry_logger is not None:
            if paradigm == "ar":
                prompt_seq = build_ar_prompt(inst["axiom"], inst["a"], inst["b"], tier)
            else:
                prompt_seq = build_sm_prompt(inst["axiom"], inst["a"], inst["b"], inst["axiom"][0], [], tier, TOKEN_S)
            prompt_len = len(prompt_seq)
            rule_dist, entry_count = calculate_distance_regressors(
                prompt_len, len(inst["axiom"]), tier, len(inst["a"]), len(inst["b"])
            )

            probe_acc_inst = None
            if getattr(model, "head_type", None) != "multi_head":
                s_i = inst["axiom"][len(inst["axiom"]) // 2]
                pr_prompt = build_probe_prompt(inst["axiom"], inst["a"], inst["b"], s_i, tier)
                target = inst["axiom"][(inst["axiom"].index(s_i) + 1) % len(inst["axiom"])]
                logits = model.next_logits(pr_prompt)
                probe_acc_inst = 1.0 if int(logits.argmax(-1).item()) == target else 0.0

            telemetry_logger.log_instance({
                "instance_id": inst.get("instance_id", f"{hash(str(inst)):08x}"),
                "rule_family": inst.get("rule_family", "addition"),
                "base": inst["base"],
                "alphabet_condition": inst.get("alphabet_condition", "symbols_random"),
                "p_prior": p_prior,
                "p_probe": p_probe,
                "len_a": len(inst["a"]),
                "len_b": len(inst["b"]),
                "prompt_token_length": prompt_len,
                "rule_to_query_distance": rule_dist,
                "entry_count_between": entry_count,
                "outcome_em": is_em,
                "per_step_accuracy": per_step_acc,
                "probe_accuracy": probe_acc_inst,
                "first_error_step_k": err_info["first_error_step_k"],
                "first_error_type": err_info["first_error_type"],
                "first_predicted_token": err_info["first_predicted_token"],
                "first_expected_token": err_info["first_expected_token"],
                "canonical_expected_token": err_info["canonical_expected_token"],
            })

    n = len(instances)
    total_errors = n - n_em
    base_0 = instances[0]["base"] if instances else 10
    r_prior_adj = compute_chance_adjusted_prior_intrusion(
        error_type_counts["prior_intrusion"], total_errors, base_0
    )

    metrics = {
        "em": n_em / n if n else 0.0,
        "digit_acc": n_digits_ok / n_digits if n_digits else 0.0,
        "carry_acc": n_carry_ok / n_carry if n_carry else float("nan"),
        "out_of_axiom": n_out_of_axiom,
        "prior_intrusion_count": error_type_counts["prior_intrusion"],
        "out_of_alphabet_count": error_type_counts["out_of_alphabet"],
        "procedural_error_count": error_type_counts["procedural_error"],
        "syntax_malformation_count": error_type_counts["syntax_malformation"],
        "r_prior_adj": r_prior_adj,
    }
    for status in ("max_steps_exceeded", "max_tokens_exceeded", "malformed"):
        metrics[status] = statuses.get(status, 0)
    return metrics


@torch.no_grad()
def teacher_forced_metrics(
    model, paradigm: str, tier: int, instances: list[dict], chunk: int = None, trace_order: str = "sum_first"
) -> dict:
    """Step-transition accuracy (all of o_k, c_{k+1}, FLAG_k correct) and halting accuracy,
    over every column of every instance. Returns per-transition (step_ok, flag_ok, terminal)."""
    model.eval()
    device = next(model.parameters()).device
    records: list[tuple[bool, bool, bool]] = []

    if chunk is None:
        if tier in (6, 7):
            chunk = 33 if paradigm == "sm_fork" else 32
        else:
            chunk = 255 if paradigm == "sm_fork" else 256

    if paradigm == "ar":
        samples = [encode_ar(inst, tier, order=trace_order) for inst in instances]
    elif paradigm == "sm_fork":
        samples = [
            encode_sm_fork(inst, k, tier, q)
            for inst in instances
            for k in range(len(inst["steps"]))
            for q in (TOKEN_S, TOKEN_C, TOKEN_FLAG)
        ]
    else:
        if paradigm == "sm_mh":
            encode = encode_sm_mh
        elif paradigm == "sm_seq_rev":
            encode = encode_sm_seq_rev
        else:
            encode = encode_sm_seq
        samples = [encode(inst, k, tier) for inst in instances for k in range(len(inst["steps"]))]

    if paradigm == "sm_fork":
        chunk = max(3, (chunk // 3) * 3)

    use_amp = device.type == "cuda"

    for start in range(0, len(samples), chunk):
        part = samples[start:start + chunk]

        if paradigm == "sm_mh":
            batch = _to_device(collate_sm_mh(part), device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                ls, lc, lf = model.forward_sm_mh(
                    batch["token_ids"], batch["role_ids"], batch["sig_ids"], batch["lengths"]
                )
            sum_ok = ls.argmax(-1) == batch["target_sum"]
            carry_ok = lc.argmax(-1) == batch["target_carry"]
            flag_ok = lf.argmax(-1) == batch["target_flag"]
            terminal = batch["target_flag"] == FLAG_CLASS_FINISHED
            records += zip((sum_ok & carry_ok & flag_ok).tolist(), flag_ok.tolist(), terminal.tolist())
            continue

        batch = _to_device(collate_labelled(part), device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            pred = model(batch["token_ids"], batch["role_ids"], batch["sig_ids"]).argmax(-1).tolist()
        labels = batch["labels"].tolist()

        if paradigm == "sm_fork":
            for i in range(0, len(pred), 3):
                p_s, l_s = pred[i], labels[i]
                p_c, l_c = pred[i + 1], labels[i + 1]
                p_f, l_f = pred[i + 2], labels[i + 2]
                idx_s = [t for t, y in enumerate(l_s) if y != IGNORE_LABEL][0]
                idx_c = [t for t, y in enumerate(l_c) if y != IGNORE_LABEL][0]
                idx_f = [t for t, y in enumerate(l_f) if y != IGNORE_LABEL][0]
                sum_ok = (p_s[idx_s] == l_s[idx_s])
                carry_ok = (p_c[idx_c] == l_c[idx_c])
                flag_ok = (p_f[idx_f] == l_f[idx_f])
                records.append((sum_ok and carry_ok and flag_ok, flag_ok, l_f[idx_f] == FLAG_FINISHED))
            continue

        for row_pred, row_labels in zip(pred, labels):
            idx = [t for t, y in enumerate(row_labels) if y != IGNORE_LABEL]
            if paradigm in ("sm_seq", "sm_seq_rev"):
                columns = [idx]                                   # 3 active targets per column step
            else:
                # Trace labels: [S] o_k [C] c_{k+1} per column, then the next [S] or [FINISHED],
                # which plays the role of FLAG_k
                columns = [[idx[4 * k + 1], idx[4 * k + 3], idx[4 * k + 4]]
                           for k in range((len(idx) - 1) // 4)]
            for col in columns:
                oks = [row_pred[t] == row_labels[t] for t in col]
                records.append((all(oks), oks[2], row_labels[col[2]] == FLAG_FINISHED))

    def rate(rows):
        return sum(rows) / len(rows) if rows else float("nan")

    return {
        "step_acc": rate([r[0] for r in records]),
        "halt_acc": rate([r[1] for r in records]),
        "halt_acc_terminal": rate([r[1] for r in records if r[2]]),
        "halt_acc_nonterminal": rate([r[1] for r in records if not r[2]]),
    }


def evaluate(
    model,
    paradigm: str,
    tier: int,
    train_bases: list[int],
    num_problems: int,
    eval_axioms: str = "both",
    seed: int = 999,
    stepper=None,
    trace_order: str = "sum_first",
    log_dir=None,
    p_prior: float | None = None,
    p_probe: float | None = None,
) -> dict:
    """Runs evaluation cells on configured alphabet conditions (random, canonical, permuted, or all).

    Keys are `{condition}_{cell}_{metric}`.
    The evaluation seed is disjoint from the training seeds {0, 1, 2}.
    """
    results = {}
    conditions = []
    if eval_axioms in ("random", "symbols_random"):
        conditions = [("", "symbols_random")]
    elif eval_axioms in ("canonical", "canonical_order"):
        conditions = [("canon", "canonical_order")]
    elif eval_axioms in ("permuted", "canonical_permuted"):
        conditions = [("permuted", "canonical_permuted")]
    elif eval_axioms == "all":
        conditions = [
            ("", "symbols_random"),
            ("canon", "canonical_order"),
            ("permuted", "canonical_permuted"),
        ]
    else:  # "both" (default)
        conditions = [
            ("", "symbols_random"),
            ("canon", "canonical_order"),
        ]

    telemetry_logger = InstanceTelemetryLogger(log_dir) if log_dir is not None else None

    device = next(model.parameters()).device
    use_amp = device.type == "cuda"

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        for prefix_label, condition in conditions:
            for name, bases, (min_len, max_len) in eval_cells(train_bases):
                gen = ProceduralArithmeticGenerator(
                    bases=bases,
                    min_len=min_len,
                    max_len=max_len,
                    alphabet_condition=condition,
                    seed=seed,
                    tier=tier,
                    trace_order=trace_order,
                )
                instances = [gen.sample_instance() for _ in range(num_problems)]
                metrics = free_running_metrics(
                    model,
                    paradigm,
                    tier,
                    instances,
                    stepper=stepper,
                    trace_order=trace_order,
                    telemetry_logger=telemetry_logger,
                    p_prior=p_prior,
                    p_probe=p_probe,
                )
                metrics.update(
                    teacher_forced_metrics(model, paradigm, tier, instances, trace_order=trace_order)
                )

                # Evaluate in-context binding probe
                probe_acc = eval_binding_probe(model, tier, instances, device=device)
                metrics["probe_acc"] = probe_acc
                # Retrieval-execution gap: Delta_exec = probe_acc - step_acc
                if not math.isnan(probe_acc) and not math.isnan(metrics.get("step_acc", float("nan"))):
                    metrics["delta_exec"] = probe_acc - metrics["step_acc"]
                else:
                    metrics["delta_exec"] = float("nan")

                prefix = f"{prefix_label}_{name}" if prefix_label else name
                for k, v in metrics.items():
                    results[f"{prefix}_{k}"] = v
                    if prefix_label == "":
                        results[f"random_{name}_{k}"] = v
    return results

