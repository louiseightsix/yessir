"""Zero-shot evaluation (spec/06_implementation.md Section 7.2), scored against the reference
adder. Free-running metrics drive the model through its harness; teacher-forced metrics pose
each transition from the oracle state, isolating single-step fidelity from error accumulation.
"""

import torch

from transformadder.core.constants import FLAG_CLASS_FINISHED, FLAG_FINISHED, IGNORE_LABEL
from transformadder.core.data import (
    ProceduralArithmeticGenerator,
    collate_labelled,
    collate_sm_mh,
    encode_ar,
    encode_sm_mh,
    encode_sm_seq,
    encode_sm_seq_rev,
)
from transformadder.harnesses.autoregressive import run_autoregressive_trace
from transformadder.harnesses.state_machine import (
    run_state_machine_mh,
    run_state_machine_seq,
    run_state_machine_seq_rev,
)

HARNESSES = {
    "sm_mh": run_state_machine_mh,
    "sm_seq": run_state_machine_seq,
    "sm_seq_rev": run_state_machine_seq_rev,
    "ar": run_autoregressive_trace,
}

STANDARD_LEN = (1, 20)
EXTRAP_LEN = (21, 40)


def eval_cells(train_bases: list[int]) -> list[tuple[str, list[int], tuple[int, int]]]:
    """Evaluation cells of spec/01_research_goal.md Section 4.1 (in-domain bases follow the
    active training regime)."""
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
def free_running_metrics(model, paradigm: str, tier: int, instances: list[dict], stepper=None) -> dict:
    """Exact match, digit accuracy, carry-chain accuracy and failure counts over rollouts."""
    run = HARNESSES[paradigm]
    n_em = n_digits = n_digits_ok = n_carry = n_carry_ok = n_out_of_axiom = 0
    statuses: dict[str, int] = {}

    for inst in instances:
        carries: list[int] = []
        out, status = run(model, inst["axiom"], inst["a"], inst["b"], tier, stepper=stepper, carries=carries)
        statuses[status] = statuses.get(status, 0) + 1
        n_em += int(status == "finished" and out == inst["expected_out"])
        n_out_of_axiom += sum(1 for t in out + carries if t not in inst["axiom"])  # sum digits and carries

        s_0 = inst["axiom"][0]
        for j, step in enumerate(inst["steps"]):
            ok = int(j < len(out) and out[j] == step["o"])
            n_digits += 1
            n_digits_ok += ok
            if step["c_in"] != s_0 or step["c_out"] != s_0:
                n_carry += 1
                n_carry_ok += ok

    n = len(instances)
    metrics = {
        "em": n_em / n,
        "digit_acc": n_digits_ok / n_digits,
        "carry_acc": n_carry_ok / n_carry if n_carry else float("nan"),
        "out_of_axiom": n_out_of_axiom,
    }
    for status in ("max_steps_exceeded", "max_tokens_exceeded", "malformed"):
        metrics[status] = statuses.get(status, 0)
    return metrics


@torch.no_grad()
def teacher_forced_metrics(model, paradigm: str, tier: int, instances: list[dict], chunk: int = 256) -> dict:
    """Step-transition accuracy (all of o_k, c_{k+1}, FLAG_k correct) and halting accuracy,
    over every column of every instance. Returns per-transition (step_ok, flag_ok, terminal)."""
    model.eval()
    device = next(model.parameters()).device
    records: list[tuple[bool, bool, bool]] = []

    if paradigm == "ar":
        samples = [encode_ar(inst, tier) for inst in instances]
    else:
        if paradigm == "sm_mh":
            encode = encode_sm_mh
        elif paradigm == "sm_seq_rev":
            encode = encode_sm_seq_rev
        else:
            encode = encode_sm_seq
        samples = [encode(inst, k, tier) for inst in instances for k in range(len(inst["steps"]))]

    for start in range(0, len(samples), chunk):
        part = samples[start:start + chunk]

        if paradigm == "sm_mh":
            batch = _to_device(collate_sm_mh(part), device)
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
        pred = model(batch["token_ids"], batch["role_ids"], batch["sig_ids"]).argmax(-1).tolist()
        labels = batch["labels"].tolist()

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
) -> dict:
    """Runs evaluation cells on configured axiom sets (random, canonical ASCII, or both).

    Keys are `{cell}_{metric}` for random axioms and `canon_{cell}_{metric}` for canonical ones.
    The evaluation seed is disjoint from the training seeds {0, 1, 2}.
    """
    results = {}
    if eval_axioms == "random":
        canonical_flags = [False]
    elif eval_axioms == "canonical":
        canonical_flags = [True]
    else:  # "both"
        canonical_flags = [False, True]

    for canonical in canonical_flags:
        for name, bases, (min_len, max_len) in eval_cells(train_bases):
            gen = ProceduralArithmeticGenerator(
                bases=bases, min_len=min_len, max_len=max_len, canonical=canonical, seed=seed
            )
            instances = [gen.sample_instance() for _ in range(num_problems)]
            metrics = free_running_metrics(model, paradigm, tier, instances, stepper=stepper)
            metrics.update(teacher_forced_metrics(model, paradigm, tier, instances))
            prefix = f"canon_{name}" if canonical else name
            results.update({f"{prefix}_{k}": v for k, v in metrics.items()})
    return results
