"""Rule protocol and concrete rule implementations.

Establishes a task-independent procedural execution contract:
- Rule: A procedure with parameters (e.g. Addition with base B and alphabet A).
- Binding: Supplying parameters in context per instance.
- Execution: Producing the output under that binding.
- Prior: Competing procedure or binding learned from training distribution.

This interface ensures that the training loop, architectures, and evaluation telemetry
scale seamlessly to general algorithmic and ARC-AGI rule-following tasks.
"""

from typing import Any, Protocol, TypedDict, runtime_checkable
import random

from experiments.addition.core.constants import (
    CANONICAL_AXIOM,
    SIGMA_POOL,
)
from experiments.addition.core.reference import reference_add, reference_expected_out


def get_canonical_alphabet(base: int) -> list[int]:
    """Returns conventional ASCII numerals '0'-'9', 'A'-'Z' truncated to base."""
    if base <= 10:
        return [ord("0") + i for i in range(base)]
    return [ord("0") + i for i in range(10)] + [ord("A") + i for i in range(base - 10)]


def sample_alphabet(base: int, condition: str, rng: random.Random) -> list[int]:
    """Samples an alphabet of length base under the specified condition.

    Conditions:
    - 'canonical_order': Conventional ASCII glyphs in ascending numerical order.
    - 'canonical_permuted': The exact same conventional ASCII glyphs in permuted order.
    - 'symbols_random': Random bytes drawn uniformly without replacement from Sigma pool.
    """
    if condition == "canonical_order":
        return get_canonical_alphabet(base)
    elif condition == "canonical_permuted":
        canonical = get_canonical_alphabet(base)
        perm = canonical[:]
        if base > 1:
            for _ in range(100):
                rng.shuffle(perm)
                if perm != canonical:
                    break
        return perm
    elif condition in ("symbols_random", "random", "random_uniform"):
        return rng.sample(SIGMA_POOL, base)
    raise ValueError(f"Unknown alphabet condition: {condition}")


class RuleInstance(TypedDict):
    instance_id: str
    rule_family: str
    rule_params: dict[str, Any]
    inputs: dict[str, Any]
    steps: list[dict[str, Any]]
    expected_out: list[int]


@runtime_checkable
class Rule(Protocol):
    """Generic protocol for procedural rule execution engines."""

    @property
    def rule_family(self) -> str:
        """Unique identifier for the rule family (e.g. 'addition', 'arc_grid')."""
        ...

    def sample_instance(
        self,
        rng: random.Random,
        base: int,
        axiom: list[int] | None = None,
        condition: str = "symbols_random",
        min_len: int = 1,
        max_len: int = 20,
        p_chain: float = 0.25,
        p_terminal: float = 0.2,
        tier: int = 1,
    ) -> RuleInstance:
        """Deterministically generates a problem instance with oracle ground truth."""
        ...

    def reference_step(
        self,
        state: Any,
        input_slice: Any,
        rule_params: dict[str, Any],
    ) -> tuple[Any, int, int]:
        """Executes a single ground-truth transition delta: (state, input) -> (next_state, out, flag)."""
        ...


class AdditionRule:
    """Concrete Rule implementation for multi-base columnar addition."""

    @property
    def rule_family(self) -> str:
        return "addition"

    def reference_step(
        self,
        state: int,
        input_slice: tuple[int, int],
        rule_params: dict[str, Any],
    ) -> tuple[int, int, int]:
        """Executes one column addition: (carry_in, (a_k, b_k)) -> (carry_out, sum_digit, flag)."""
        axiom = rule_params["axiom"]
        base = rule_params.get("base", len(axiom))
        s_0 = axiom[0]
        a_sym, b_sym = input_slice
        val_a = axiom.index(a_sym) if a_sym in axiom else 0
        val_b = axiom.index(b_sym) if b_sym in axiom else 0
        val_cin = axiom.index(state) if state in axiom else 0

        total = val_a + val_b + val_cin
        o_val = total % base
        cout_val = total // base
        return axiom[cout_val], axiom[o_val], 0

    def sample_instance(
        self,
        rng: random.Random,
        base: int,
        axiom: list[int] | None = None,
        condition: str = "symbols_random",
        min_len: int = 1,
        max_len: int = 20,
        p_chain: float = 0.25,
        p_terminal: float = 0.2,
        tier: int = 1,
    ) -> dict[str, Any]:
        if axiom is None:
            axiom = sample_alphabet(base, condition, rng)

        effective_min = max(min_len, base) if tier == 7 else min_len
        eff_max = max(effective_min, max_len)
        len_a = rng.randint(effective_min, eff_max)
        len_b = rng.randint(effective_min, eff_max)

        a_vals = [rng.randrange(base) for _ in range(len_a)]
        b_vals = [rng.randrange(base) for _ in range(len_b)]

        if rng.random() < p_chain:
            n = min(len(a_vals), len(b_vals))
            if n >= 2:
                L = rng.randint(1, n - 1)
                i = rng.randint(0, n - 1 - L)
                a_vals[i] = rng.randint(1, base - 1)
                b_vals[i] = rng.randint(base - a_vals[i], base - 1)
                for j in range(i + 1, i + 1 + L):
                    a_vals[j] = rng.randrange(base)
                    b_vals[j] = base - 1 - a_vals[j]

        a = [axiom[v] for v in a_vals]
        b = [axiom[v] for v in b_vals]

        steps = reference_add(axiom, a, b)
        expected_out = reference_expected_out(steps)

        import hashlib
        h = hashlib.sha256(bytes([base] + axiom + [0xFF] + a + [0xFF] + b)).hexdigest()[:16]

        return {
            "instance_id": h,
            "rule_family": self.rule_family,
            "rule_params": {
                "base": base,
                "axiom": axiom,
                "alphabet_condition": condition,
            },
            "inputs": {"a": a, "b": b},
            "steps": steps,
            "expected_out": expected_out,
            # Top-level mirrors for backward compatibility
            "base": base,
            "axiom": axiom,
            "a": a,
            "b": b,
            "alphabet_condition": condition,
        }

