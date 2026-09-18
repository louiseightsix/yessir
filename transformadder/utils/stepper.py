"""Interactive educational stepper for pedagogical inspection of TransformAdder.
Pauses after each step, displays the model's emissions next to the reference computation,
and supports single-key controls ([Enter], c, q, v).

The stepper sits OUTSIDE the zero-arithmetic harness boundary: it calls the reference adder to
show what the correct transition is. It has no access to, and makes no claims about, the model's
internal mechanism.
"""

import sys
import torch
import torch.nn.functional as F

from transformadder.core.constants import FLAG_FINISHED, token_to_str
from transformadder.core.reference import reference_step
from transformadder.core.sequences import Seq


class InteractiveStepper:
    """Educational step-by-step inspector for training and test harnesses."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.paused = enabled
        self.verbose = False

    def step(
        self,
        k: int,
        prompt: Seq,
        emitted_sum: int,
        emitted_carry: int,
        emitted_flag: int,
        logits_list: list[torch.Tensor] | None = None,
        axiom: list[int] | None = None,
        a: list[int] | None = None,
        b: list[int] | None = None,
        carry_in: int | None = None,
    ):
        """Pause and display one state transition for the State Machine paradigms."""
        if not self.enabled or not self.paused:
            return

        print("\n" + "=" * 80)
        print(f"TransformAdder Interactive Stepper | Step k = {k} | Base B = {len(axiom) if axiom else '?'}")
        print("=" * 80)
        print("Prompt:     " + " ".join(token_to_str(t) for t in prompt.tokens))

        # The reference is posed from the state the model is actually in (its own carry-in),
        # and is undefined if that carry is not an axiom symbol.
        ref = None
        if axiom and a is not None and b is not None and carry_in in axiom:
            ref = reference_step(axiom, a, b, k, carry_in)

        if axiom:
            print("Axiom:      " + ", ".join(f"{token_to_str(s)} ({i})" for i, s in enumerate(axiom)))
        if ref:
            digit_a = token_to_str(a[k]) if k < len(a) else "absent"
            digit_b = token_to_str(b[k]) if k < len(b) else "absent"
            print(f"Operand A:  Column {k} Digit = {digit_a} (value {ref['v_a']})")
            print(f"Operand B:  Column {k} Digit = {digit_b} (value {ref['v_b']})")
            print(f"Carry-In:   c_{k} = {token_to_str(carry_in)} (value {ref['v_c']})")
        elif carry_in is not None:
            print(f"Carry-In:   c_{k} = {token_to_str(carry_in)} (NOT an axiom symbol; no reference available)")
        print("-" * 80)

        def verdict(emitted: int, key: str) -> str:
            if ref is None:
                return ""
            return "[matches reference]" if emitted == ref[key] else f"[reference: {token_to_str(ref[key])}]"

        finished = emitted_flag == FLAG_FINISHED
        print("Model Predictions:")
        print(f"  Sum Output:     o_{k} = {token_to_str(emitted_sum)} {verdict(emitted_sum, 'o')}")
        print(f"  Outgoing Carry: c_{k + 1} = {token_to_str(emitted_carry)} {verdict(emitted_carry, 'c_out')}")
        print(f"  Control Flag:   {token_to_str(emitted_flag)} {verdict(emitted_flag, 'flag')}")
        print("-" * 80)

        if ref:
            base = len(axiom)
            print("Reference Computation (oracle, not a readout of the model's internals):")
            print(f"    val(A_{k}) + val(B_{k}) + val(c_{k}) = {ref['v_a']} + {ref['v_b']} + {ref['v_c']} = {ref['total']}")
            print(f"    Sum digit:  {ref['total']} mod {base} = {ref['total'] % base} -> Symbol {token_to_str(ref['o'])}")
            print(f"    Carry-out:  floor({ref['total']}/{base}) = {ref['total'] // base} -> Symbol {token_to_str(ref['c_out'])}")

        print("  * What Happens Next:")
        print(f"    Harness appends the model's {token_to_str(emitted_sum)} to accumulated OUT.")
        if finished:
            print("    Harness detects the termination flag. Addition loop halts; full OUT is returned.")
        else:
            print(f"    Active carry becomes c_{k + 1} = {token_to_str(emitted_carry)}; loop proceeds to column k = {k + 1}.")
        print("=" * 80)

        # Verbose logit inspection if toggled
        if self.verbose and logits_list:
            self._print_verbose_logits(logits_list)

        self._await_user_input()

    def show_trace(self, prompt: Seq, trace: list[int], out_digits: list[int], status: str):
        """Display trace generation results for the Autoregressive paradigm."""
        if not self.enabled or not self.paused:
            return

        print("\n" + "=" * 80)
        print(f"TransformAdder Interactive Stepper | Autoregressive Rollout | Status: {status}")
        print("=" * 80)
        print("Prompt Prefix:   " + " ".join(token_to_str(t) for t in prompt.tokens))
        print("Generated Trace: " + " ".join(token_to_str(t) for t in trace))
        print("Extracted Sum:   " + " ".join(token_to_str(t) for t in out_digits))
        print("=" * 80)
        self._await_user_input()

    def _print_verbose_logits(self, logits_list: list[torch.Tensor]):
        print("\n[Verbose Top-5 Distribution]")
        names = ["Sum Logits", "Carry Logits", "Flag Logits"]
        for name, logit_tensor in zip(names, logits_list):
            probs = F.softmax(logit_tensor.squeeze().float(), dim=-1)
            topk_vals, topk_inds = torch.topk(probs, min(5, probs.shape[-1]))
            top_str = ", ".join(
                [f"{token_to_str(int(idx))}: {float(val)*100:.1f}%" for idx, val in zip(topk_inds, topk_vals)]
            )
            print(f"  {name}: {top_str}")
        print()

    def _await_user_input(self):
        """Wait for keyboard command."""
        prompt = "[Enter: Step | c: Continue without pause | v: Toggle verbose | q: Quit]: "
        while True:
            try:
                cmd = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive stepper.")
                sys.exit(0)

            if cmd == "":
                break
            elif cmd == "c":
                self.paused = False
                print("Continuing execution without pausing...")
                break
            elif cmd == "v":
                self.verbose = not self.verbose
                print(f"Verbose mode set to: {self.verbose}")
                break
            elif cmd == "q":
                print("Execution aborted by user.")
                sys.exit(0)
            else:
                print(f"Unrecognised command: '{cmd}'. Press Enter, 'c', 'v', or 'q'.")
