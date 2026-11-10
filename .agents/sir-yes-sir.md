# The sir-yes-sir experiment

`experiments/sir_yes_sir/` is a minimal, end-to-end smoke-test experiment. We are building it first—before rebuilding `experiments/addition/`—one small, reviewed piece at a time (see "Working with Louis" in `AGENTS.md`).

## Purpose

- **Exercise the full stack**: It touches every part of the shared `yessir/` machinery (tokenisation/sequences, model, training loop, evaluation, and result recording).
- **Separate infrastructure from science**: The task itself is trivial to learn and interpret, and trains quickly. If it fails to converge, the bug is in the code, not the hypothesis.
- **Living reference**: Kept up to date as the canonical example of how an experiment in this repo uses `yessir/`.
- **Learning vehicle**: Introduces each piece of shared machinery in the simplest possible context before applying it to `experiments/addition/`.

## Plan

1. **Choose the task and write the spec** in `experiments/sir_yes_sir/spec/`.
2. **Build piece by piece**: introduce each component of `yessir/` alongside its use in `experiments/sir_yes_sir/`, with tests passing via `./yessir.sh test` and a review pause after each piece.
3. **Run and record**: run the completed smoke-test experiment and record its output in `results/sir_yes_sir/` (following `.agents/results.md`).
4. **Move on to `experiments/addition/`**: once the shared machinery is proven and understood in `sir_yes_sir`, rebuild `addition` on top of it.
