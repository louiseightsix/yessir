# yessir

*"Yes sir."* Whatever the rule says.

**yessir** is a set of experiments on whether transformers can follow a rule that is **given to them in context**. That means doing what the prompt says even when it isn't what training made familiar, and still doing it on inputs bigger or stranger than anything seen in training.

Each experiment picks one small, exact rule. The rule's parameters are stated in the prompt, and the experiment asks:

- does the model actually *read* the rule, or did it memorise the training cases?
- does it keep following the rule on unseen parameters and longer inputs, zero-shot?
- when it fails, did it misread the rule or mis-execute it?

## Repo rules

| Directory | What lives there | Promise |
| :--- | :--- | :--- |
| `experiments/` | Live experiments: code, tests, spec | **Kept live.** It runs and its tests pass. Code is added in small, reviewed pieces. |
| `results/` | Outputs of live experiments, one directory per experiment | **Stand-by-able.** Every result is one you would stand by, or is marked `FLAWED:` with a reason ([details](results/README.md)). |
| `yessir/` | Code shared by more than one experiment | Reviewed, and only there once it's clearly shared. |
| `prototypes/` | Quick explorations, each with its own code, spec, notes and results | **Allowed to rot.** Kept as reference only; not maintained, not tested by default. |

## Experiments

| Experiment | Rule | Status |
| :--- | :--- | :--- |
| [`addition`](experiments/addition/) | Column addition in a numeral system defined in the prompt (an ordered alphabet of arbitrary digit symbols, so any base) | Being rebuilt piece by piece |

## Prototypes

| Prototype | What it was |
| :--- | :--- |
| [`transformadder`](prototypes/transformadder/) | The first, quick exploration of the addition idea: 7 prompt representations × 5 execution schemes, ~40 training runs (Sept–Oct 2026). Showed in-domain addition is easy and zero-shot transfer to unseen bases is partial and unstable. |

## Setup

Needs [uv](https://docs.astral.sh/uv/). Everything goes through `yessir.sh`, which creates and syncs the environment on first use:

```bash
./yessir.sh test             # live experiments only; prototypes: ./yessir.sh test prototypes/<name>
./yessir.sh uv add <pkg>     # add a dependency (any uv command works: ./yessir.sh uv <args>)
```

The environment is per OS (`.venv-darwin`, `.venv-linux`), so a checkout shared between a macOS host and a Linux VM works from both. Set `UV_PROJECT_ENVIRONMENT` to override. Prototypes keep their own instructions and use plain `uv` (a single `.venv`).
