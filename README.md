# Yessir! 🫡

**Yessir!** is a playground for exploring rule-following behaviour in Transformers-based models. 

## Experiments

| Experiment | Rule | Status |
| :--- | :--- | :--- |
| [`addition`](experiments/addition/) | Column addition in a numeral system defined in the prompt (an ordered alphabet of arbitrary digit symbols, so any base) | No code |

## Layout

| Directory | What's there |
| :--- | :--- |
| `experiments/` | The experiments: code, tests and spec for each. |
| `results/` | Results from the experiments, one directory per experiment. A run whose `README.md` starts with `FLAWED:` is known to be wrong, for the reason given. |
| `yessir/` | Code shared between experiments. |
| `prototypes/` | Early, quick explorations, each with its own code, spec, notes and results. Unmaintained and unreviewed: useful as background, not as evidence. |

## Let's Play

Needs [uv](https://docs.astral.sh/uv/). Everything goes through `yessir.sh`, which creates and syncs the environment on first use:

```bash
./yessir.sh test             # live experiments only; prototypes: ./yessir.sh test prototypes/<name>
./yessir.sh uv add <pkg>     # add a dependency (any uv command works: ./yessir.sh uv <args>)
```

## Housekeeping

Add a dep:

```bash
./yessir.sh uv add <pkg>     # add a dependency (any uv command works: ./yessir.sh uv <args>)
```

Agent sandboxes

The environment is per OS (`.venv-darwin`, `.venv-linux`), so a checkout shared between a macOS host and a Linux VM works from both. Set `UV_PROJECT_ENVIRONMENT` to override. Prototypes keep their own instructions and use plain `uv` (a single `.venv`).
