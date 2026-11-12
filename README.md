# Yessir! 🫡

**Yessir!** is a playground for exploring rule-following behaviour in transformer-based models. 

## Experiments

| Experiment | Rule | Status |
| :--- | :--- | :--- |
| [`sir_yes_sir`](experiments/sir_yes_sir/) | Next symbol in an in-context alphabet (mod B, no carry) | Live smoke-test & reference experiment |
| [`addition`](experiments/addition/) | Long addition in bases defined in the prompt | Prototype done; implementation in progress |

## Layout

| Directory | What's there |
| :--- | :--- |
| `experiments/` | The experiments: code, tests and spec for each. |
| `results/` | Results from the experiments, one directory per experiment. A run whose `README.md` starts with `FLAWED:` is known to be wrong, for the reason given. |
| `web/` | Static web UI (Astro) for browsing `results/` and experiment configs. |
| `yessir/` | Code shared between experiments. |

## Let's Play

Needs [uv](https://docs.astral.sh/uv/) (and [pnpm](https://pnpm.io/) for `./yessir.sh browser`). Everything goes through `yessir.sh`, which creates and syncs the environment on first use:

```bash
./yessir.sh run              # run an experiment (interactive TUI or CLI flags)
./yessir.sh browser          # open the results browser (astro dev server)
./yessir.sh test             # run the tests
./yessir.sh check            # lint and typecheck (ruff check + ty check)
./yessir.sh fix              # typecheck, autofix lint issues, and format
./yessir.sh format           # format with ruff format
```

## Housekeeping

### Adding a dependency

```bash
./yessir.sh uv add <pkg>     # add a dependency (any uv command works: ./yessir.sh uv <args>)
```

### Agent sandboxes

The environment is per OS (`.venv-darwin`, `.venv-linux`), so a checkout shared between a macOS host and a Linux VM works from both. Set `UV_PROJECT_ENVIRONMENT` to override.
