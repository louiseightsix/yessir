# AGENTS.md

Guidance for AI coding agents working in this repo. Read it before doing anything.

## What this project is for

**yessir** is a set of experiments probing what rule-following behaviour transformers are capable of.

**This project doubles as a learning exercise for Louis**: transformers, training, and running careful experiments. A working result produced by code he hasn't understood is worth little here. The first version of the code was written quickly by an agent and never reviewed. It now lives in `.prototypes/transformadder/` as Louis's internal reference notes, and the live experiment is being rebuilt **one small, reviewed piece at a time**.

## Repo layout

```
AGENTS.md                this file: instructions for agents
.agents/                 topic-specific instructions, linked from this file
README.md                project intro for humans
yessir.sh                runner: ./yessir.sh run | browser | test | check | fix | format | uv <args>
pyproject.toml, uv.lock  one uv project; Python 3.12; package name "yessir"
yessir/                  shared code: vocabulary, alphabets, model, training, CLI/TUI, artefacts
experiments/sir_yes_sir/ minimal smoke-test and reference experiment (see .agents/sir-yes-sir.md)
experiments/addition/    the live addition experiment (see .agents/addition.md)
results/                 results of live experiments, one directory per experiment
studies/                 studies grouping runs around research questions (`studies/<experiment>/<slug>.json`)
web/                     static web interface (Astro, pnpm) for browsing results, studies, and configs
.prototypes/transformadder/  Louis's internal notes: the original, unreviewed exploration (code, tests, spec, notes, ~40 runs)
```

### Rules per directory

| Directory | Rule |
| :--- | :--- |
| `experiments/` | **Kept live.** It runs and `./yessir.sh test` passes. Code arrives in small pieces that Louis has reviewed. |
| `yessir/` | Only reviewed code that is **clearly shared by more than one experiment**. Don't create abstractions in advance. |
| `results/` | Only defensible results, or runs explicitly marked as flawed (see [`.agents/results.md`](.agents/results.md)). |
| `.prototypes/` | **Louis's internal notes. Allowed to rot.** Never mention it in any README. Reference only: read it, learn from it, point to it. Don't maintain it, don't fix it unless asked, and never copy it wholesale into `experiments/` or `yessir/`. Its results are leads, not findings. |

### Environment

- Use `./yessir.sh` for everything in the live repo. `./yessir.sh run` runs an experiment (opening the interactive config editor by default, or taking CLI flags); `./yessir.sh browser` starts the Astro dev server for `web/` (in agent shells Astro 7 auto-backgrounds and exits—use `./yessir.sh browser stop | status | logs`, or pass `--ignore-lock` to stay in the foreground); `./yessir.sh test` runs the tests in `experiments/` and `yessir/`; `./yessir.sh check` runs `ruff check` and `ty check`; `./yessir.sh fix` typechecks, autofixes lint issues, and formats; `./yessir.sh format` runs `ruff format`; `./yessir.sh uv <args>` runs any uv command (e.g. `./yessir.sh uv add numpy`).
- There is one venv per OS (`.venv-darwin`, `.venv-linux`), because the same checkout is used from a macOS host and a Linux VM. Don't set a global `UV_PROJECT_ENVIRONMENT`, and don't create or delete venvs unless asked.
- Prototypes keep their own run instructions (see each prototype's README).

### Where text goes

- **Instructions for agents** go in this file, or in a topic file under `.agents/` linked from here. There are no nested `AGENTS.md` files.
- **Every `README.md` is for a human new to the repo.** It describes what exists *now*. No plans or intentions ("will be", "being rebuilt"), no process, no instructions for agents. If a sentence wouldn't help a newcomer understand what's in front of them, it doesn't belong in a README.
- The same goes for module docstrings: describe the code, not repo policy.

### Topic files

- [`.agents/sir-yes-sir.md`](.agents/sir-yes-sir.md): the smoke-test and reference experiment (`experiments/sir_yes_sir/`).
- [`.agents/addition.md`](.agents/addition.md): working on the addition experiment.
- [`.agents/results.md`](.agents/results.md): what may go in `results/`, and how to mark a flawed run.

## Working with Louis

1. **A question is not a go-ahead.** If Louis asks a question or raises a concern ("shouldn't it…?", "wdyt?"), answer it and propose what you'd do, then stop. Act only on an explicit "yes" / "do it". His own design ideas are often better than the quick fix, so leave room for them.
2. **Restate big or ambiguous requests before acting.** For anything that restructures, deletes or spans many files, say back what you understood and wait for confirmation.
3. **Small, reviewable pieces.** In `experiments/` and `yessir/`, add one understandable piece at a time and stop for review before the next. Prefer plain, readable code over clever or general code. Explain *why* each piece is shaped the way it is, not just what it does. This is a learning project.
4. **Never commit.** Don't run `git commit` or anything else that creates commits, and don't offer to. Leave changes in the working tree and finish by summarising what changed and what's uncommitted.
5. **Report honestly.** Say what you verified and how, what you didn't verify, and when something failed. If you get a fact wrong, correct it plainly. Check numbers against the actual files before quoting them.
6. **Look before destroying.** Before deleting or overwriting anything, check what it is and make sure it's preserved elsewhere (e.g. in git). Prefer `git mv` for moves so history follows.
7. **Respect changes on disk.** If a file changed since you last read it, Louis probably changed it on purpose. Work from the current version and don't revert it.

## House style

- British English in docs, docstrings and comments (*generalisation*, *behaviour*).
- Plain text and code formatting only in docs, specs, and comments—never LaTeX maths (`$...$`, `\mathcal`, etc.).
- Follow standard AI research and transformer community naming conventions where they exist (e.g. `special_tokens`, `vocab_size`, `input_ids`, `position_ids`, `collate_fn`, `reference` / `oracle`). Avoid idiosyncratic home-brewed terms when an established research standard already exists, so the codebase is immediately recognisable and legible to researchers.
- Specs live next to the experiment they describe (`experiments/<name>/spec/`). A prototype's spec is a record of what it tested, not a plan.
