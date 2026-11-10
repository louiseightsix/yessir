# AGENTS.md

Guidance for AI coding agents working in this repo. Read it before doing anything.

## What this project is for

**yessir** is a set of experiments on whether transformers can follow a rule that is given to them **in context**, rather than one memorised from training. The first experiment is **addition**: column addition in a numeral system defined in the prompt by an ordered alphabet of arbitrary digit symbols (so any base). It asks whether a model trained on a few bases can add zero-shot in unseen bases and at longer lengths.

**The real goal is for Louis to learn how this works**: transformers, training, and running careful experiments. A working result produced by code he hasn't understood is worth little here. The first version of the code was written quickly by an agent and never reviewed. It now lives in `prototypes/transformadder/` as a reference, and the live experiment is being rebuilt **one small, reviewed piece at a time**.

Each experiment stands on its own. Don't frame addition as "a step toward ARC-AGI" or as one instance of a generic rule-following framework.

## Repo layout

```
AGENTS.md                this file
README.md                project intro and the repo rules
yessir.sh                runner: ./yessir.sh test [pytest args] | ./yessir.sh uv <uv args>
pyproject.toml, uv.lock  one uv project; Python 3.12; package name "yessir"
yessir/                  shared code (currently empty)
experiments/addition/    the live addition experiment (currently just a README)
results/                 results of live experiments, one directory per experiment
prototypes/transformadder/   the original, unreviewed exploration: code, tests, spec, notes, ~40 runs
```

### Rules per directory

| Directory | Rule |
| :--- | :--- |
| `experiments/` | **Kept live.** It runs and `./yessir.sh test` passes. Code arrives in small pieces that Louis has reviewed. |
| `yessir/` | Only reviewed code that is **clearly shared by more than one experiment**. Don't create abstractions in advance. |
| `results/` | Only results Louis would stand by. A flawed run keeps a `README.md` in its directory whose first line is `FLAWED: <reason>` (see `results/README.md`). Never present a flawed result without its annotation. |
| `prototypes/` | **Allowed to rot.** Reference only: read it, learn from it, point to it. Don't maintain it, don't fix it unless asked, and never copy it wholesale into `experiments/` or `yessir/`. Its results are leads, not findings. |

### Environment

- Use `./yessir.sh` for everything in the live repo. `./yessir.sh test` runs the tests in `experiments/`; `./yessir.sh uv <args>` runs any uv command (e.g. `./yessir.sh uv add numpy`).
- There is one venv per OS (`.venv-darwin`, `.venv-linux`), because the same checkout is used from a macOS host and a Linux VM. Don't set a global `UV_PROJECT_ENVIRONMENT`, and don't create or delete venvs unless asked.
- Plain `pytest` collects nothing until the first live tests exist. That's expected.
- Prototypes keep their own run instructions (see each prototype's README).

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
- Specs live next to the experiment they describe (`experiments/<name>/spec/`). The live addition spec hasn't been written yet; `prototypes/transformadder/spec/` is a record of what the prototype tested, not the plan.
