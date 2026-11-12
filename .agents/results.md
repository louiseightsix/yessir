# Results

Everything in `results/` is either a defensible result or explicitly marked as flawed.

To mark a run as flawed, put a `README.md` in its directory whose first line is

```
FLAWED: <one-line reason>
```

followed by any detail: what is wrong, what it affects, and whether a corrected run exists. Flawed runs stay, so the history is honest, and they are never cited without their annotation.

Prototype results are not in `results/`. They live with their prototype under `.prototypes/` and carry no such guarantee.

## Studies (`studies/`)

`studies/<experiment>/<slug>.json` files group zero or more runs from `results/<experiment>/` around a research question (`Study` in `yessir/artefacts.py`). A study has a `status` (`draft`, `in_progress`, or `concluded`) and an optional `kind` (`exploratory`, `ablation`, `controlled_comparison`, or `sweep`). Because studies reference run directory names rather than containing them, a single run can belong to multiple studies, and a `draft` study can be recorded before any runs exist.
