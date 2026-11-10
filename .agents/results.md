# Results

Everything in `results/` is either a defensible result or explicitly marked as flawed.

To mark a run as flawed, put a `README.md` in its directory whose first line is

```
FLAWED: <one-line reason>
```

followed by any detail: what is wrong, what it affects, and whether a corrected run exists. Flawed runs stay, so the history is honest, and they are never cited without their annotation.

Prototype results are not in `results/`. They live with their prototype under `.prototypes/` and carry no such guarantee.
