# yessir

*"Yes sir."* Whatever the rule says.

**yessir** is a set of experiments on whether transformers can follow a rule that is **given to them in context**. That means doing what the prompt says even when it isn't what training made familiar, and still doing it on inputs bigger or stranger than anything seen in training.

Each experiment picks one small, exact rule. The rule's parameters are stated in the prompt, and the experiment asks:

- does the model actually *read* the rule, or did it memorise the training cases?
- does it keep following the rule on unseen parameters and longer inputs, zero-shot?
- when it fails, did it misread the rule or mis-execute it?

## Experiments

| Experiment | Rule | Status |
| :--- | :--- | :--- |
| [`addition`](experiments/addition/) | Column addition in a numeral system defined in the prompt (an arbitrary ordered alphabet of digit symbols, so any base) | Active: three rounds of runs |

## Layout

```
yessir/          shared code, kept to what is clearly experiment-agnostic
experiments/     one directory per experiment: code, tests, spec, notes
results/         run outputs, one directory per experiment
```

Code moves into `yessir/` only once more than one experiment needs it.

## Setup

```bash
uv venv && uv pip install -e '.[dev]'    # or: python -m venv .venv && pip install -e '.[dev]'
pytest                                   # runs every experiment's tests
```

Each experiment's README explains how to run it.
