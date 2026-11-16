# Addition

**Can a transformer trained to do long addition in a few bases, each defined in the prompt, add in bases it never saw in training?**

See [`spec/task.md`](spec/task.md) for the task specification, [`reference.py`](reference.py) for the ground-truth column-addition oracle, [`inputs.py`](inputs.py) for shared tokens, sequence prefixes, and problem sampling, [`evaluation.py`](evaluation.py) for shared evaluation metrics and the first-error taxonomy, [`config.py`](config.py) for experiment configuration and evaluation-split schemas, [`train_and_evaluate.py`](train_and_evaluate.py) and [`run.py`](run.py) for the training/evaluation runner and CLI entry point, and [`variants/cls/`](variants/cls/) for the per-step parallel classification variant ([`spec/variants/cls.md`](spec/variants/cls.md)).
