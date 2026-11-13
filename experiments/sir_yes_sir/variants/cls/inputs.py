"""Parallel classification label construction (`build_cls_labelled_example`, `build_cls_step_labelled_example`) for the `cls` variant of `sir_yes_sir`."""

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    build_example,
    validate_problem,
)
from experiments.sir_yes_sir.reference import reference_step
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.vocab import TokenSequence


def build_cls_labelled_example(
    alphabet: TokenSequence,
    x: TokenSequence,
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> LabelledExample:
    """Build a single-pass teacher-forced `LabelledExample` for `(alphabet, x)`.

    For a prompt prefix `[BOS] alphabet [SEP] x [SEP]` of length `P` and `L` step
    targets `steps = ((o_0, flag_0), ..., (o_{L-1}, flag_{L-1}))`
    (`experiments/sir_yes_sir/spec/variants/cls.md` §2):
    - `example` is `build_example(alphabet, x, output=(o_0, ..., o_{L-2}))`
      (length `P + L - 1`).
    - `labels` has `((IGNORE_INDEX, IGNORE_INDEX),) * (P - 1) + steps` (length
      `P + L - 1`), so `ParallelClassificationHead` (`lengths=None`) +
      `parallel_classification_cross_entropy` supervise `heads[0..1]` at every
      query position `P - 1 .. P + L - 2` (the second `[SEP]` at `position_id = 0`
      for `k == 0`, and `o_{k-1}` at `position_id = k` for `k >= 1`) in a single
      forward pass, while masking earlier prompt positions with `IGNORE_INDEX = -100`.
    """
    alphabet_tokens, x_tokens = validate_problem(alphabet, x)
    steps = tuple(
        (o_k, int(flag_k))
        for o_k, flag_k in (
            reference_step(alphabet_tokens, x_tokens, k) for k in range(len(x_tokens))
        )
    )
    output = tuple(o_k for o_k, _ in steps[:-1])
    example = build_example(
        alphabet_tokens,
        x_tokens,
        output=output,
        embedding_mode=embedding_mode,
    )
    ignored_prefix = ((IGNORE_INDEX, IGNORE_INDEX),) * (len(example) - len(steps))
    return LabelledExample(example=example, labels=ignored_prefix + steps)


def build_cls_step_labelled_example(
    alphabet: TokenSequence,
    x: TokenSequence,
    k: int,
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> LabelledExample:
    """Build a single-step `LabelledExample` for step `k` (`0 <= k < len(x)`).

    Used by the `all_steps` and `random_step` supervision modes
    (`studies/sir_yes_sir/cls-training-passes.json`):
    - `example` is `build_example(alphabet, x, output=(o_0, ..., o_{k-1}))`
      (ending at the step-`k` query token at index `lengths - 1`).
    - `labels` is the 1D per-head tuple `(o_k, int(flag_k))`.
    """
    alphabet_tokens, x_tokens = validate_problem(alphabet, x)
    o_k, flag_k = reference_step(alphabet_tokens, x_tokens, k)
    output = tuple(reference_step(alphabet_tokens, x_tokens, j)[0] for j in range(k))
    example = build_example(
        alphabet_tokens,
        x_tokens,
        output=output,
        embedding_mode=embedding_mode,
    )
    return LabelledExample(example=example, labels=(o_k, int(flag_k)))
