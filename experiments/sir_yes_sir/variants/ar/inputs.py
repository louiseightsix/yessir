"""Autoregressive per-position label construction (`build_ar_labelled_example`) for the `ar` variant of `sir_yes_sir`."""

from experiments.sir_yes_sir.inputs import (
    EmbeddingMode,
    build_example,
    validate_problem,
)
from experiments.sir_yes_sir.reference import reference_step
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.special_tokens import SpecialToken
from yessir.vocab import TokenSequence


def build_ar_labelled_example(
    alphabet: TokenSequence,
    x: TokenSequence,
) -> LabelledExample:
    """Build a pure-LM (`input_ids`-only) teacher-forced next-token `LabelledExample` for `(alphabet, x)`.

    For a prompt prefix `[BOS] alphabet [SEP] x [SEP]` of length `P` and ground-truth
    targets `targets = (o_0, ..., o_{L-1}, EOS)` of length `L + 1`
    (`experiments/sir_yes_sir/spec/variants/ar.md` §2):
    - `example` is `Example(input_ids=prefix + (o_0, ..., o_{L-1}))` (length `P + L`,
      with `token_type_ids=None` and `position_ids=None`).
    - `labels` is `(IGNORE_INDEX,) * (P - 1) + targets` (length `P + L`), so
      `SubspaceLMHead` + `next_token_cross_entropy` supervise predictions from
      index `P - 1` (the second `[SEP]`, predicting `o_0`) through `o_{L-1}`
      (predicting `EOS`), while masking earlier prompt positions with `IGNORE_INDEX = -100`.
    """
    alphabet_tokens, x_tokens = validate_problem(alphabet, x)
    output = tuple(
        reference_step(alphabet_tokens, x_tokens, k)[0] for k in range(len(x_tokens))
    )
    targets = (*output, int(SpecialToken.EOS))
    example = build_example(
        alphabet_tokens,
        x_tokens,
        output=output,
        embedding_mode=EmbeddingMode.NOPE,
    )
    labels = (IGNORE_INDEX,) * (len(example) - len(targets)) + targets
    return LabelledExample(example=example, labels=labels)
