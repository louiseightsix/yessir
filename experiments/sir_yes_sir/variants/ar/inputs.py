"""Autoregressive sequence-label construction (`build_ar_labelled_example`) for the `ar` variant of `sir_yes_sir`."""

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
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> LabelledExample:
    """Build a teacher-forced next-token `LabelledExample` for `(alphabet, x)`.

    For a prompt prefix `[BOS] alphabet [SEP] x [SEP]` of length `P` and ground-truth
    targets `targets = (o_0, ..., o_{L-1}, EOS)` of length `L + 1`
    (`experiments/sir_yes_sir/spec/variants/ar.md` §2):
    - `example` is `build_example(alphabet, x, output=(o_0, ..., o_{L-1}))` (length `P + L`).
    - `labels` is `(IGNORE_INDEX,) * (P - 1) + targets` (length `P + L`), so
      `SubspaceLMHead` + `next_token_cross_entropy` supervise predictions from
      index `P - 1` (the second `[SEP]` at `position_id = 0`, predicting `o_0`)
      through `o_{L-1}` (at `position_id = L`, predicting `EOS`), while masking
      earlier prompt positions with `IGNORE_INDEX = -100`.
    """
    alphabet_tokens, x_tokens = validate_problem(alphabet, x)
    outputs = tuple(
        reference_step(alphabet_tokens, x_tokens, k)[0] for k in range(len(x_tokens))
    )
    targets = (*outputs, int(SpecialToken.EOS))
    example = build_example(
        alphabet_tokens,
        x_tokens,
        output=outputs,
        embedding_mode=embedding_mode,
    )
    labels = (IGNORE_INDEX,) * (len(example) - len(targets)) + targets
    return LabelledExample(example=example, labels=labels)
