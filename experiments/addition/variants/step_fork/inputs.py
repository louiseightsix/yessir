"""Input sequence and step-label construction (`build_step_fork_branch_example`, `build_step_fork_examples`, `build_step_fork_step_labelled_examples`) for the `step_fork` variant of `addition`."""

from collections.abc import Sequence

from experiments.addition.inputs import (
    AdditionToken,
    EmbeddingMode,
    TokenType,
    build_prefix_tokens,
    make_example,
)
from experiments.addition.reference import ColumnStep, reference_add
from yessir.dataset import IGNORE_INDEX, LabelledExample
from yessir.example import Example
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token, TokenSequence

STEP_FORK_QUERY_TOKENS: tuple[AdditionToken, AdditionToken, AdditionToken] = (
    AdditionToken.QUERY_SUM,
    AdditionToken.QUERY_CARRY,
    AdditionToken.QUERY_FLAG,
)


def _build_step_fork_base_tokens(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    carry: Token,
    output: TokenSequence,
) -> tuple[list[Token], list[int], list[int]]:
    """Build the shared `step_fork` prefix + state lists prior to the trailing `[S]`/`[C]`/`[F]` query token."""
    input_ids, token_type_ids, position_ids = build_prefix_tokens(alphabet, a, b)
    out_tokens = tuple(output)
    k = len(out_tokens)

    # Incoming carry `c_k` at `(TokenType.CARRY, k)`, fourth `[SEP]` at `(TokenType.SPECIAL, 0)`,
    # and historical outputs `o_0, ..., o_{k-1}` at `(TokenType.OUTPUT, 0..k-1)`.
    input_ids.extend((int(carry), int(SpecialToken.SEP), *out_tokens))
    token_type_ids.extend(
        (
            TokenType.CARRY,
            TokenType.SPECIAL,
            *([TokenType.OUTPUT] * k),
        )
    )
    position_ids.extend((k, 0, *range(k)))
    return input_ids, token_type_ids, position_ids


def build_step_fork_branch_example(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    carry: Token,
    output: TokenSequence = (),
    *,
    query_token: AdditionToken,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> Example:
    """Build one branched query `Example` (`[S]`, `[C]`, or `[F]`) at column `k = len(output)` (`spec/variants/step_fork.md` §1)."""
    if query_token not in STEP_FORK_QUERY_TOKENS:
        raise ValueError(
            f"query_token must be one of {STEP_FORK_QUERY_TOKENS}, got {query_token!r}"
        )
    base_ids, base_tt, base_pos = _build_step_fork_base_tokens(
        alphabet, a, b, carry, output
    )
    k = len(tuple(output))
    base_ids.append(int(query_token))
    base_tt.append(TokenType.SPECIAL)
    base_pos.append(k)
    return make_example(base_ids, base_tt, base_pos, embedding_mode)


def build_step_fork_examples(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    carry: Token,
    output: TokenSequence = (),
    *,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> tuple[Example, Example, Example]:
    """Build the 3 parallel `(sum, carry, flag)` query `Example`s for column `k = len(output)`.

    Sequence layout at column `k` (`spec/variants/step_fork.md` §1):
        input_ids:       [BOS] alphabet [SEP] a [SEP] b [SEP]  c_k  [SEP]  o_0 ... o_{k-1}  [S]/[C]/[F]
        token_type_ids:    0      1       0   2   0   3   0     4     0     5  ...    5          0
        position_ids:      0    0..B-1    0 0..La 0 0..Lb 0     k     0     0  ...   k-1         k

    Why historical sum digits `o_j` sit at `position_id = j` (`0..k-1`) while `[S]`/`[C]`/`[F]` sits at `k`:
    Unlike `cls` (where the final historical output token `o_{k-1}` itself serves as
    the query position at `k >= 1`), `step_fork` appends an explicit trailing query
    token `[S]`, `[C]`, or `[F]` at `position_id = k`, aligning the query position
    directly with `a_k`, `b_k`, and `c_k` while keeping `o_j` indexed by its own
    column `j`.
    """
    base_ids, base_tt, base_pos = _build_step_fork_base_tokens(
        alphabet, a, b, carry, output
    )
    k = len(tuple(output))
    sum_ex, carry_ex, flag_ex = (
        make_example(
            [*base_ids, int(q_tok)],
            [*base_tt, TokenType.SPECIAL],
            [*base_pos, k],
            embedding_mode,
        )
        for q_tok in STEP_FORK_QUERY_TOKENS
    )
    return sum_ex, carry_ex, flag_ex


def build_step_fork_step_labelled_examples(
    alphabet: TokenSequence,
    a: TokenSequence,
    b: TokenSequence,
    k: int,
    *,
    steps: Sequence[ColumnStep] | None = None,
    embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
) -> tuple[LabelledExample, LabelledExample, LabelledExample]:
    """Build the 3 teacher-forced `(sum, carry, flag)` `LabelledExample`s for column `k` (`0 <= k <= K`).

    Each branch of length `P_k` masks all `P_k - 1` prefix positions with
    `IGNORE_INDEX` and supervises only the final query position (`P_k - 1`) via
    `SubspaceLMHead` + `next_token_cross_entropy` (`spec/variants/step_fork.md` §3).
    """
    oracle_steps = tuple(steps) if steps is not None else reference_add(alphabet, a, b)
    if not (0 <= k < len(oracle_steps)):
        raise IndexError(f"Step index k={k} out of range 0..{len(oracle_steps) - 1}")

    step = oracle_steps[k]
    output_prefix = tuple(s.sum_digit for s in oracle_steps[:k])
    sum_ex, carry_ex, flag_ex = build_step_fork_examples(
        alphabet,
        a,
        b,
        step.carry_in,
        output_prefix,
        embedding_mode=embedding_mode,
    )
    masked_prefix = (IGNORE_INDEX,) * (len(sum_ex) - 1)
    return (
        LabelledExample(example=sum_ex, labels=(*masked_prefix, step.sum_digit)),
        LabelledExample(example=carry_ex, labels=(*masked_prefix, step.carry_out)),
        LabelledExample(example=flag_ex, labels=(*masked_prefix, int(step.flag))),
    )
