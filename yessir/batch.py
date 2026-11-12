"""Batched model inputs (`Batch`) and right-padding collation (`collate`)."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from yessir.example import Example
from yessir.special_tokens import SpecialToken


@dataclass(frozen=True)
class Batch:
    """A right-padded batch of `Example`s ready for the model's embedding layer.

    Attributes:
        input_ids: LongTensor of shape (batch_size, max_seq_len), right-padded
            with `SpecialToken.PAD`.
        lengths: LongTensor of shape (batch_size,) with each example's unpadded
            length. Output heads and losses use `lengths` (or an ignore mask) to
            discard outputs at padded positions `>= length`.
        token_type_ids: LongTensor of shape (batch_size, max_seq_len) right-padded
            with 0, or None when examples omit `token_type_ids`.
        position_ids: LongTensor of shape (batch_size, max_seq_len) right-padded
            with 0, or None when examples omit `position_ids` (NoPE).
    """

    input_ids: torch.Tensor
    lengths: torch.Tensor
    token_type_ids: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None


def _pad_sequences(
    rows: Sequence[Sequence[int]], pad_value: int, max_len: int
) -> torch.Tensor:
    out = torch.full((len(rows), max_len), pad_value, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def collate(examples: Sequence[Example]) -> Batch:
    """Right-pad a non-empty sequence of `Example`s into a `Batch`.

    Why right-padding:
    - Under a causal attention mask, real tokens at indices `0 .. length - 1`
      cannot attend to later positions, so trailing `PAD` tokens do not affect
      real token representations and no attention padding mask is needed.
    - `[BOS]` stays fixed at index 0 for every example in the batch.
    - Padded positions in `token_type_ids` and `position_ids` are filled with 0
      so embedding lookups stay in bounds. Those padded positions still run
      through the model and produce outputs, so output heads and loss functions
      must ignore positions `>= length` (e.g. by reading at `lengths - 1` or
      masking target labels with `-100`).
    """
    if not examples:
        raise ValueError("cannot collate an empty sequence of Example")

    lengths = [len(ex) for ex in examples]
    max_len = max(lengths)

    type_rows = [ex.token_type_ids for ex in examples if ex.token_type_ids is not None]
    if type_rows and len(type_rows) != len(examples):
        raise ValueError(
            "all examples in a batch must either provide token_type_ids or omit them"
        )

    pos_rows = [ex.position_ids for ex in examples if ex.position_ids is not None]
    if pos_rows and len(pos_rows) != len(examples):
        raise ValueError(
            "all examples in a batch must either provide position_ids or omit them"
        )

    return Batch(
        input_ids=_pad_sequences(
            [ex.input_ids for ex in examples], SpecialToken.PAD, max_len
        ),
        lengths=torch.tensor(lengths, dtype=torch.long),
        token_type_ids=_pad_sequences(type_rows, 0, max_len) if type_rows else None,
        position_ids=_pad_sequences(pos_rows, 0, max_len) if pos_rows else None,
    )
