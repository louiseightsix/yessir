"""Batched model inputs (`Batch`, `LeftPaddedBatch`) and padding collation (`collate`, `collate_left_padded`, `collate_labelled`)."""

import array
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch

from yessir.dataset import IGNORE_INDEX, LabelledExample
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

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "Batch":
        """Return a `Batch` with all present tensors on `device`."""
        return Batch(
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            lengths=self.lengths.to(device, non_blocking=non_blocking),
            token_type_ids=(
                self.token_type_ids.to(device, non_blocking=non_blocking)
                if self.token_type_ids is not None
                else None
            ),
            position_ids=(
                self.position_ids.to(device, non_blocking=non_blocking)
                if self.position_ids is not None
                else None
            ),
        )

    def pin_memory(self) -> "Batch":
        """Return a `Batch` with all present CPU tensors placed in pinned (page-locked) host memory."""
        return Batch(
            input_ids=self.input_ids.pin_memory(),
            lengths=self.lengths.pin_memory(),
            token_type_ids=(
                self.token_type_ids.pin_memory()
                if self.token_type_ids is not None
                else None
            ),
            position_ids=(
                self.position_ids.pin_memory()
                if self.position_ids is not None
                else None
            ),
        )


def _scatter_sequences(
    rows: Sequence[Sequence[int]],
    pad_value: int,
    max_len: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Pack unpadded `rows` into a `(len(rows), max_len)` LongTensor at `mask` positions."""
    out = torch.full((len(rows), max_len), pad_value, dtype=torch.long)
    buf = array.array("q", itertools.chain.from_iterable(rows))
    if buf:
        out[mask] = torch.frombuffer(buf, dtype=torch.long)
    return out


@dataclass(frozen=True)
class LeftPaddedBatch:
    """A left-padded batch of `Example`s aligned at their final token (`index -1`).

    Used to prefill `KVCache` for batched autoregressive decoding: every example's
    final prompt token sits at column `max_seq_len - 1`, while leading `PAD` slots
    (`0 .. max_seq_len - length - 1`) are masked out via `key_padding_mask`
    (`True` = real token, `False` = leading `PAD`).
    """

    input_ids: torch.Tensor
    key_padding_mask: torch.Tensor
    lengths: torch.Tensor
    token_type_ids: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "LeftPaddedBatch":
        """Return a `LeftPaddedBatch` with all present tensors on `device`."""
        return LeftPaddedBatch(
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            key_padding_mask=self.key_padding_mask.to(
                device, non_blocking=non_blocking
            ),
            lengths=self.lengths.to(device, non_blocking=non_blocking),
            token_type_ids=(
                self.token_type_ids.to(device, non_blocking=non_blocking)
                if self.token_type_ids is not None
                else None
            ),
            position_ids=(
                self.position_ids.to(device, non_blocking=non_blocking)
                if self.position_ids is not None
                else None
            ),
        )


def collate_left_padded(examples: Sequence[Example]) -> LeftPaddedBatch:
    """Left-pad a non-empty sequence of `Example`s into a `LeftPaddedBatch`."""
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

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    col_indices = torch.arange(max_len, dtype=torch.long).unsqueeze(0)
    pad_counts = (max_len - lengths_tensor).unsqueeze(1)
    key_padding_mask = col_indices >= pad_counts

    return LeftPaddedBatch(
        input_ids=_scatter_sequences(
            [ex.input_ids for ex in examples],
            SpecialToken.PAD,
            max_len,
            key_padding_mask,
        ),
        key_padding_mask=key_padding_mask,
        lengths=lengths_tensor,
        token_type_ids=(
            _scatter_sequences(type_rows, 0, max_len, key_padding_mask)
            if type_rows
            else None
        ),
        position_ids=(
            _scatter_sequences(pos_rows, 0, max_len, key_padding_mask)
            if pos_rows
            else None
        ),
    )


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

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    mask = torch.arange(max_len, dtype=torch.long).unsqueeze(
        0
    ) < lengths_tensor.unsqueeze(1)

    return Batch(
        input_ids=_scatter_sequences(
            [ex.input_ids for ex in examples], SpecialToken.PAD, max_len, mask
        ),
        lengths=lengths_tensor,
        token_type_ids=(
            _scatter_sequences(type_rows, 0, max_len, mask) if type_rows else None
        ),
        position_ids=(
            _scatter_sequences(pos_rows, 0, max_len, mask) if pos_rows else None
        ),
    )


def collate_labelled(items: Sequence[LabelledExample]) -> tuple[Batch, torch.Tensor]:
    """Right-pad per-position `LabelledExample`s into `(Batch, labels)`.

    Each `item` must have one label entry per sequence position
    (`len(item.labels) == len(item.example)`):
    - 1D per-example labels (`tuple[int, ...]`, for `SubspaceLMHead`) produce a
      `labels` LongTensor of shape `(batch_size, max_seq_len)`.
    - 2D per-example labels (`tuple[tuple[int, ...], ...]`, for per-position
      `ParallelClassificationHead`) produce a `labels` LongTensor of shape
      `(batch_size, max_seq_len, num_targets)`.
    Positions at `>= length` are right-padded with `IGNORE_INDEX = -100`.

    Args:
        items: Non-empty sequence of per-position `LabelledExample`s.

    Returns:
        Tuple `(batch, labels)` where `batch` is a right-padded `Batch` and
        `labels` is right-padded with `IGNORE_INDEX`.
    """
    if not items:
        raise ValueError("cannot collate an empty sequence of LabelledExample")

    for idx, item in enumerate(items):
        if len(item.labels) != len(item.example):
            raise ValueError(
                f"per-position item {idx} has {len(item.labels)} labels "
                f"for example of length {len(item.example)}"
            )

    batch = collate([item.example for item in items])
    max_len = int(batch.input_ids.size(1))

    label_shapes = [
        len(item.labels[0]) if isinstance(item.labels[0], tuple) else 0
        for item in items
    ]
    if len(set(label_shapes)) > 1:
        raise ValueError(
            "all per-position items in a batch must share the same label dimensionality"
        )

    mask = torch.arange(max_len, dtype=torch.long).unsqueeze(
        0
    ) < batch.lengths.unsqueeze(1)
    num_targets = label_shapes[0]
    if num_targets > 0:
        labels = torch.full(
            (len(items), max_len, num_targets), IGNORE_INDEX, dtype=torch.long
        )
        rows_2d = (cast("Sequence[Sequence[int]]", item.labels) for item in items)
        buf = array.array(
            "q",
            itertools.chain.from_iterable(itertools.chain.from_iterable(rows_2d)),
        )
        if buf:
            labels[mask] = torch.frombuffer(buf, dtype=torch.long).view(-1, num_targets)
        return batch, labels

    labels_1d = [cast("Sequence[int]", item.labels) for item in items]
    return batch, _scatter_sequences(labels_1d, IGNORE_INDEX, max_len, mask)
