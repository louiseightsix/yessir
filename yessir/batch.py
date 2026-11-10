"""Right-padding batch collation primitives (`Batch`, `collate`)."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from yessir.example import ChannelType, Example
from yessir.special_tokens import SpecialToken

# Standard PyTorch cross-entropy ignore value for labels
IGNORE_INDEX = -100

PAD_VALUE_BY_TYPE = {
    ChannelType.COORDINATE: 0,
    ChannelType.MASK: 0,
    ChannelType.LABEL: IGNORE_INDEX,
}


@dataclass(frozen=True)
class Batch:
    """A right-padded batch of examples ready for the model.

    Attributes:
        input_ids: LongTensor of shape (batch_size, max_seq_len), padded with SpecialToken.PAD.
        lengths: LongTensor of shape (batch_size,) holding each sequence's unpadded length.
            In state-machine mode, the output heads read at index `lengths - 1`.
        channels: Mapping of channel name to LongTensor of shape (batch_size, max_seq_len).
    """

    input_ids: torch.Tensor
    lengths: torch.Tensor
    channels: dict[str, torch.Tensor]


def _pad_stack(
    rows: Sequence[Sequence[int]], pad_value: int, max_len: int
) -> torch.Tensor:
    out = torch.full((len(rows), max_len), pad_value, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def collate(examples: Sequence[Example]) -> Batch:
    """Right-pad a sequence of `Example` into a `Batch`.

    Why right-padding:
    - Under a causal attention mask, tokens at indices `0 .. length - 1` cannot
      attend to later positions, so trailing `PAD` tokens never affect real
      token representations and no padding mask is needed.
    - `[BOS]` stays fixed at index 0 for every example in the batch.
    - Side channels are padded according to their `ChannelType` (COORDINATE/MASK with 0,
      LABEL with -100).
    """
    if not examples:
        raise ValueError("cannot collate an empty sequence of Example")

    lengths = [len(ex) for ex in examples]
    max_len = max(lengths)

    first_channels = examples[0].channels
    for i, ex in enumerate(examples[1:], start=1):
        if set(ex.channels.keys()) != set(first_channels.keys()):
            raise ValueError(
                f"inconsistent side channels at example {i}: all examples in a batch "
                f"must have the same set of side channels. Expected {set(first_channels.keys())}, "
                f"got {set(ex.channels.keys())}"
            )

    batched_channels: dict[str, torch.Tensor] = {}
    for name, sample_channel in first_channels.items():
        pad_val = PAD_VALUE_BY_TYPE[sample_channel.channel_type]
        rows = [ex.channels[name].values for ex in examples]
        batched_channels[name] = _pad_stack(rows, pad_val, max_len)

    return Batch(
        input_ids=_pad_stack([ex.input_ids for ex in examples], SpecialToken.PAD, max_len),
        lengths=torch.tensor(lengths, dtype=torch.long),
        channels=batched_channels,
    )
