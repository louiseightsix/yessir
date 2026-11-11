"""Byte-level vocabulary: the token types, the vocabulary size, and how it is split."""

from collections.abc import Sequence
from typing import Annotated

# A token is one byte, held as an int in 0..255. There is no separate string
# form, so the token and its id are the same thing. In Hugging Face terms,
# a Token is both the token and its token id, and a TokenSequence is what
# they would call `input_ids`.
type Token = Annotated[int, "Value in 0..255"]
type TokenSequence = Sequence[Token] | bytes

# A TokenTypeId marks which section of an input a token belongs to (like BERT's
# segment / token_type_ids). Each experiment defines its own IntEnum of section
# types with values in TOKEN_TYPE_RANGE (0..15).
MAX_TOKEN_TYPES = 16
TOKEN_TYPE_RANGE = range(0, MAX_TOKEN_TYPES)  # noqa: PIE808
type TokenTypeId = Annotated[int, "Value in 0..15 (TOKEN_TYPE_RANGE)"]

# ---------------------------------------------------------------------------
# The vocabulary and how it is split
# ---------------------------------------------------------------------------
#
# Every token is one byte, so the model's vocabulary is the 256 byte values.
# They are split into contiguous ranges:
#
#   0x00                     0xEF  0xF0            0xF7  0xF8            0xFF
#  ┌─────────────────────────────┬─────────────────────┬─────────────────────┐
#  │         Symbol pool         │   Shared special    │ Experiment special  │
#  │         (240 bytes)         │      (8 bytes)      │      (8 bytes)      │
#  └─────────────────────────────┴─────────────────────┴─────────────────────┘
#                                └────────────── Special tokens ─────────────┘
#
# Alphabets are drawn from the symbol pool. Special tokens (BOS, PAD, ...) have
# fixed ids. The shared range holds the ones every experiment uses; the
# experiment range is reserved for ones that only a single experiment needs.

VOCAB_SIZE = 256

SYMBOL_RANGE = range(0x00, 0xF0)  # noqa: PIE808 (explicit start, to match lines below)
SPECIAL_RANGE = range(0xF0, 0x100)
SHARED_SPECIAL_RANGE = range(0xF0, 0xF8)
EXPERIMENT_SPECIAL_RANGE = range(0xF8, 0x100)

assert (
    len(SYMBOL_RANGE) + len(SHARED_SPECIAL_RANGE) + len(EXPERIMENT_SPECIAL_RANGE)
    == VOCAB_SIZE
), "Vocabulary partition ranges must exactly cover VOCAB_SIZE"
