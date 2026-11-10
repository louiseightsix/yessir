"""Tests for the shared special tokens."""

from yessir.special_tokens import SpecialToken
from yessir.vocab import SHARED_SPECIAL_RANGE


def test_shared_special_tokens_are_in_the_shared_range():
    for token in SpecialToken:
        assert token in SHARED_SPECIAL_RANGE, f"{token.name} = 0x{token:02X}"
