"""Special tokens: the fixed ids used for markers, padding and flags."""

from enum import IntEnum, unique


@unique
class SpecialToken(IntEnum):
    """Special tokens shared by every experiment, in vocab.SHARED_SPECIAL_RANGE.

    An experiment that needs special tokens of its own defines its own IntEnum
    with ids in vocab.EXPERIMENT_SPECIAL_RANGE.
    """

    SEP = 0xF0  # |            separates the sections of a prompt
    PAD = 0xF1  # [PAD]        fills a batched sequence out to the batch's length
    BOS = 0xF2  # [BOS]        beginning of sequence
    EOS = 0xF3  # [EOS]        end of sequence
    STEP = 0xF4  # [S]         end of a prompt: "give the output for this step"
    FLAG_CONTINUE = 0xF5  #    halting flag: more steps to come
    FLAG_FINISHED = 0xF6  #    halting flag: this was the last step
    # 0xF7 is unassigned.
