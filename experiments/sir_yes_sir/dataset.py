"""Data generator for the `sir_yes_sir` experiment (`SirYesSirDataset`)."""

import random
from collections.abc import Iterator

from experiments.sir_yes_sir.inputs import EmbeddingMode, build_sm_example
from experiments.sir_yes_sir.reference import reference_step
from yessir.alphabet import SymbolSet, create_alphabet
from yessir.dataset import LabelledExample
from yessir.vocab import Token


class SirYesSirDataset:
    """Infinite deterministic stream of state-machine `LabelledExample`s for `sir_yes_sir`.

    Each problem samples a fresh alphabet of size `alphabet_size` from `SymbolSet.ALL`
    and a random input sequence `x` of length `seq_length` over that alphabet, then
    yields one `LabelledExample` for each step `k = 0 .. seq_length - 1`.
    """

    def __init__(
        self,
        *,
        alphabet_size: int = 4,
        seq_length: int = 3,
        embedding_mode: EmbeddingMode = EmbeddingMode.TOKEN_TYPE_PE,
        seed: int = 42,
    ) -> None:
        if not (2 <= alphabet_size <= len(SymbolSet.ALL)):
            raise ValueError(
                f"alphabet_size must be in 2..{len(SymbolSet.ALL)}, got {alphabet_size}"
            )
        if seq_length < 1:
            raise ValueError(f"seq_length must be >= 1, got {seq_length}")

        self._alphabet_size = alphabet_size
        self._seq_length = seq_length
        self._embedding_mode = embedding_mode
        self._seed = seed

    def __iter__(self) -> Iterator[LabelledExample]:
        """Yield `LabelledExample`s deterministically from `self._seed`."""
        rng = random.Random(self._seed)
        while True:
            alphabet = create_alphabet(
                SymbolSet.ALL, sample_size=self._alphabet_size, rng=rng
            )
            x = tuple(rng.choice(alphabet) for _ in range(self._seq_length))

            output_history: list[Token] = []
            for k in range(self._seq_length):
                o_k, w_k, flag_k = reference_step(alphabet, x, k)
                example = build_sm_example(
                    alphabet,
                    x,
                    output_history,
                    embedding_mode=self._embedding_mode,
                )
                yield LabelledExample(
                    example=example,
                    targets=(o_k, w_k, int(flag_k)),
                )
                output_history.append(o_k)
