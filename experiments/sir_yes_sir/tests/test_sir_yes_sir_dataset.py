"""Tests for `experiments/sir_yes_sir/variants/cls/dataset.py` and `experiments/sir_yes_sir/variants/ar/dataset.py`."""

import itertools

import pytest

from experiments.sir_yes_sir.config import ClsTrainingMode
from experiments.sir_yes_sir.inputs import EmbeddingMode, TokenType
from experiments.sir_yes_sir.reference import reference_step
from experiments.sir_yes_sir.variants.ar.dataset import (
    SirYesSirArDataset,
)
from experiments.sir_yes_sir.variants.ar.dataset import (
    main as ar_main,
)
from experiments.sir_yes_sir.variants.cls.dataset import SirYesSirClsDataset, main
from yessir.alphabet import SymbolSet
from yessir.dataset import IGNORE_INDEX
from yessir.special_tokens import SpecialToken


def test_dataset_reproducibility_and_seed_variation():
    ds = SirYesSirClsDataset(alphabet_size=4, x_length=3, seed=123)
    run1 = list(itertools.islice(ds, 6))
    run2 = list(itertools.islice(ds, 6))
    assert run1 == run2

    ds_other = SirYesSirClsDataset(alphabet_size=4, x_length=3, seed=456)
    run_other = list(itertools.islice(ds_other, 6))
    assert run1 != run_other


def test_dataset_matches_reference_step_and_accumulates_output_history():
    alphabet_size = 5
    x_length = 4
    ds = SirYesSirClsDataset(
        alphabet_size=alphabet_size,
        x_length=x_length,
        seed=7,
    )
    item = next(iter(ds))

    # Extract alphabet and input sequence x from the example's sections:
    # [BOS] alphabet[0..B-1] [SEP] x[0..L-1] [SEP] o_0 .. o_{L-2}
    ids = item.example.input_ids
    alphabet = ids[1 : 1 + alphabet_size]
    x_start = 1 + alphabet_size + 1
    x = ids[x_start : x_start + x_length]
    prefix_len = x_start + x_length + 1

    expected_steps = tuple(
        (o_k, int(flag_k))
        for o_k, flag_k in (reference_step(alphabet, x, k) for k in range(x_length))
    )
    expected_outputs = tuple(o_k for o_k, _ in expected_steps[:-1])
    assert ids[prefix_len:] == expected_outputs
    assert item.labels[: prefix_len - 1] == ((IGNORE_INDEX, IGNORE_INDEX),) * (
        prefix_len - 1
    )
    assert item.labels[prefix_len - 1 :] == expected_steps


@pytest.mark.parametrize(
    "mode",
    [
        EmbeddingMode.NOPE,
        EmbeddingMode.TOKEN_TYPE_NOPE,
        EmbeddingMode.TOKEN_TYPE_PE,
    ],
)
def test_dataset_respects_embedding_mode(mode: EmbeddingMode):
    ds = SirYesSirClsDataset(
        alphabet_size=3,
        x_length=2,
        embedding_mode=mode,
        seed=0,
    )
    item = next(iter(ds))
    ex = item.example

    if mode == EmbeddingMode.NOPE:
        assert ex.token_type_ids is None
        assert ex.position_ids is None
    elif mode == EmbeddingMode.TOKEN_TYPE_NOPE:
        assert ex.token_type_ids is not None
        assert ex.token_type_ids[-1] == TokenType.OUTPUT
        assert ex.position_ids is None
    else:
        assert ex.token_type_ids is not None
        assert ex.position_ids is not None


def test_dataset_samples_only_from_given_symbols_pool():
    pool = SymbolSet.DECIMAL[:3]
    ds = SirYesSirClsDataset(alphabet_size=3, x_length=2, symbols=pool, seed=42)
    items = list(itertools.islice(ds, 12))
    for item in items[::2]:
        alphabet = item.example.input_ids[1:4]
        assert set(alphabet) == set(pool)


def test_dataset_canonical_order_when_permute_alphabet_is_false():
    pool = SymbolSet.DECIMAL[:4]
    ds = SirYesSirClsDataset(
        alphabet_size=4,
        x_length=3,
        symbols=pool,
        permute_alphabet=False,
        seed=42,
    )
    for item in itertools.islice(ds, 6):
        alphabet = item.example.input_ids[1:5]
        assert alphabet == tuple(pool)


def test_cls_dataset_all_steps_and_random_step_modes():
    ds_all = SirYesSirClsDataset(
        alphabet_size=4,
        x_length=3,
        training_mode=ClsTrainingMode.ALL_STEPS,
        seed=11,
    )
    step_items = list(itertools.islice(ds_all, 3))
    # All 3 step examples come from the same problem and have 1D (o_k, flag_k) labels.
    prefix_len = 1 + 4 + 1 + 3 + 1
    for k, step_item in enumerate(step_items):
        assert len(step_item.example) == prefix_len + k
        assert len(step_item.labels) == 2
        assert isinstance(step_item.labels[0], int)

    ds_rand = SirYesSirClsDataset(
        alphabet_size=4,
        x_length=3,
        training_mode=ClsTrainingMode.RANDOM_STEP,
        seed=11,
    )
    rand_item = next(iter(ds_rand))
    # First problem's alphabet and x match ds_all's first problem.
    assert (
        rand_item.example.input_ids[:prefix_len]
        == (step_items[0].example.input_ids[:prefix_len])
    )
    assert len(rand_item.labels) == 2
    assert isinstance(rand_item.labels[0], int)

    # Verify `step_rng` is decoupled from `iter_problems` on problem 0 (when B == L and
    # permute_alphabet=False, sharing the same seed would force k == index(x[0]) for every seed).
    decoupled = False
    for s in range(20):
        item_s = next(
            iter(
                SirYesSirClsDataset(
                    alphabet_size=3,
                    x_length=3,
                    symbols=SymbolSet.DECIMAL[:3],
                    permute_alphabet=False,
                    training_mode=ClsTrainingMode.RANDOM_STEP,
                    seed=s,
                )
            )
        )
        # Prefix length is 1 + 3 + 1 + 3 + 1 = 9; output history length is k.
        k = len(item_s.example) - 9
        x0_rank = SymbolSet.DECIMAL[:3].index(item_s.example.input_ids[5])
        if k != x0_rank:
            decoupled = True
            break
    assert decoupled


@pytest.mark.parametrize(
    ("alphabet_size", "x_length", "symbols", "match"),
    [
        (1, 3, SymbolSet.ALL, "alphabet_size must be in 2..240"),
        (241, 3, SymbolSet.ALL, "alphabet_size must be in 2..240"),
        (4, 3, SymbolSet.DECIMAL[:3], "alphabet_size must be in 2..3"),
        (4, 0, SymbolSet.ALL, "x_length must be >= 1"),
        (4, -2, SymbolSet.ALL, "x_length must be >= 1"),
    ],
)
def test_dataset_rejects_invalid_init_args(alphabet_size, x_length, symbols, match):
    with pytest.raises(ValueError, match=match):
        SirYesSirClsDataset(
            alphabet_size=alphabet_size, x_length=x_length, symbols=symbols
        )


def test_dataset_cli_main_prints_examples(capsys: pytest.CaptureFixture[str]):
    main(["-n", "2", "-B", "3", "-L", "2", "--embedding-mode", "nope", "--seed", "9"])
    out = capsys.readouterr().out
    assert "Example 0:" in out
    assert "Example 1:" in out
    assert "token_type_ids" not in out


def test_ar_dataset_yields_one_labelled_example_per_problem_and_cli(
    capsys: pytest.CaptureFixture[str],
):
    alphabet_size = 4
    x_length = 3
    ds = SirYesSirArDataset(
        alphabet_size=alphabet_size,
        x_length=x_length,
        seed=17,
    )
    item = next(iter(ds))
    prefix_len = 1 + alphabet_size + 1 + x_length + 1
    ids = item.example.input_ids
    alphabet = ids[1 : 1 + alphabet_size]
    x = ids[1 + alphabet_size + 1 : prefix_len - 1]

    expected_outputs = tuple(reference_step(alphabet, x, k)[0] for k in range(x_length))
    expected_targets = (*expected_outputs, int(SpecialToken.EOS))
    assert item.labels[: prefix_len - 1] == (IGNORE_INDEX,) * (prefix_len - 1)
    assert item.labels[prefix_len - 1 :] == expected_targets

    ar_main(["-n", "1", "-B", "3", "-L", "2"])
    out = capsys.readouterr().out
    assert "Example 0:" in out
    assert "[-100]" in out
