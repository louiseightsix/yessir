"""Tests for `experiments/sir_yes_sir/variants/sm_mh/dataset.py` and `experiments/sir_yes_sir/variants/ar/dataset.py`."""

import itertools

import pytest

from experiments.sir_yes_sir.inputs import EmbeddingMode, TokenType
from experiments.sir_yes_sir.reference import reference_step
from experiments.sir_yes_sir.variants.ar.dataset import (
    SirYesSirArDataset,
)
from experiments.sir_yes_sir.variants.ar.dataset import (
    main as ar_main,
)
from experiments.sir_yes_sir.variants.ar.inputs import StepOrder, build_ar_trace
from experiments.sir_yes_sir.variants.sm_mh.dataset import SirYesSirSmMhDataset, main
from yessir.alphabet import SymbolSet
from yessir.dataset import IGNORE_INDEX
from yessir.special_tokens import SpecialToken


def test_dataset_reproducibility_and_seed_variation():
    ds = SirYesSirSmMhDataset(alphabet_size=4, x_length=3, seed=123)
    run1 = list(itertools.islice(ds, 6))
    run2 = list(itertools.islice(ds, 6))
    assert run1 == run2

    ds_other = SirYesSirSmMhDataset(alphabet_size=4, x_length=3, seed=456)
    run_other = list(itertools.islice(ds_other, 6))
    assert run1 != run_other


def test_dataset_matches_reference_step_and_accumulates_output_history():
    alphabet_size = 5
    x_length = 4
    ds = SirYesSirSmMhDataset(
        alphabet_size=alphabet_size,
        x_length=x_length,
        seed=7,
    )
    items = list(itertools.islice(ds, x_length))

    # Extract alphabet and input sequence x from the first example's sections:
    # [BOS] alphabet[0..B-1] [SEP] x[0..L-1] [SEP] out_history[0..k-1] [STEP]
    first_ids = items[0].example.input_ids
    alphabet = first_ids[1 : 1 + alphabet_size]
    x_start = 1 + alphabet_size + 1
    x = first_ids[x_start : x_start + x_length]
    out_start = x_start + x_length + 1

    expected_history: list[int] = []
    for k, item in enumerate(items):
        o_k, w_k, flag_k = reference_step(alphabet, x, k)
        assert item.labels == (o_k, w_k, int(flag_k))

        ids = item.example.input_ids
        assert ids[out_start:-1] == tuple(expected_history)
        assert ids[-1] == SpecialToken.STEP

        expected_flag = (
            SpecialToken.FLAG_FINISHED
            if k == x_length - 1
            else SpecialToken.FLAG_CONTINUE
        )
        assert item.labels[2] == int(expected_flag)
        expected_history.append(o_k)


@pytest.mark.parametrize(
    "mode",
    [
        EmbeddingMode.NOPE,
        EmbeddingMode.TOKEN_TYPE_NOPE,
        EmbeddingMode.TOKEN_TYPE_PE,
    ],
)
def test_dataset_respects_embedding_mode(mode: EmbeddingMode):
    ds = SirYesSirSmMhDataset(
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
    ds = SirYesSirSmMhDataset(alphabet_size=3, x_length=2, symbols=pool, seed=42)
    items = list(itertools.islice(ds, 12))
    for item in items[::2]:
        alphabet = item.example.input_ids[1:4]
        assert set(alphabet) == set(pool)


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
        SirYesSirSmMhDataset(
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
        step_order=StepOrder.WRAP_FIRST,
        seed=17,
    )
    item = next(iter(ds))
    prefix_len = 1 + alphabet_size + 1 + x_length + 1
    ids = item.example.input_ids
    alphabet = ids[1 : 1 + alphabet_size]
    x = ids[1 + alphabet_size + 1 : prefix_len - 1]

    expected_trace = build_ar_trace(alphabet, x, step_order=StepOrder.WRAP_FIRST)
    assert item.labels[: prefix_len - 1] == (IGNORE_INDEX,) * (prefix_len - 1)
    assert item.labels[prefix_len - 1 :] == expected_trace

    ar_main(["-n", "1", "-B", "3", "-L", "2", "--step-order", "wrap_first"])
    out = capsys.readouterr().out
    assert "Example 0:" in out
    assert "[-100]" in out
