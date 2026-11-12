"""Tests for `yessir/dataset.py` and `yessir/inspect_data.py`."""

import json
from pathlib import Path

import pytest

from yessir.dataset import LabelledExample
from yessir.example import Example
from yessir.inspect_data import (
    format_labelled_example,
    format_token,
    inspect_dataset,
)
from yessir.special_tokens import SpecialToken


def test_labelled_example_valid_construction():
    ex = Example(input_ids=(1, 2, 3), token_type_ids=(0, 1, 2), position_ids=(0, 0, 0))
    item = LabelledExample(
        example=ex,
        targets=(10, 20, int(SpecialToken.FLAG_FINISHED)),
    )
    assert item.example == ex
    assert item.targets == (10, 20, int(SpecialToken.FLAG_FINISHED))


@pytest.mark.parametrize("bad_example", [None, (1, 2, 3), [1, 2]])
def test_labelled_example_rejects_non_example(bad_example):
    with pytest.raises(TypeError, match="example must be an Example"):
        LabelledExample(example=bad_example, targets=(1,))


@pytest.mark.parametrize("bad_targets", [[1, 2], "ab", 123])
def test_labelled_example_rejects_non_tuple_targets(bad_targets):
    ex = Example(input_ids=(1, 2))
    with pytest.raises(TypeError, match="targets must be a tuple"):
        LabelledExample(example=ex, targets=bad_targets)


def test_labelled_example_rejects_empty_targets():
    ex = Example(input_ids=(1, 2))
    with pytest.raises(ValueError, match="targets must not be empty"):
        LabelledExample(example=ex, targets=())


@pytest.mark.parametrize("bad_token", [-1, 256, True, False, "a", 1.5])
def test_labelled_example_rejects_invalid_target_token(bad_token):
    ex = Example(input_ids=(1, 2))
    with pytest.raises(ValueError, match="not a valid target token id"):
        LabelledExample(example=ex, targets=(bad_token,))


def test_format_token_special_and_hex():
    assert format_token(int(SpecialToken.BOS)) == "[BOS]"
    assert format_token(int(SpecialToken.FLAG_CONTINUE)) == "[FLAG_CONTINUE]"
    assert format_token(0x06) == "0x06"
    assert format_token(0xBD) == "0xbd"


def test_format_labelled_example_with_and_without_coordinates():
    ex_full = Example(
        input_ids=(int(SpecialToken.BOS), 0x0A, int(SpecialToken.STEP)),
        token_type_ids=(0, 1, 3),
        position_ids=(0, 0, 0),
    )
    item_full = LabelledExample(
        example=ex_full,
        targets=(0x0A, int(SpecialToken.FLAG_FINISHED)),
    )
    formatted_full = format_labelled_example(item_full, index=0)
    assert "Example 0:" in formatted_full
    assert "input_ids:      [BOS]  0x0a  [STEP]" in formatted_full
    assert "token_type_ids:     0     1       3" in formatted_full
    assert "position_ids:       0     0       0" in formatted_full
    assert "targets:        (0x0a, [FLAG_FINISHED])" in formatted_full

    ex_nope = Example(input_ids=(int(SpecialToken.BOS), 0x0A))
    item_nope = LabelledExample(example=ex_nope, targets=(0x0A,))
    formatted_nope = format_labelled_example(item_nope, index=1)
    assert "token_type_ids" not in formatted_nope
    assert "position_ids" not in formatted_nope


def test_inspect_dataset_writes_jsonl_and_prints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    items = [
        LabelledExample(
            example=Example(
                input_ids=(1, 2), token_type_ids=(0, 1), position_ids=(0, 0)
            ),
            targets=(3, 4),
        ),
        LabelledExample(
            example=Example(input_ids=(5, 6)),
            targets=(7,),
        ),
    ]
    out_file = tmp_path / "dump.jsonl"
    inspect_dataset(items, num_examples=2, out=out_file, print_console=True)

    captured = capsys.readouterr().out
    assert "Wrote 2 examples" in captured
    assert "Example 0:" in captured
    assert "Example 1:" in captured

    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0 == {
        "input_ids": [1, 2],
        "token_type_ids": [0, 1],
        "position_ids": [0, 0],
        "targets": [3, 4],
    }
    rec1 = json.loads(lines[1])
    assert rec1 == {
        "input_ids": [5, 6],
        "token_type_ids": None,
        "position_ids": None,
        "targets": [7],
    }


def test_inspect_dataset_rejects_non_positive_num_examples():
    with pytest.raises(ValueError, match="num_examples must be >= 1"):
        inspect_dataset([], num_examples=0)
