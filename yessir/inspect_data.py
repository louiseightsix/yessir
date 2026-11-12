"""Shared formatting, JSONL dumping, and CLI runner for `yessir` `Dataset`s."""

import argparse
import itertools
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from yessir.dataset import Dataset, LabelledExample
from yessir.special_tokens import SpecialToken
from yessir.vocab import Token


def format_token(token: Token) -> str:
    """Format a byte token as `[SPECIAL_NAME]` or two-digit hex `0x..`."""
    try:
        return f"[{SpecialToken(token).name}]"
    except ValueError:
        return f"0x{token:02x}"


def format_labelled_example(item: LabelledExample, index: int) -> str:
    """Format one `LabelledExample` with aligned columns for console inspection."""
    ex = item.example
    token_strs = [format_token(t) for t in ex.input_ids]
    col_widths = [
        max(
            len(token_strs[i]),
            len(str(ex.token_type_ids[i])) if ex.token_type_ids is not None else 0,
            len(str(ex.position_ids[i])) if ex.position_ids is not None else 0,
        )
        for i in range(len(token_strs))
    ]

    in_row = "  ".join(s.rjust(w) for s, w in zip(token_strs, col_widths, strict=True))
    lines = [
        f"Example {index}:",
        f"  input_ids:      {in_row}",
    ]
    if ex.token_type_ids is not None:
        type_row = "  ".join(
            str(t).rjust(w) for t, w in zip(ex.token_type_ids, col_widths, strict=True)
        )
        lines.append(f"  token_type_ids: {type_row}")
    if ex.position_ids is not None:
        pos_row = "  ".join(
            str(p).rjust(w) for p, w in zip(ex.position_ids, col_widths, strict=True)
        )
        lines.append(f"  position_ids:   {pos_row}")
    lines.append(
        f"  targets:        ({', '.join(format_token(t) for t in item.targets)})"
    )
    return "\n".join(lines)


def to_json_record(item: LabelledExample) -> dict[str, object]:
    """Serialise one `LabelledExample` to a JSON-compatible dict."""
    ex = item.example
    return {
        "input_ids": list(ex.input_ids),
        "token_type_ids": (
            list(ex.token_type_ids) if ex.token_type_ids is not None else None
        ),
        "position_ids": (
            list(ex.position_ids) if ex.position_ids is not None else None
        ),
        "targets": list(item.targets),
    }


def inspect_dataset(
    dataset: Dataset,
    *,
    num_examples: int,
    out: Path | None = None,
    print_console: bool = False,
) -> None:
    """Take `num_examples` from `dataset` and print them and/or write them to `out`."""
    if num_examples < 1:
        raise ValueError(f"num_examples must be >= 1, got {num_examples}")

    examples = list(itertools.islice(dataset, num_examples))

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for item in examples:
                f.write(json.dumps(to_json_record(item)) + "\n")
        print(f"Wrote {len(examples)} examples to {out}")

    if out is None or print_console:
        for idx, item in enumerate(examples):
            print(format_labelled_example(item, idx))
            if idx < len(examples) - 1:
                print()


def run_inspect_cli(
    *,
    description: str,
    add_dataset_args: Callable[[argparse.ArgumentParser], None],
    build_dataset: Callable[[argparse.Namespace], Dataset],
    argv: Sequence[str] | None = None,
) -> None:
    """Parse shared + experiment CLI flags and run `inspect_dataset`."""
    parser = argparse.ArgumentParser(description=description)
    add_dataset_args(parser)
    parser.add_argument(
        "-n",
        "--num-examples",
        type=int,
        default=3,
        help="Number of LabelledExamples to generate (default: 3)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write examples as JSON Lines (.jsonl)",
    )
    parser.add_argument(
        "--print",
        dest="print_console",
        action="store_true",
        help="Also print examples to console when --out is given",
    )
    args = parser.parse_args(argv)
    dataset = build_dataset(args)
    inspect_dataset(
        dataset,
        num_examples=args.num_examples,
        out=args.out,
        print_console=args.print_console,
    )
