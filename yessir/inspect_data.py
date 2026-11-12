"""Command-line tool to inspect or dump `LabelledExample`s from any `yessir` `Dataset`."""

import argparse
import itertools
import json
from pathlib import Path

from experiments.sir_yes_sir.dataset import SirYesSirDataset
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
    """Format one `LabelledExample` for human inspection on the console."""
    ex = item.example
    lines = [
        f"Example {index}:",
        f"  input_ids:      {' '.join(format_token(t) for t in ex.input_ids)}",
    ]
    if ex.token_type_ids is not None:
        lines.append(f"  token_type_ids: {' '.join(str(t) for t in ex.token_type_ids)}")
    if ex.position_ids is not None:
        lines.append(f"  position_ids:   {' '.join(str(p) for p in ex.position_ids)}")
    lines.append(
        f"  targets:        ({', '.join(format_token(t) for t in item.targets)})"
    )
    return "\n".join(lines)


def _to_json_record(item: LabelledExample) -> dict[str, object]:
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


def _build_dataset(args: argparse.Namespace) -> Dataset:
    if args.experiment == "sir_yes_sir":
        return SirYesSirDataset(
            alphabet_size=args.alphabet_size,
            seq_length=args.seq_length,
            seed=args.seed,
        )
    raise ValueError(f"unknown experiment: {args.experiment!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate, print, or dump training examples from a yessir Dataset."
    )
    parser.add_argument(
        "experiment",
        choices=("sir_yes_sir",),
        nargs="?",
        default="sir_yes_sir",
        help="Experiment dataset to run (default: sir_yes_sir)",
    )
    parser.add_argument(
        "-n",
        "--num-examples",
        type=int,
        default=3,
        help="Number of LabelledExamples to generate (default: 3)",
    )
    parser.add_argument(
        "-B",
        "--alphabet-size",
        type=int,
        default=4,
        help="Alphabet size B (default: 4)",
    )
    parser.add_argument(
        "-L",
        "--seq-length",
        type=int,
        default=3,
        help="Input sequence length L (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
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
    args = parser.parse_args()

    dataset = _build_dataset(args)
    examples = list(itertools.islice(dataset, args.num_examples))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for item in examples:
                f.write(json.dumps(_to_json_record(item)) + "\n")
        print(f"Wrote {len(examples)} examples to {args.out}")

    if args.out is None or args.print_console:
        for idx, item in enumerate(examples):
            print(format_labelled_example(item, idx))
            if idx < len(examples) - 1:
                print()


if __name__ == "__main__":
    main()
