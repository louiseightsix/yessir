"""Tests for the full-trace `addressed_ar` variant (`inputs`, `dataset`, `evaluation`, `generation`)."""

import itertools
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from experiments.addition.inputs import (
    AdditionProblem,
    AdditionToken,
    StepOrder,
    encode_address,
    iter_problems,
    pack_step_order,
)
from experiments.addition.reference import reference_add
from experiments.addition.variants.addressed_ar.dataset import (
    AdditionAddressedArDataset,
)
from experiments.addition.variants.addressed_ar.dataset import (
    main as addressed_ar_dataset_main,
)
from experiments.addition.variants.addressed_ar.evaluation import evaluate_addressed_ar
from experiments.addition.variants.addressed_ar.generation import (
    AddressedTraceStep,
    parse_addressed_ar_trace,
    rollout_addressed_ar,
)
from experiments.addition.variants.addressed_ar.inputs import (
    build_addressed_ar_example,
    build_addressed_ar_labelled_example,
    build_addressed_ar_trace_tokens,
)
from yessir.dataset import IGNORE_INDEX
from yessir.example import Example
from yessir.heads import HeadType
from yessir.model import Transformer
from yessir.special_tokens import SpecialToken
from yessir.training import seed_everything
from yessir.vocab import VOCAB_SIZE, Token


class _ScriptedRolloutAddressedArModel:
    """Fake `SubspaceLMHead` model returning scripted next-token argmax at `lengths - 1`."""

    def __init__(self, scripted_tokens: Sequence[Token]) -> None:
        self._tokens = list(scripted_tokens)
        self._cursor = 0
        self.training = True
        self.device = torch.device("cpu")
        self.seen_examples: list[Example] = []

    def eval(self) -> "_ScriptedRolloutAddressedArModel":
        self.training = False
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert lengths is not None
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, VOCAB_SIZE)
        for b in range(batch_size):
            n = int(lengths[b].item())
            self.seen_examples.append(
                Example(
                    input_ids=tuple(input_ids[b, :n].tolist()),
                    token_type_ids=(
                        tuple(token_type_ids[b, :n].tolist())
                        if token_type_ids is not None
                        else None
                    ),
                    position_ids=(
                        tuple(position_ids[b, :n].tolist())
                        if position_ids is not None
                        else None
                    ),
                )
            )
            tok = self._tokens[self._cursor]
            self._cursor += 1
            logits[b, n - 1, tok] = 10.0
        return logits


class _ScriptedEvalAddressedArModel:
    """Stateless prefix-keyed fake `SubspaceLMHead` model for rollout + oracle `evaluate_addressed_ar` tests."""

    def __init__(
        self,
        problems: Sequence[AdditionProblem],
        scripted_targets: Sequence[Sequence[Token]],
        *,
        include_alphabet: bool = True,
        oracle_targets: Sequence[Sequence[Token]] | None = None,
    ) -> None:
        self._entries: list[
            tuple[tuple[int, ...], tuple[int, ...], list[int], list[int]]
        ] = []
        oracle_list = (
            [list(seq) for seq in oracle_targets]
            if oracle_targets is not None
            else [list(seq) for seq in scripted_targets]
        )
        for prob, roll_seq, ora_seq in zip(
            problems, scripted_targets, oracle_list, strict=True
        ):
            prefix = build_addressed_ar_example(
                prob.alphabet, prob.a, prob.b, include_alphabet=include_alphabet
            ).input_ids
            steps = reference_add(prob.alphabet, prob.a, prob.b)
            oracle_full_noq = prefix + build_addressed_ar_trace_tokens(
                steps, prob.alphabet, use_query_tokens=False
            )
            self._entries.append((prefix, oracle_full_noq, list(roll_seq), ora_seq))
        self._has_custom_oracle = oracle_targets is not None
        self.training = True
        self.device = torch.device("cpu")

    def eval(self) -> "_ScriptedEvalAddressedArModel":
        self.training = False
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert lengths is not None
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, VOCAB_SIZE)

        for b in range(batch_size):
            n = int(lengths[b].item())
            row = tuple(input_ids[b, :n].tolist())
            logits[b, :, int(SpecialToken.PAD)] = 50.0
            matched = next(
                entry for entry in self._entries if row[: len(entry[0])] == entry[0]
            )
            prefix, oracle_full_noq, roll_seq, ora_seq = matched
            seq = (
                ora_seq
                if (self._has_custom_oracle and row == oracle_full_noq)
                else roll_seq
            )
            start = len(prefix) - 1
            for idx, tok in enumerate(seq):
                if start + idx < n:
                    logits[b, start + idx, int(SpecialToken.PAD)] = 0.0
                    logits[b, start + idx, tok] = 100.0
        return logits


def test_addressed_ar_spec_example_1_all_modes_and_prefix_consistency():
    """Verify `spec/variants/addressed_ar.md` §1–§3 Example 1 (`987 + 25 = 1012`) across all 4 modes."""
    alphabet = b"0123456789"
    a = b"789"
    b = b"52"
    bos = int(SpecialToken.BOS)
    sep = int(SpecialToken.SEP)
    eos = int(SpecialToken.EOS)
    idx = int(AdditionToken.IDX)
    q_s = int(AdditionToken.QUERY_SUM)
    q_c = int(AdditionToken.QUERY_CARRY)

    prefix_ids = (
        bos,
        *alphabet,
        sep,
        idx,
        ord("0"),
        ord("7"),
        idx,
        ord("1"),
        ord("8"),
        idx,
        ord("2"),
        ord("9"),
        sep,
        idx,
        ord("0"),
        ord("5"),
        idx,
        ord("1"),
        ord("2"),
        sep,
    )
    p_len = len(prefix_ids)

    # 1. sum_first, use_query_tokens=False:
    # [IDX] 0 2 1  [IDX] 1 1 1  [IDX] 2 0 1  [IDX] 3 1 0  [EOS]
    sf_noq = build_addressed_ar_labelled_example(
        alphabet, a, b, step_order=StepOrder.SUM_FIRST, use_query_tokens=False
    )
    expected_sf_trace = (
        idx,
        ord("0"),
        ord("2"),
        ord("1"),
        idx,
        ord("1"),
        ord("1"),
        ord("1"),
        idx,
        ord("2"),
        ord("0"),
        ord("1"),
        idx,
        ord("3"),
        ord("1"),
        ord("0"),
    )
    assert sf_noq.example.input_ids == (*prefix_ids, *expected_sf_trace)
    assert sf_noq.example.token_type_ids is None
    assert sf_noq.example.position_ids is None
    assert sf_noq.labels == (IGNORE_INDEX,) * (p_len - 1) + (*expected_sf_trace, eos)

    # 2. carry_first, use_query_tokens=False:
    # [IDX] 0 1 2  [IDX] 1 1 1  [IDX] 2 1 0  [IDX] 3 0 1  [EOS]
    cf_noq = build_addressed_ar_labelled_example(
        alphabet, a, b, step_order=StepOrder.CARRY_FIRST, use_query_tokens=False
    )
    expected_cf_trace = (
        idx,
        ord("0"),
        ord("1"),
        ord("2"),
        idx,
        ord("1"),
        ord("1"),
        ord("1"),
        idx,
        ord("2"),
        ord("1"),
        ord("0"),
        idx,
        ord("3"),
        ord("0"),
        ord("1"),
    )
    assert cf_noq.example.input_ids == (*prefix_ids, *expected_cf_trace)
    assert cf_noq.example.token_type_ids is None
    assert cf_noq.example.position_ids is None
    assert cf_noq.labels == (IGNORE_INDEX,) * (p_len - 1) + (*expected_cf_trace, eos)

    # 3. sum_first, use_query_tokens=True
    sf_q = build_addressed_ar_labelled_example(
        alphabet, a, b, step_order=StepOrder.SUM_FIRST, use_query_tokens=True
    )
    expected_sf_q_trace = (
        idx,
        ord("0"),
        q_s,
        ord("2"),
        q_c,
        ord("1"),
        idx,
        ord("1"),
        q_s,
        ord("1"),
        q_c,
        ord("1"),
        idx,
        ord("2"),
        q_s,
        ord("0"),
        q_c,
        ord("1"),
        idx,
        ord("3"),
        q_s,
        ord("1"),
        q_c,
        ord("0"),
    )
    assert sf_q.example.input_ids == (*prefix_ids, *expected_sf_q_trace)
    assert sf_q.labels == (IGNORE_INDEX,) * (p_len - 1) + (*expected_sf_q_trace, eos)

    # 4. carry_first, use_query_tokens=True
    cf_q = build_addressed_ar_labelled_example(
        alphabet, a, b, step_order=StepOrder.CARRY_FIRST, use_query_tokens=True
    )
    expected_cf_q_trace = (
        idx,
        ord("0"),
        q_c,
        ord("1"),
        q_s,
        ord("2"),
        idx,
        ord("1"),
        q_c,
        ord("1"),
        q_s,
        ord("1"),
        idx,
        ord("2"),
        q_c,
        ord("1"),
        q_s,
        ord("0"),
        idx,
        ord("3"),
        q_c,
        ord("0"),
        q_s,
        ord("1"),
    )
    assert cf_q.example.input_ids == (*prefix_ids, *expected_cf_q_trace)
    assert cf_q.labels == (IGNORE_INDEX,) * (p_len - 1) + (*expected_cf_q_trace, eos)

    # Verify rollout's incremental prefixes match slices of the full teacher-forced Example!
    steps = reference_add(alphabet, a, b)
    for order in StepOrder:
        for use_q in (False, True):
            full = build_addressed_ar_labelled_example(
                alphabet, a, b, steps=steps, step_order=order, use_query_tokens=use_q
            )
            trace = build_addressed_ar_trace_tokens(
                steps, alphabet, step_order=order, use_query_tokens=use_q
            )
            for t_len in range(len(trace) + 1):
                partial = build_addressed_ar_example(alphabet, a, b, trace[:t_len])
                end = p_len + t_len
                assert partial.input_ids == full.example.input_ids[:end]
                assert partial.token_type_ids is None
                assert partial.position_ids is None


def test_addressed_ar_spec_example_2_zero_preamble_and_multidigit_addresses():
    """Verify `spec/variants/addressed_ar.md` §2 Example 2, `include_alphabet=False`, and multi-digit `addr(k)`."""
    alphabet = b"q#7"
    a = b"7#"
    b = b"#"
    idx = int(AdditionToken.IDX)
    eos = int(SpecialToken.EOS)

    sf = build_addressed_ar_labelled_example(
        alphabet, a, b, step_order=StepOrder.SUM_FIRST
    )
    assert sf.labels[-9:] == (
        idx,
        ord("q"),
        ord("q"),
        ord("#"),
        idx,
        ord("#"),
        ord("7"),
        ord("q"),
        eos,
    )

    cf = build_addressed_ar_labelled_example(
        alphabet, a, b, step_order=StepOrder.CARRY_FIRST
    )
    assert cf.labels[-9:] == (
        idx,
        ord("q"),
        ord("#"),
        ord("q"),
        idx,
        ord("#"),
        ord("q"),
        ord("7"),
        eos,
    )

    # Zero-preamble requires L_a, L_b >= B.
    with pytest.raises(ValueError, match="Zero-preamble mode"):
        build_addressed_ar_example(alphabet, a, b, include_alphabet=False)

    # Multi-digit base-2 column addresses (`k = 2 -> [a, b]`, `k = 3 -> [b, b]`) with zero-preamble:
    bin_alpha = b"ab"
    bin_a = b"bba"  # 3
    bin_b = b"bab"  # 5 -> sum = 8 (4 steps: k = 0..3)
    bin_steps = reference_add(bin_alpha, bin_a, bin_b)
    assert len(bin_steps) == 4
    full_bin = build_addressed_ar_labelled_example(
        bin_alpha,
        bin_a,
        bin_b,
        steps=bin_steps,
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
        include_alphabet=False,
    )
    assert full_bin.example.token_type_ids is None
    assert full_bin.example.position_ids is None

    bin_trace = build_addressed_ar_trace_tokens(
        bin_steps,
        bin_alpha,
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
    )
    bin_prefix_len = len(full_bin.example) - len(bin_trace)
    for t_len in range(len(bin_trace) + 1):
        partial = build_addressed_ar_example(
            bin_alpha,
            bin_a,
            bin_b,
            bin_trace[:t_len],
            include_alphabet=False,
        )
        end = bin_prefix_len + t_len
        assert partial.input_ids == full_bin.example.input_ids[:end]
        assert partial.token_type_ids is None
        assert partial.position_ids is None

    ex_nope = build_addressed_ar_example(alphabet, a, b, (idx, ord("q")))
    assert ex_nope.token_type_ids is None
    assert ex_nope.position_ids is None


def test_addition_addressed_ar_dataset_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(ValueError, match="include_alphabet=False"):
        AdditionAddressedArDataset(bases=(4,), max_length=3, include_alphabet=False)

    ds = AdditionAddressedArDataset(
        bases=(3,),
        min_length=1,
        max_length=4,
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
        include_alphabet=False,
        seed=11,
    )
    first_prob = next(
        iter_problems(
            bases=(3,), min_length=1, max_length=4, include_alphabet=False, seed=11
        )
    )
    expected = build_addressed_ar_labelled_example(
        first_prob.alphabet,
        first_prob.a,
        first_prob.b,
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
        include_alphabet=False,
    )
    assert next(iter(ds)) == expected

    dump_path = tmp_path / "addressed_ar_dump.jsonl"
    addressed_ar_dataset_main(
        [
            "--bases",
            "2",
            "3",
            "--min-length",
            "1",
            "--max-length",
            "4",
            "--step-order",
            "carry_first",
            "--use-query-tokens",
            "--no-include-alphabet",
            "--num-examples",
            "3",
            "--out",
            str(dump_path),
            "--print",
        ]
    )
    assert len(dump_path.read_text().splitlines()) == 3
    out = capsys.readouterr().out
    assert "Example 0:" in out


def test_rollout_and_parse_addressed_ar_trace():
    alphabet = b"ab"  # base 2: a=0, b=1
    a = b"bb"  # 3
    b = b"b"  # 1 -> sum = 4 = [a, a, b] (3 steps: k = 0, 1, 2; addr(2) = ('a', 'b'))
    steps = reference_add(alphabet, a, b)
    eos = int(SpecialToken.EOS)
    idx = int(AdditionToken.IDX)

    gold_trace = build_addressed_ar_trace_tokens(
        steps, alphabet, step_order=StepOrder.CARRY_FIRST, use_query_tokens=True
    )
    scripted = [*gold_trace, eos, ord("a")]
    model = _ScriptedRolloutAddressedArModel(scripted)
    emitted = rollout_addressed_ar(
        model,  # ty: ignore[invalid-argument-type]
        alphabet,
        a,
        b,
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
    )
    assert model.training is False
    assert emitted == [*gold_trace, eos]

    parsed, well_formed = parse_addressed_ar_trace(
        emitted, step_order=StepOrder.CARRY_FIRST, use_query_tokens=True
    )
    assert well_formed is True
    assert parsed == [
        AddressedTraceStep(
            address=encode_address(k, alphabet),
            sum_digit=st.sum_digit,
            carry_out=st.carry_out,
        )
        for k, st in enumerate(steps)
    ]

    # Premature EOS stops immediately and parses as malformed.
    early = _ScriptedRolloutAddressedArModel([eos, ord("a")])
    assert rollout_addressed_ar(early, alphabet, a, b) == [eos]  # ty: ignore[invalid-argument-type]
    assert parse_addressed_ar_trace([eos]) == ([], False)

    # Missing leading IDX or wrong query token or truncated segment parses as malformed.
    assert parse_addressed_ar_trace([ord("a"), ord("b"), eos]) == ([], False)
    assert parse_addressed_ar_trace([idx, ord("a"), eos]) == ([], False)
    assert parse_addressed_ar_trace(
        [
            idx,
            ord("a"),
            int(AdditionToken.QUERY_SUM),
            ord("a"),
            int(AdditionToken.QUERY_SUM),  # expected QUERY_CARRY
            ord("b"),
            eos,
        ],
        step_order=StepOrder.SUM_FIRST,
        use_query_tokens=True,
    ) == ([], False)

    # Control / special tokens inside address or value slots parse as malformed.
    assert parse_addressed_ar_trace(
        [
            idx,
            int(SpecialToken.SEP),
            int(SpecialToken.PAD),
            int(SpecialToken.BOS),
            eos,
        ]
    ) == ([], False)
    assert parse_addressed_ar_trace(
        [idx, ord("a"), ord("a"), int(AdditionToken.QUERY_FLAG), eos]
    ) == ([], False)
    assert parse_addressed_ar_trace(
        [
            idx,
            int(AdditionToken.QUERY_SUM),
            int(AdditionToken.QUERY_SUM),
            ord("a"),
            int(AdditionToken.QUERY_CARRY),
            ord("b"),
            eos,
        ],
        use_query_tokens=True,
    ) == ([], False)

    # Length cap when EOS is never emitted: for `max_columns = 3` in base 2 (`addr` lengths 1, 1, 2)
    # and `use_query_tokens=False` (`body_len = 2`), `max_tokens = (4 + 4 + 5) + 1 == 14`.
    never_noq = _ScriptedRolloutAddressedArModel([ord("a")] * 20)
    capped = rollout_addressed_ar(
        never_noq,  # ty: ignore[invalid-argument-type]
        alphabet,
        a,
        b,
        use_query_tokens=False,
    )
    assert len(capped) == 14
    assert parse_addressed_ar_trace(capped)[1] is False


@pytest.mark.parametrize("step_order", list(StepOrder))
@pytest.mark.parametrize("use_query_tokens", [False, True])
@pytest.mark.parametrize("include_alphabet", [True, False])
def test_evaluate_addressed_ar_all_modes_and_batch_size_invariance(
    step_order: StepOrder, use_query_tokens: bool, include_alphabet: bool
):
    problems = list(
        itertools.islice(
            iter_problems(
                bases=(2, 4),
                min_length=1,
                max_length=5,
                p_prior=1.0,
                include_alphabet=include_alphabet,
                seed=5,
            ),
            2,
        )
    )
    s0 = reference_add(problems[0].alphabet, problems[0].a, problems[0].b)
    s1 = reference_add(problems[1].alphabet, problems[1].a, problems[1].b)
    eos = int(SpecialToken.EOS)

    t0 = [
        *build_addressed_ar_trace_tokens(
            s0,
            problems[0].alphabet,
            step_order=step_order,
            use_query_tokens=use_query_tokens,
        ),
        eos,
    ]
    t1 = [
        *build_addressed_ar_trace_tokens(
            s1,
            problems[1].alphabet,
            step_order=step_order,
            use_query_tokens=use_query_tokens,
        ),
        eos,
    ]
    wrong_digit = (
        problems[1].alphabet[0]
        if s1[0].sum_digit != problems[1].alphabet[0]
        else problems[1].alphabet[1]
    )
    body_sum_offset = (
        (1 if step_order == StepOrder.SUM_FIRST else 3)
        if use_query_tokens
        else (0 if step_order == StepOrder.SUM_FIRST else 1)
    )
    t1[2 + body_sum_offset] = wrong_digit

    model = _ScriptedEvalAddressedArModel(
        problems, [t0, t1], include_alphabet=include_alphabet
    )
    m_b1 = evaluate_addressed_ar(
        model,  # ty: ignore[invalid-argument-type]
        problems,
        num_problems=2,
        batch_size=1,
        step_order=step_order,
        use_query_tokens=use_query_tokens,
        include_alphabet=include_alphabet,
    )
    m_b5 = evaluate_addressed_ar(
        model,  # ty: ignore[invalid-argument-type]
        problems,
        num_problems=2,
        batch_size=5,
        step_order=step_order,
        use_query_tokens=use_query_tokens,
        include_alphabet=include_alphabet,
    )
    assert m_b1 == m_b5
    assert m_b1.exact_match == pytest.approx(0.5)
    assert m_b1.halt_accuracy == pytest.approx(1.0)
    assert m_b1.address_accuracy == pytest.approx(1.0)
    total_steps = len(s0) + len(s1)
    assert m_b1.step_accuracy == pytest.approx((total_steps - 1) / total_steps)
    assert m_b1.oracle_step_accuracy == pytest.approx((total_steps - 1) / total_steps)


def test_evaluate_addressed_ar_address_idx_eos_and_step_order_taxonomy():
    prob = AdditionProblem(alphabet=tuple(b"q#7"), a=tuple(b"7"), b=tuple(b"#"))
    steps = reference_add(prob.alphabet, prob.a, prob.b)
    assert len(steps) == 2
    idx = int(AdditionToken.IDX)
    eos = int(SpecialToken.EOS)

    gold_noq = [
        *build_addressed_ar_trace_tokens(
            steps, prob.alphabet, step_order=StepOrder.SUM_FIRST, use_query_tokens=False
        ),
        eos,
    ]

    bad_addr = list(gold_noq)
    bad_addr[1] = ord("#")
    m_addr = evaluate_addressed_ar(
        _ScriptedEvalAddressedArModel([prob], [bad_addr]),  # ty: ignore[invalid-argument-type]
        [prob],
        num_problems=1,
    )
    assert m_addr.exact_match == 0.0
    assert m_addr.address_accuracy == pytest.approx(0.5)
    assert m_addr.step_accuracy == pytest.approx(0.5)
    assert m_addr.oracle_step_accuracy == pytest.approx(0.5)
    assert m_addr.halt_accuracy == 1.0
    assert m_addr.syntax_malformation == 1.0

    bad_init_idx = list(gold_noq)
    bad_init_idx[0] = ord("q")
    m_init = evaluate_addressed_ar(
        _ScriptedEvalAddressedArModel([prob], [bad_init_idx]),  # ty: ignore[invalid-argument-type]
        [prob],
        num_problems=1,
    )
    assert m_init.exact_match == 0.0
    assert m_init.address_accuracy == pytest.approx(0.5)
    assert m_init.step_accuracy == pytest.approx(0.5)
    assert m_init.halt_accuracy == 1.0
    assert m_init.syntax_malformation == 1.0

    bad_cont_idx = list(gold_noq)
    bad_cont_idx[4] = ord("q")
    m_cont = evaluate_addressed_ar(
        _ScriptedEvalAddressedArModel([prob], [bad_cont_idx]),  # ty: ignore[invalid-argument-type]
        [prob],
        num_problems=1,
    )
    assert m_cont.exact_match == 0.0
    assert m_cont.address_accuracy == 0.0
    assert m_cont.step_accuracy == 0.0
    assert m_cont.halt_accuracy == 0.0
    assert m_cont.syntax_malformation == 1.0

    premature = list(gold_noq)
    premature[2] = eos
    m_prem = evaluate_addressed_ar(
        _ScriptedEvalAddressedArModel([prob], [premature]),  # ty: ignore[invalid-argument-type]
        [prob],
        num_problems=1,
    )
    assert m_prem.address_accuracy == pytest.approx(0.5)
    assert m_prem.step_accuracy == 0.0
    assert m_prem.halt_accuracy == 0.0
    assert m_prem.out_of_alphabet == 1.0

    missing_eos = list(gold_noq)
    missing_eos[-1] = idx
    m_miss = evaluate_addressed_ar(
        _ScriptedEvalAddressedArModel([prob], [missing_eos]),  # ty: ignore[invalid-argument-type]
        [prob],
        num_problems=1,
    )
    assert m_miss.exact_match == 0.0
    assert m_miss.address_accuracy == 1.0
    assert m_miss.step_accuracy == 1.0
    assert m_miss.halt_accuracy == 0.0
    assert m_miss.syntax_malformation == 1.0

    wrong_in_alpha = ord("7")
    out_of_alpha = 255
    for order, expected_proc, expected_ooa in (
        (StepOrder.SUM_FIRST, 1.0, 0.0),
        (StepOrder.CARRY_FIRST, 0.0, 1.0),
    ):
        seq = [
            idx,
            ord("q"),
            *pack_step_order(wrong_in_alpha, out_of_alpha, order),
            idx,
            ord("#"),
            *pack_step_order(steps[1].sum_digit, steps[1].carry_out, order),
            eos,
        ]
        m_ord = evaluate_addressed_ar(
            _ScriptedEvalAddressedArModel([prob], [seq]),  # ty: ignore[invalid-argument-type]
            [prob],
            num_problems=1,
            step_order=order,
        )
        assert m_ord.procedural_error == expected_proc
        assert m_ord.out_of_alphabet == expected_ooa

    bin_prob = AdditionProblem(alphabet=tuple(b"01"), a=tuple(b"101"), b=tuple(b"010"))
    bin_steps = reference_add(bin_prob.alphabet, bin_prob.a, bin_prob.b)
    wrong_width_rollout = [
        idx,
        ord("0"),
        bin_steps[0].sum_digit,
        bin_steps[0].carry_out,
        idx,
        ord("1"),
        bin_steps[1].sum_digit,
        bin_steps[1].carry_out,
        idx,
        ord("0"),
        bin_steps[2].sum_digit,
        bin_steps[2].carry_out,
        eos,
    ]
    m_width = evaluate_addressed_ar(
        _ScriptedEvalAddressedArModel([bin_prob], [wrong_width_rollout]),  # ty: ignore[invalid-argument-type]
        [bin_prob],
        num_problems=1,
    )
    assert m_width.exact_match == 0.0
    assert m_width.address_accuracy == pytest.approx(2 / 3)
    assert m_width.step_accuracy == pytest.approx(2 / 3)
    assert m_width.syntax_malformation == 1.0


def test_evaluate_and_rollout_addressed_ar_with_real_transformer_and_validation():
    seed_everything(0)
    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        use_token_type=False,
        d_position=0,
        d_scratch=24,
    )
    problems = list(
        itertools.islice(
            iter_problems(
                bases=(2, 3),
                min_length=1,
                max_length=4,
                include_alphabet=False,
                seed=77,
            ),
            4,
        )
    )
    m_b1 = evaluate_addressed_ar(
        model,
        problems,
        num_problems=4,
        batch_size=1,
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
        include_alphabet=False,
    )
    m_b3 = evaluate_addressed_ar(
        model,
        problems,
        num_problems=4,
        batch_size=3,
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
        include_alphabet=False,
    )
    assert m_b1 == m_b3
    assert 0.0 <= m_b3.exact_match <= 1.0
    assert 0.0 <= m_b3.step_accuracy <= 1.0
    assert 0.0 <= m_b3.oracle_step_accuracy <= 1.0
    assert m_b3.address_accuracy is not None
    assert 0.0 <= m_b3.address_accuracy <= 1.0
    assert 0.0 <= m_b3.halt_accuracy <= 1.0

    # Verify KV-cached `evaluate_addressed_ar` rollout matches uncached `rollout_addressed_ar`.
    from experiments.addition.variants.addressed_ar.evaluation import (
        _rollout_addressed_ar_chunk,
    )

    max_tokens_all = [
        sum(
            1 + len(encode_address(k, p.alphabet)) + 4
            for k in range(max(len(p.a), len(p.b)) + 1)
        )
        + 1
        for p in problems
    ]
    cached_rollouts = _rollout_addressed_ar_chunk(
        model,
        problems,
        max_tokens_all,
        include_alphabet=False,
        device=model.device,
    )
    for prob, cached in zip(problems, cached_rollouts, strict=True):
        uncached = rollout_addressed_ar(
            model,
            prob.alphabet,
            prob.a,
            prob.b,
            step_order=StepOrder.CARRY_FIRST,
            use_query_tokens=True,
            include_alphabet=False,
        )
        assert cached == uncached

    with pytest.raises(ValueError, match="num_problems must be >= 1"):
        evaluate_addressed_ar(model, problems, num_problems=0)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        evaluate_addressed_ar(model, problems, num_problems=1, batch_size=0)
    with pytest.raises(ValueError, match="exhausted early"):
        evaluate_addressed_ar(model, [], num_problems=1)

    cls_model = Transformer(
        head_type=HeadType.PARALLEL_CLASSIFICATION,
        num_classes=(VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE),
        n_layers=1,
        n_heads=4,
        d_ff=32,
        use_token_type=False,
        d_position=0,
        d_scratch=24,
    )
    with pytest.raises(ValueError, match="SubspaceLMHead"):
        evaluate_addressed_ar(
            cls_model,
            iter_problems(bases=(2,), min_length=1, max_length=1, seed=1),
            num_problems=1,
        )
    with pytest.raises(ValueError, match="SubspaceLMHead"):
        rollout_addressed_ar(cls_model, b"01", b"0", b"1")


def test_evaluate_addressed_ar_token_budget_subchunking_is_invariant(
    monkeypatch: pytest.MonkeyPatch,
):
    import experiments.addition.variants.addressed_ar.evaluation as eval_mod

    seed_everything(0)
    model = Transformer(
        head_type=HeadType.SUBSPACE_LM,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        use_token_type=False,
        d_position=0,
        d_scratch=0,
    )
    problems = list(
        itertools.islice(
            iter_problems(bases=(2,), min_length=6, max_length=8, seed=3), 5
        )
    )
    baseline = evaluate_addressed_ar(model, problems, num_problems=5, batch_size=1)
    monkeypatch.setattr(eval_mod, "MAX_EVAL_CHUNK_TOKENS", 100)
    subchunked = evaluate_addressed_ar(model, problems, num_problems=5, batch_size=256)
    assert subchunked == baseline
