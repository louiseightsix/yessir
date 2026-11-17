"""Tests for `experiments.addition.config` (`AdditionConfig`, `AdditionClsConfig`, `parse_bases_csv`)."""

import json
import subprocess
import sys

import pytest

from experiments.addition.config import (
    AdditionAddressedArConfig,
    AdditionAddressedStepArConfig,
    AdditionArConfig,
    AdditionClsConfig,
    AdditionConfig,
    AdditionStepArConfig,
    AdditionStepConfig,
    AdditionStepForkConfig,
    EvalSplit,
    SymbolSetName,
    parse_bases_csv,
)
from experiments.addition.inputs import EmbeddingMode, StepOrder
from yessir.alphabet import SymbolSet


def test_parse_bases_csv():
    assert parse_bases_csv("2, 4, 8, 10", field_name="bases") == (2, 4, 8, 10)
    assert parse_bases_csv("3", field_name="bases") == (3,)

    with pytest.raises(TypeError, match="bases must be a str"):
        parse_bases_csv([2, 10], field_name="bases")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="at least 1 base"):
        parse_bases_csv(" , ", field_name="bases")
    with pytest.raises(ValueError, match="invalid integer"):
        parse_bases_csv("2,ten", field_name="bases")
    with pytest.raises(ValueError, match="must be >= 2"):
        parse_bases_csv("1,2", field_name="bases")
    with pytest.raises(ValueError, match="must not contain duplicates"):
        parse_bases_csv("2,4,2", field_name="bases")


def test_default_configs_and_derived_properties():
    cfg = AdditionConfig()
    assert cfg.bases == "2,4,8,10"
    assert cfg.parsed_bases == (2, 4, 8, 10)
    assert cfg.eval_ood_bases == "3,12"
    assert cfg.parsed_eval_ood_bases == (3, 12)
    assert cfg.min_length == 1
    assert cfg.max_length == 20
    assert cfg.symbol_set == SymbolSetName.ALL
    assert cfg.symbol_pool_size is None
    assert cfg.symbol_pool == bytes(SymbolSet.ALL)
    assert not hasattr(cfg, "embedding_mode")
    assert not hasattr(cfg, "d_position")
    assert not hasattr(cfg, "fixed_d_model")
    assert cfg.use_token_type is False
    assert cfg.effective_d_position == 0
    assert cfg.effective_d_scratch == 256
    assert cfg.d_model == 256 + 256 == 512
    assert cfg.log_every == 100
    assert cfg.eval_every is None
    assert cfg.eval_permuted_canonical is True
    assert cfg.effective_eval_every == 5_000
    assert (
        cfg.derived_summary()
        == "d_model=512, symbol_pool=240, eval_every=5000, splits=[id,ood_base3,ood_base12,extrap_len,permuted]"
    )
    assert AdditionConfig(max_steps=500_000).effective_eval_every == 25_000
    assert AdditionConfig(max_steps=30_000).effective_eval_every == 2_500
    assert AdditionConfig(eval_every=250).effective_eval_every == 250
    assert AdditionConfig(eval_every=0).effective_eval_every is None
    assert (
        AdditionConfig(eval_every=0).derived_summary()
        == "d_model=512, symbol_pool=240, eval_every=final, splits=[id,ood_base3,ood_base12,extrap_len,permuted]"
    )

    cls_cfg = AdditionClsConfig()
    assert isinstance(cls_cfg, AdditionStepConfig)
    assert cls_cfg.embedding_mode == EmbeddingMode.TOKEN_TYPE_PE
    assert cls_cfg.use_token_type is True
    assert cls_cfg.fixed_d_model is True
    assert cls_cfg.effective_d_position == 32
    assert cls_cfg.effective_d_scratch == 208
    assert cls_cfg.d_model == 512
    assert cls_cfg.p_terminal == 0.2
    assert cls_cfg.all_steps is False

    fork_cfg = AdditionStepForkConfig()
    assert isinstance(fork_cfg, AdditionStepConfig)
    assert fork_cfg.embedding_mode == EmbeddingMode.TOKEN_TYPE_PE
    assert fork_cfg.p_terminal == 0.2
    assert fork_cfg.all_steps is False

    step_ar_cfg = AdditionStepArConfig()
    assert isinstance(step_ar_cfg, AdditionStepConfig)
    assert step_ar_cfg.embedding_mode == EmbeddingMode.TOKEN_TYPE_PE
    assert step_ar_cfg.step_order == StepOrder.SUM_FIRST
    assert step_ar_cfg.use_query_tokens is False

    ar_cfg = AdditionArConfig()
    assert not isinstance(ar_cfg, AdditionStepConfig)
    assert not hasattr(ar_cfg, "embedding_mode")
    assert ar_cfg.d_scratch == 256
    assert ar_cfg.d_model == 512
    assert ar_cfg.step_order == StepOrder.SUM_FIRST
    assert ar_cfg.use_query_tokens is False

    addr_step_ar_cfg = AdditionAddressedStepArConfig()
    assert isinstance(addr_step_ar_cfg, AdditionStepArConfig)
    assert addr_step_ar_cfg.embedding_mode == EmbeddingMode.NOPE
    assert addr_step_ar_cfg.use_token_type is False
    assert addr_step_ar_cfg.effective_d_position == 0
    assert addr_step_ar_cfg.effective_d_scratch == 256
    assert addr_step_ar_cfg.d_model == 512
    assert addr_step_ar_cfg.step_order == StepOrder.SUM_FIRST
    assert addr_step_ar_cfg.use_query_tokens is False
    assert addr_step_ar_cfg.include_alphabet is True

    addr_ar_cfg = AdditionAddressedArConfig()
    assert isinstance(addr_ar_cfg, AdditionArConfig)
    assert not isinstance(addr_ar_cfg, AdditionStepConfig)
    assert not hasattr(addr_ar_cfg, "embedding_mode")
    assert addr_ar_cfg.d_scratch == 256
    assert addr_ar_cfg.d_model == 512
    assert addr_ar_cfg.step_order == StepOrder.SUM_FIRST
    assert addr_ar_cfg.use_query_tokens is False
    assert addr_ar_cfg.include_alphabet is True


def test_embedding_mode_and_fixed_d_model_combinations():
    cfg_tt_nope = AdditionStepConfig(embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE)
    assert cfg_tt_nope.use_token_type is True
    assert cfg_tt_nope.effective_d_position == 0
    assert cfg_tt_nope.effective_d_scratch == 208 + 32 == 240
    assert cfg_tt_nope.d_model == 512

    cfg_nope = AdditionStepConfig(embedding_mode=EmbeddingMode.NOPE)
    assert cfg_nope.use_token_type is False
    assert cfg_nope.effective_d_position == 0
    assert cfg_nope.effective_d_scratch == 208 + 16 + 32 == 256
    assert cfg_nope.d_model == 512

    cfg_tt_nope_shrink = AdditionStepConfig(
        embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE,
        fixed_d_model=False,
    )
    assert cfg_tt_nope_shrink.effective_d_position == 0
    assert cfg_tt_nope_shrink.effective_d_scratch == 208
    assert cfg_tt_nope_shrink.d_model == 480

    cfg_nope_shrink = AdditionStepConfig(
        embedding_mode=EmbeddingMode.NOPE,
        fixed_d_model=False,
    )
    assert cfg_nope_shrink.effective_d_position == 0
    assert cfg_nope_shrink.effective_d_scratch == 208
    assert cfg_nope_shrink.d_model == 464


def test_eval_splits_construction_and_seed_sweep_invariance():
    # OOD bases already present in `bases` are automatically skipped; blank string disables OOD bases.
    cfg = AdditionConfig(
        bases="2, 3, 10",
        eval_ood_bases="3, 12",
        max_length=10,
        eval_extrap_max_length=25,
        eval_permuted_canonical=True,
        p_prior=0.5,
        p_chain=0.0,
        eval_p_chain=0.3,
        seed=1999,
        eval_seed=999,
    )
    cfg_blank_ood = AdditionConfig(
        bases="2,10",
        eval_ood_bases="",
        eval_extrap_max_length=30,
        eval_permuted_canonical=False,
        seed=2999,
        eval_seed=999,
    )
    assert cfg_blank_ood.eval_ood_bases is None
    assert cfg_blank_ood.parsed_eval_ood_bases == ()
    assert [s.name for s in cfg_blank_ood.eval_splits()] == ["id", "extrap_len"]

    # The stored value keeps only the OOD bases that get a split; `permuted` is enabled by default.
    assert cfg.eval_ood_bases == "12"
    assert cfg.parsed_eval_ood_bases == (12,)
    all_trained = AdditionConfig(bases="2,3,12", eval_ood_bases="3,12")
    assert all_trained.eval_ood_bases is None
    assert [s.name for s in all_trained.eval_splits()] == [
        "id",
        "extrap_len",
        "permuted",
    ]

    splits = cfg.eval_splits()
    assert splits == (
        EvalSplit(
            name="id",
            prefix="",
            bases=(2, 3, 10),
            min_length=1,
            max_length=10,
            p_prior=0.5,
            permuted_canonical=False,
            p_chain=0.3,
            seed="999:eval:id",
        ),
        EvalSplit(
            name="ood_base12",
            prefix="ood_base12_",
            bases=(12,),
            min_length=1,
            max_length=10,
            p_prior=0.0,
            permuted_canonical=False,
            p_chain=0.3,
            seed="999:eval:ood_base12",
        ),
        EvalSplit(
            name="extrap_len",
            prefix="extrap_len_",
            bases=(2, 3, 10),
            min_length=11,
            max_length=25,
            p_prior=0.5,
            permuted_canonical=False,
            p_chain=0.3,
            seed="999:eval:extrap_len",
        ),
        EvalSplit(
            name="permuted",
            prefix="permuted_",
            bases=(3, 10),
            min_length=1,
            max_length=10,
            p_prior=0.0,
            permuted_canonical=True,
            p_chain=0.3,
            seed="999:eval:permuted",
        ),
    )

    # Pure-binary `bases="2"` falls back to `(2,)` on `permuted` instead of an empty tuple.
    b2_only = AdditionConfig(
        bases="2", eval_ood_bases="", eval_permuted_canonical=True
    ).eval_splits()
    assert b2_only[-1].name == "permuted"
    assert b2_only[-1].bases == (2,)

    # Multi-seed training sweeps (including seeds like 2002, 2011, 2999, 3999 that
    # previously triggered integer +1 collision bumps at `eval_seed=999`) evaluate on
    # strictly identical problem sets, and adjacent `ood_base3`/`ood_base4` splits
    # receive distinct domain-separated seeds.
    from itertools import islice

    from experiments.addition.inputs import iter_problems

    baseline_cfg = AdditionConfig(
        bases="2,8,10",
        eval_ood_bases="3,4,12",
        eval_permuted_canonical=True,
        seed=42,
        eval_seed=999,
    )
    baseline_splits = baseline_cfg.eval_splits()
    assert len({s.seed for s in baseline_splits}) == len(baseline_splits)
    baseline_problems = {
        s.name: list(
            islice(
                iter_problems(
                    bases=s.bases,
                    min_length=s.min_length,
                    max_length=s.max_length,
                    p_prior=s.p_prior,
                    permuted_canonical=s.permuted_canonical,
                    p_chain=s.p_chain,
                    seed=s.seed,
                ),
                10,
            )
        )
        for s in baseline_splits
    }
    for sweep_seed in (1999, 2002, 2003, 2011, 2999, 3999):
        sweep_cfg = AdditionConfig(
            bases="2,8,10",
            eval_ood_bases="3,4,12",
            eval_permuted_canonical=True,
            seed=sweep_seed,
            eval_seed=999,
        )
        assert sweep_cfg.eval_splits() == baseline_splits
        for s in sweep_cfg.eval_splits():
            probs = list(
                islice(
                    iter_problems(
                        bases=s.bases,
                        min_length=s.min_length,
                        max_length=s.max_length,
                        p_prior=s.p_prior,
                        permuted_canonical=s.permuted_canonical,
                        p_chain=s.p_chain,
                        seed=s.seed,
                    ),
                    10,
                )
            )
            assert probs == baseline_problems[s.name]
        # Domain-separated `"999:eval:id"` never replays integer `sweep_seed` stream.
        train_probs = list(islice(iter_problems(bases=(2, 8, 10), seed=sweep_seed), 10))
        assert train_probs != baseline_problems["id"]


def test_config_validation_rejections():
    with pytest.raises(ValueError, match="symbol_pool_size must be in 2..10"):
        AdditionConfig(symbol_set=SymbolSetName.DECIMAL, symbol_pool_size=11)
    with pytest.raises(ValueError, match="exceeds symbol_pool size"):
        AdditionConfig(
            bases="2,10",
            eval_ood_bases=None,
            symbol_set=SymbolSetName.DECIMAL,
            symbol_pool_size=5,
        )
    with pytest.raises(ValueError, match="eval_ood_bases entry 12 exceeds symbol_pool"):
        AdditionConfig(
            bases="2,8",
            eval_ood_bases="12",
            symbol_set=SymbolSetName.DECIMAL,
        )
    with pytest.raises(ValueError, match="p_prior > 0 requires bases <= 16"):
        AdditionConfig(bases="2,20", p_prior=0.5)
    with pytest.raises(
        ValueError, match="eval_permuted_canonical=True requires bases <= 16"
    ):
        AdditionConfig(bases="2,20", eval_permuted_canonical=True)
    with pytest.raises(ValueError, match="min_length must be >= 1"):
        AdditionConfig(min_length=0)
    with pytest.raises(ValueError, match="max_length .* must be >= min_length"):
        AdditionConfig(min_length=5, max_length=4)
    with pytest.raises(ValueError, match="eval_extrap_max_length .* must be >"):
        AdditionConfig(max_length=20, eval_extrap_max_length=20)
    with pytest.raises(ValueError, match="p_prior must be in 0.0..1.0"):
        AdditionConfig(p_prior=1.2)
    with pytest.raises(ValueError, match="p_chain must be in 0.0..1.0"):
        AdditionConfig(p_chain=-0.1)
    with pytest.raises(ValueError, match="eval_p_chain must be in 0.0..1.0"):
        AdditionConfig(eval_p_chain=1.5)
    with pytest.raises(ValueError, match="non-negative even int"):
        AdditionStepConfig(d_position=15)
    with pytest.raises(ValueError, match="d_position must be > 0"):
        AdditionStepConfig(embedding_mode=EmbeddingMode.TOKEN_TYPE_PE, d_position=0)
    with pytest.raises(ValueError, match="d_scratch must be >= 0"):
        AdditionConfig(d_scratch=-1)
    with pytest.raises(ValueError, match="n_layers must be > 0"):
        AdditionConfig(n_layers=0)
    with pytest.raises(ValueError, match="n_heads must be > 0"):
        AdditionConfig(n_heads=0)
    with pytest.raises(ValueError, match="divisible by n_heads"):
        AdditionConfig(n_heads=7)
    with pytest.raises(ValueError, match="d_ff must be > 0"):
        AdditionConfig(d_ff=0)
    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        AdditionConfig(max_steps=0)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        AdditionConfig(batch_size=0)
    for bad_lr in (0.0, -1e-4, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="lr must be > 0"):
            AdditionConfig(lr=bad_lr)
    with pytest.raises(ValueError, match="log_every must be >= 1"):
        AdditionConfig(log_every=0)
    with pytest.raises(ValueError, match="eval_every must be an int >= 0 or None"):
        AdditionConfig(eval_every=-5)
    with pytest.raises(ValueError, match="eval_every must be an int >= 0 or None"):
        AdditionConfig(eval_every=False)
    with pytest.raises(ValueError, match="num_eval_problems must be >= 1"):
        AdditionConfig(num_eval_problems=0)
    with pytest.raises(ValueError, match="eval_batch_size must be >= 1"):
        AdditionConfig(eval_batch_size=0)
    with pytest.raises(ValueError, match="eval_seed .* must differ from training seed"):
        AdditionConfig(seed=42, eval_seed=42)
    with pytest.raises(ValueError, match="p_terminal must be in 0.0..1.0"):
        AdditionClsConfig(p_terminal=1.5)
    with pytest.raises(ValueError, match="include_alphabet=False requires max_length"):
        AdditionAddressedStepArConfig(
            bases="2,10", max_length=8, eval_ood_bases=None, include_alphabet=False
        )
    with pytest.raises(
        ValueError, match="include_alphabet=False requires max_length .* eval_ood_bases"
    ):
        AdditionAddressedStepArConfig(
            bases="2,8",
            max_length=10,
            eval_ood_bases="3,12",
            include_alphabet=False,
        )
    with pytest.raises(ValueError, match="include_alphabet=False requires max_length"):
        AdditionAddressedArConfig(
            bases="2,10", max_length=8, eval_ood_bases=None, include_alphabet=False
        )


def test_json_round_trip():
    original = AdditionClsConfig(
        bases="2, 5, 10",
        min_length=2,
        max_length=8,
        symbol_set=SymbolSetName.HEX,
        symbol_pool_size=12,
        p_prior=0.25,
        p_chain=0.4,
        embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        d_position=8,
        d_scratch=16,
        fixed_d_model=False,
        max_steps=500,
        batch_size=32,
        lr=1e-3,
        log_every=50,
        seed=7,
        num_eval_problems=40,
        eval_batch_size=16,
        eval_seed=100,
        eval_p_chain=0.3,
        eval_ood_bases="3, 8",
        eval_extrap_max_length=None,
        eval_permuted_canonical=False,
        p_terminal=0.35,
        all_steps=True,
    )
    assert original.bases == "2,5,10"
    assert original.eval_ood_bases == "3,8"
    payload = json.dumps(original.to_dict())
    restored = AdditionClsConfig.from_dict(json.loads(payload))
    assert restored == original
    assert isinstance(restored.symbol_set, SymbolSetName)
    assert isinstance(restored.embedding_mode, EmbeddingMode)

    step_ar = AdditionStepArConfig(
        step_order=StepOrder.CARRY_FIRST, use_query_tokens=True
    )
    restored_step_ar = AdditionStepArConfig.from_dict(
        json.loads(json.dumps(step_ar.to_dict()))
    )
    assert restored_step_ar == step_ar
    assert isinstance(restored_step_ar.step_order, StepOrder)

    ar = AdditionArConfig(step_order=StepOrder.CARRY_FIRST, use_query_tokens=True)
    restored_ar = AdditionArConfig.from_dict(json.loads(json.dumps(ar.to_dict())))
    assert restored_ar == ar
    assert isinstance(restored_ar.step_order, StepOrder)

    addr_step_ar = AdditionAddressedStepArConfig(
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
        include_alphabet=False,
    )
    restored_addr_step_ar = AdditionAddressedStepArConfig.from_dict(
        json.loads(json.dumps(addr_step_ar.to_dict()))
    )
    assert restored_addr_step_ar == addr_step_ar
    assert isinstance(restored_addr_step_ar.step_order, StepOrder)
    assert restored_addr_step_ar.include_alphabet is False

    addr_ar = AdditionAddressedArConfig(
        step_order=StepOrder.CARRY_FIRST,
        use_query_tokens=True,
        include_alphabet=False,
    )
    restored_addr_ar = AdditionAddressedArConfig.from_dict(
        json.loads(json.dumps(addr_ar.to_dict()))
    )
    assert restored_addr_ar == addr_ar
    assert isinstance(restored_addr_ar.step_order, StepOrder)
    assert restored_addr_ar.include_alphabet is False


def test_config_import_does_not_load_torch():
    code = (
        "import sys\n"
        "import experiments.addition.config\n"
        "assert 'torch' not in sys.modules, 'importing addition.config loaded torch'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
