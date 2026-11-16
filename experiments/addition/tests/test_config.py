"""Tests for `experiments.addition.config` (`AdditionConfig`, `AdditionClsConfig`, `parse_bases_csv`)."""

import json
import subprocess
import sys

import pytest

from experiments.addition.config import (
    AdditionClsConfig,
    AdditionConfig,
    AdditionStepConfig,
    AdditionStepForkConfig,
    EvalSplit,
    SymbolSetName,
    parse_bases_csv,
)
from experiments.addition.inputs import EmbeddingMode
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
    assert cfg.embedding_mode == EmbeddingMode.TOKEN_TYPE_PE
    assert cfg.use_token_type is True
    assert cfg.fixed_d_model is True
    assert cfg.effective_d_position == 32
    assert cfg.effective_d_scratch == 208
    assert cfg.d_model == 256 + 16 + 32 + 208 == 512
    assert cfg.log_every == 100
    assert (
        cfg.derived_summary()
        == "d_model=512, symbol_pool=240, splits=[id,ood_base3,ood_base12,extrap_len]"
    )

    cls_cfg = AdditionClsConfig()
    assert isinstance(cls_cfg, AdditionStepConfig)
    assert cls_cfg.p_terminal == 0.2
    assert cls_cfg.all_steps is False

    fork_cfg = AdditionStepForkConfig()
    assert isinstance(fork_cfg, AdditionStepConfig)
    assert fork_cfg.p_terminal == 0.2
    assert fork_cfg.all_steps is False


def test_embedding_mode_and_fixed_d_model_combinations():
    cfg_tt_nope = AdditionConfig(embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE)
    assert cfg_tt_nope.use_token_type is True
    assert cfg_tt_nope.effective_d_position == 0
    assert cfg_tt_nope.effective_d_scratch == 208 + 32 == 240
    assert cfg_tt_nope.d_model == 512

    cfg_nope = AdditionConfig(embedding_mode=EmbeddingMode.NOPE)
    assert cfg_nope.use_token_type is False
    assert cfg_nope.effective_d_position == 0
    assert cfg_nope.effective_d_scratch == 208 + 16 + 32 == 256
    assert cfg_nope.d_model == 512

    cfg_tt_nope_shrink = AdditionConfig(
        embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE,
        fixed_d_model=False,
    )
    assert cfg_tt_nope_shrink.effective_d_position == 0
    assert cfg_tt_nope_shrink.effective_d_scratch == 208
    assert cfg_tt_nope_shrink.d_model == 480

    cfg_nope_shrink = AdditionConfig(
        embedding_mode=EmbeddingMode.NOPE,
        fixed_d_model=False,
    )
    assert cfg_nope_shrink.effective_d_position == 0
    assert cfg_nope_shrink.effective_d_scratch == 208
    assert cfg_nope_shrink.d_model == 464


def test_eval_splits_construction_and_seed_non_collision():
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
    # Note: seed=1999 collides with eval_seed + 1000? For b=12, 999 + 1012 = 2011;
    # let's also explicitly check collision avoidance when seed == eval_seed + 2000.
    cfg_collide = AdditionConfig(
        bases="2,10",
        eval_ood_bases="",
        eval_extrap_max_length=30,
        seed=2999,
        eval_seed=999,
    )
    assert cfg_collide.eval_ood_bases is None
    assert cfg_collide.parsed_eval_ood_bases == ()
    collide_splits = cfg_collide.eval_splits()
    assert [s.name for s in collide_splits] == ["id", "extrap_len"]
    assert collide_splits[1].seed == 3000 != cfg_collide.seed

    # The stored value keeps only the OOD bases that get a split.
    assert cfg.eval_ood_bases == "12"
    assert cfg.parsed_eval_ood_bases == (12,)
    all_trained = AdditionConfig(bases="2,3,12", eval_ood_bases="3,12")
    assert all_trained.eval_ood_bases is None
    assert [s.name for s in all_trained.eval_splits()] == ["id", "extrap_len"]

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
            seed=999,
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
            seed=999 + 1012,
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
            seed=999 + 2000,
        ),
        EvalSplit(
            name="permuted",
            prefix="permuted_",
            bases=(2, 3, 10),
            min_length=1,
            max_length=10,
            p_prior=0.0,
            permuted_canonical=True,
            p_chain=0.3,
            seed=999 + 3000,
        ),
    )


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
        AdditionConfig(d_position=15)
    with pytest.raises(ValueError, match="d_position must be > 0"):
        AdditionConfig(embedding_mode=EmbeddingMode.TOKEN_TYPE_PE, d_position=0)
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
    with pytest.raises(ValueError, match="lr must be > 0"):
        AdditionConfig(lr=0.0)
    with pytest.raises(ValueError, match="log_every must be >= 1"):
        AdditionConfig(log_every=0)
    with pytest.raises(ValueError, match="num_eval_problems must be >= 1"):
        AdditionConfig(num_eval_problems=0)
    with pytest.raises(ValueError, match="eval_batch_size must be >= 1"):
        AdditionConfig(eval_batch_size=0)
    with pytest.raises(ValueError, match="eval_seed .* must differ from training seed"):
        AdditionConfig(seed=42, eval_seed=42)
    with pytest.raises(ValueError, match="p_terminal must be in 0.0..1.0"):
        AdditionClsConfig(p_terminal=1.5)


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
        eval_permuted_canonical=True,
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


def test_config_import_does_not_load_torch():
    code = (
        "import sys\n"
        "import experiments.addition.config\n"
        "assert 'torch' not in sys.modules, 'importing addition.config loaded torch'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
