"""Tests for `experiments.sir_yes_sir.config` (`SirYesSirConfig`, `SirYesSirClsConfig`)."""

import json
import subprocess
import sys

import pytest

from experiments.sir_yes_sir.config import (
    ClsSupervision,
    SirYesSirClsConfig,
    SirYesSirConfig,
    SymbolSetName,
)
from experiments.sir_yes_sir.inputs import EmbeddingMode
from yessir.alphabet import SymbolSet


def test_default_config_is_valid():
    cfg = SirYesSirConfig()
    assert cfg.alphabet_size == 4
    assert cfg.x_length == 3
    assert cfg.symbol_set == SymbolSetName.ALL
    assert cfg.symbol_pool_size is None
    assert cfg.symbol_pool == bytes(SymbolSet.ALL)
    assert cfg.resample_alphabet is True
    assert cfg.embedding_mode == EmbeddingMode.TOKEN_TYPE_PE
    assert cfg.use_token_type is True
    assert cfg.fixed_d_model is True
    assert cfg.effective_d_position == 32
    assert cfg.effective_d_scratch == 208
    assert cfg.d_model == 256 + 16 + 32 + 208 == 512
    assert cfg.derived_summary() == "d_model=512, symbol_pool_size=240"


def test_embedding_mode_and_d_position_combinations():
    # Default fixed_d_model=True keeps d_model=512 across modes by growing effective_d_scratch.
    cfg_tt_nope = SirYesSirConfig(embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE)
    assert cfg_tt_nope.use_token_type is True
    assert cfg_tt_nope.effective_d_position == 0
    assert cfg_tt_nope.effective_d_scratch == 208 + 32 == 240
    assert cfg_tt_nope.d_model == 256 + 16 + 0 + 240 == 512

    cfg_nope = SirYesSirConfig(embedding_mode=EmbeddingMode.NOPE)
    assert cfg_nope.use_token_type is False
    assert cfg_nope.effective_d_position == 0
    assert cfg_nope.effective_d_scratch == 208 + 16 + 32 == 256
    assert cfg_nope.d_model == 256 + 0 + 0 + 256 == 512

    # With fixed_d_model=False, omitted slices shrink d_model instead of growing d_scratch.
    cfg_tt_nope_shrink = SirYesSirConfig(
        embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE,
        fixed_d_model=False,
    )
    assert cfg_tt_nope_shrink.effective_d_position == 0
    assert cfg_tt_nope_shrink.effective_d_scratch == 208
    assert cfg_tt_nope_shrink.d_model == 256 + 16 + 0 + 208 == 480

    cfg_nope_shrink = SirYesSirConfig(
        embedding_mode=EmbeddingMode.NOPE,
        fixed_d_model=False,
    )
    assert cfg_nope_shrink.effective_d_position == 0
    assert cfg_nope_shrink.effective_d_scratch == 208
    assert cfg_nope_shrink.d_model == 256 + 0 + 0 + 208 == 464

    # TOKEN_TYPE_PE rejects d_position == 0.
    with pytest.raises(ValueError, match="d_position must be > 0"):
        SirYesSirConfig(embedding_mode=EmbeddingMode.TOKEN_TYPE_PE, d_position=0)

    # Odd or negative d_position is rejected.
    with pytest.raises(ValueError, match="non-negative even int"):
        SirYesSirConfig(d_position=15)
    with pytest.raises(ValueError, match="non-negative even int"):
        SirYesSirConfig(d_position=-2)


def test_symbol_pool_and_alphabet_size_validation():
    cfg = SirYesSirConfig(
        symbol_set=SymbolSetName.DECIMAL,
        symbol_pool_size=3,
        alphabet_size=3,
    )
    assert cfg.symbol_pool == b"012"

    # alphabet_size cannot exceed symbol_pool_size or symbol_set length.
    with pytest.raises(ValueError, match="alphabet_size must be in 2..3"):
        SirYesSirConfig(
            symbol_set=SymbolSetName.DECIMAL,
            symbol_pool_size=3,
            alphabet_size=4,
        )
    with pytest.raises(ValueError, match="alphabet_size must be in 2..10"):
        SirYesSirConfig(symbol_set=SymbolSetName.DECIMAL, alphabet_size=11)
    with pytest.raises(ValueError, match="alphabet_size must be in 2.."):
        SirYesSirConfig(alphabet_size=1)

    # symbol_pool_size itself must be in 2..len(symbol_set).
    with pytest.raises(ValueError, match="symbol_pool_size must be in 2..10"):
        SirYesSirConfig(symbol_set=SymbolSetName.DECIMAL, symbol_pool_size=1)
    with pytest.raises(ValueError, match="symbol_pool_size must be in 2..10"):
        SirYesSirConfig(symbol_set=SymbolSetName.DECIMAL, symbol_pool_size=11)


def test_model_and_training_bounds():
    with pytest.raises(ValueError, match="x_length must be >= 1"):
        SirYesSirConfig(x_length=0)
    with pytest.raises(ValueError, match="d_scratch must be >= 0"):
        SirYesSirConfig(d_scratch=-1)
    with pytest.raises(ValueError, match="n_layers must be > 0"):
        SirYesSirConfig(n_layers=0)
    with pytest.raises(ValueError, match="n_heads must be > 0"):
        SirYesSirConfig(n_heads=0)
    with pytest.raises(ValueError, match="divisible by n_heads"):
        SirYesSirConfig(n_heads=7)
    with pytest.raises(ValueError, match="d_ff must be > 0"):
        SirYesSirConfig(d_ff=0)
    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        SirYesSirConfig(max_steps=0)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        SirYesSirConfig(batch_size=0)
    with pytest.raises(ValueError, match="lr must be > 0"):
        SirYesSirConfig(lr=0.0)
    with pytest.raises(ValueError, match="num_eval_problems must be >= 1"):
        SirYesSirConfig(num_eval_problems=0)
    with pytest.raises(ValueError, match="eval_batch_size must be >= 1"):
        SirYesSirConfig(eval_batch_size=0)
    with pytest.raises(ValueError, match="eval_seed .* must differ from training seed"):
        SirYesSirConfig(seed=42, eval_seed=42)


def test_json_round_trip():
    original = SirYesSirClsConfig(
        alphabet_size=3,
        x_length=4,
        symbol_set=SymbolSetName.HEX,
        symbol_pool_size=6,
        resample_alphabet=False,
        embedding_mode=EmbeddingMode.TOKEN_TYPE_NOPE,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        d_position=8,
        d_scratch=16,
        fixed_d_model=False,
        max_steps=600,
        batch_size=24,
        lr=1e-3,
        seed=7,
        num_eval_problems=50,
        eval_batch_size=32,
        eval_seed=100,
        supervision=ClsSupervision.RANDOM_STEP,
    )
    payload = json.dumps(original.to_dict())
    restored = SirYesSirClsConfig.from_dict(json.loads(payload))
    assert restored == original
    assert isinstance(restored.symbol_set, SymbolSetName)
    assert isinstance(restored.embedding_mode, EmbeddingMode)
    assert isinstance(restored.supervision, ClsSupervision)


def test_config_import_does_not_load_torch():
    # Runs in a fresh interpreter: in this pytest process other tests have
    # already imported torch, so checking `sys.modules` here would prove nothing.
    code = (
        "import sys\n"
        "import experiments.sir_yes_sir.config\n"
        "assert 'torch' not in sys.modules, 'importing the config loaded torch'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
