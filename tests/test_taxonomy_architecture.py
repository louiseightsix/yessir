"""Tests for the 5-Axis Decoupled Model Architecture and Admissibility Matrix.
Conforms to spec/04_experimental_matrix.md and spec/06_implementation_blueprint.md.
"""

import pytest
import torch

from transformadder.core.constants import D_MODEL
from transformadder.core.embedding import Representation
from transformadder.core.heads import Channel, MultiHeadOutput, TiedIdentityHead
from transformadder.core.model import (
    ADMISSIBLE_TUPLES,
    Agency,
    Topology,
    TransformAdder,
)


def test_admissible_tuples_instantiation():
    # Test each of the 4 admissible tuples across representations
    configs = [
        (Agency.AUTONOMOUS, Topology.UNIFIED, Channel.SEQUENCE, Representation.ROLE_EMBED),
        (Agency.HARNESSED, Topology.UNIFIED, Channel.WEIGHT, Representation.EXPLICIT_CONTINUOUS),
        (Agency.HARNESSED, Topology.PIPELINED, Channel.SEQUENCE, Representation.EXPLICIT_NOPE),
        (Agency.HARNESSED, Topology.BRANCHING, Channel.SEQUENCE, Representation.ADDRESSED_KV),
    ]

    for agency, topology, channel, repr_type in configs:
        model = TransformAdder.from_axes(
            agency=agency,
            topology=topology,
            channel=channel,
            representation=repr_type,
            n_layers=2,
        )
        assert model.agency == agency
        assert model.topology == topology
        assert model.channel == channel
        assert model.representation == repr_type
        if channel == Channel.WEIGHT:
            assert isinstance(model.head, MultiHeadOutput)
        else:
            assert isinstance(model.head, TiedIdentityHead)


def test_inadmissible_tuples_raise_value_error():
    # Inadmissible: Autonomous + Branching + Weight
    with pytest.raises(ValueError, match="Inadmissible experimental tuple"):
        TransformAdder.from_axes(
            agency=Agency.AUTONOMOUS,
            topology=Topology.BRANCHING,
            channel=Channel.WEIGHT,
            representation=Representation.EXPLICIT_CONTINUOUS,
            n_layers=2,
        )

    # Inadmissible: Harnessed + Unified + Sequence
    with pytest.raises(ValueError, match="Inadmissible experimental tuple"):
        TransformAdder.from_axes(
            agency=Agency.HARNESSED,
            topology=Topology.UNIFIED,
            channel=Channel.SEQUENCE,
            representation=Representation.ROLE_EMBED,
            n_layers=2,
        )


def test_admissible_model_forward_passes():
    # 1. Autonomous + Unified + Sequence (AR)
    ar_model = TransformAdder.from_axes(
        agency="autonomous",
        topology="unified",
        channel="sequence",
        representation="R-Explicit-Continuous",
        n_layers=2,
    )
    tokens = torch.randint(0, 256, (2, 16))
    out = ar_model(tokens)
    assert out.shape == (2, 16, 256)

    # 2. Harnessed + Unified + Weight (SM-MH)
    sm_mh_model = TransformAdder.from_axes(
        agency="harnessed",
        topology="unified",
        channel="weight",
        representation="R-Explicit-Continuous",
        n_layers=2,
    )
    lengths = torch.tensor([10, 16])
    l_sum, l_carry, l_flag = sm_mh_model.forward_sm_mh(tokens, None, None, lengths)
    assert l_sum.shape == (2, 256)
    assert l_carry.shape == (2, 256)
    assert l_flag.shape == (2, 2)
