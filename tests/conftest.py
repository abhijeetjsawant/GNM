from pathlib import Path

import pytest

from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.rig import ControlRig
from autoanim_gnm.semantic_decoder import ExpressionDecoder


@pytest.fixture(scope="session")
def adapter() -> GNMAdapter:
    return GNMAdapter()


@pytest.fixture(scope="session")
def decoder() -> ExpressionDecoder:
    return ExpressionDecoder(
        Path("gnm/shape/data/semantic_sampler/expression_decoder_model.h5")
    )


@pytest.fixture(scope="session")
def rig(adapter: GNMAdapter, decoder: ExpressionDecoder) -> ControlRig:
    return ControlRig(adapter, decoder)
