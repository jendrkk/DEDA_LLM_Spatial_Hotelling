"""Unit tests for GroupEnvelope and ChainEnvelopeOutput schema validation.

Verifies that Pydantic validators on GroupEnvelope reject out-of-range values
for price midpoint, delta_p width, and epsilon, and that ChainEnvelopeOutput
requires all expected group keys to be present.
"""
from __future__ import annotations

import pytest

from hotelling.llm.schemas import ChainEnvelopeOutput, GroupEnvelope  # noqa: F401


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_group_envelope_rejects_price_below_mc() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_group_envelope_rejects_narrow_delta_p() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_group_envelope_rejects_epsilon_out_of_range() -> None:
    pass


@pytest.mark.xfail(reason="not yet implemented", strict=True)
def test_chain_envelope_output_all_groups_present() -> None:
    pass
