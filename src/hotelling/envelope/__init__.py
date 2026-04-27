"""Strategy envelope package: group envelopes, chain envelopes, division registry."""
from hotelling.envelope.envelope import ChainEnvelope, GroupEnvelope
from hotelling.envelope.groups import GroupDivision, assign_groups

__all__ = ["ChainEnvelope", "GroupDivision", "GroupEnvelope", "assign_groups"]
