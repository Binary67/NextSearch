"""Provider-neutral query agent for NextSearch."""

from nextsearch.agent.models import (
    AgentAnswer,
    Citation,
    EvidenceRecord,
    GraphSearchDecision,
    QueryPlan,
    QueryRequest,
)

__all__ = [
    "AgentAnswer",
    "Citation",
    "EvidenceRecord",
    "GraphSearchDecision",
    "QueryPlan",
    "QueryRequest",
    "build_query_agent",
]


def __getattr__(name: str):
    if name == "build_query_agent":
        from nextsearch.agent.graph import build_query_agent

        return build_query_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
