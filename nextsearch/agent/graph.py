from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from nextsearch.agent.nodes import QueryAgentNodes, route_after_decision
from nextsearch.agent.state import QueryAgentInput, QueryAgentOutput, QueryAgentState
from nextsearch.llm.service import LLMService
from nextsearch.retrieval.graph_store import GraphStore
from nextsearch.retrieval.source_store import SourceStore


def build_query_agent(
    *,
    llm: LLMService,
    graph_store: GraphStore,
    source_store: SourceStore,
    max_search_iterations: int = 3,
):
    nodes = QueryAgentNodes(
        llm=llm,
        graph_store=graph_store,
        source_store=source_store,
        max_search_iterations=max_search_iterations,
    )

    builder = StateGraph(
        QueryAgentState,
        input_schema=QueryAgentInput,
        output_schema=QueryAgentOutput,
    )
    builder.add_node("plan_query", nodes.plan_query)
    builder.add_node("search_graph", nodes.search_graph)
    builder.add_node("inspect_evidence", nodes.inspect_evidence)
    builder.add_node("decide_next_step", nodes.decide_next_step)
    builder.add_node("generate_answer", nodes.generate_answer)

    builder.add_edge(START, "plan_query")
    builder.add_edge("plan_query", "search_graph")
    builder.add_edge("search_graph", "inspect_evidence")
    builder.add_edge("inspect_evidence", "decide_next_step")
    builder.add_conditional_edges(
        "decide_next_step",
        route_after_decision,
        {
            "search_graph": "search_graph",
            "generate_answer": "generate_answer",
        },
    )
    builder.add_edge("generate_answer", END)
    return builder.compile(name="query_agent")
