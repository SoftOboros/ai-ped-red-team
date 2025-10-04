"""State-chart validation utilities."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import networkx as nx
from pydantic import BaseModel


class ValidationReport(BaseModel):
    """Report describing the validity of a state chart."""

    valid: bool
    issues: List[str]


class ChartEdge(BaseModel):
    source: str
    target: str


def validate_chart(
    nodes: Sequence[str] | Iterable[str],
    edges: Sequence[ChartEdge] | Iterable[ChartEdge],
    *,
    allow_cycles: bool = False,
) -> ValidationReport:
    """Validate structure of a state chart using NetworkX."""

    node_ids = list(nodes)
    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    for edge in edges:
        graph.add_edge(edge.source, edge.target)

    issues: List[str] = []

    if not node_ids:
        issues.append("Chart must contain at least one node.")
        return ValidationReport(valid=False, issues=issues)

    missing_nodes = [n for n in node_ids if n not in graph.nodes]
    if missing_nodes:
        issues.append(f"Missing nodes referenced in edges: {missing_nodes}.")

    orphans = [n for n in graph.nodes if graph.degree(n) == 0]
    if orphans:
        issues.append(f"Orphan nodes detected: {orphans}.")

    roots = [n for n in graph.nodes if graph.in_degree(n) == 0]
    if len(roots) == 0:
        issues.append("State chart must have at least one entry node.")
    elif len(roots) > 1:
        issues.append(f"State chart must have a single entry node. Found: {roots}.")

    terminals = [n for n in graph.nodes if graph.out_degree(n) == 0]
    if not terminals:
        issues.append("State chart must include at least one terminal node.")

    if not allow_cycles:
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            issues.append(f"Cycles are not permitted: {cycles}.")

    valid = not issues
    return ValidationReport(valid=valid, issues=issues)


__all__ = ["ChartEdge", "ValidationReport", "validate_chart"]
