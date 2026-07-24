"""Workflow detector for identifying flawed workflow design.

Detects F5: Flawed Workflow Design (MAST Taxonomy):
- Unreachable nodes (dead ends)
- Missing error handling paths
- Infinite loop potential
- Bottleneck nodes
- Missing termination conditions
- Orphan nodes
- Excessive sequential depth

Version History:
- v1.0: Initial pisama-core port from backend v1.1
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.models import Trace


@dataclass
class WorkflowNode:
    """A node in a workflow graph."""

    id: str
    name: str
    node_type: str
    incoming: list[str]
    outgoing: list[str]
    has_error_handler: bool = False
    is_terminal: bool = False


class WorkflowDetector(BaseDetector):
    """Detects flawed workflow design -- structural problems in agent graphs.

    Analyzes workflow topology for unreachable nodes, dead ends,
    missing error handling, bottlenecks, orphan nodes, excessive depth,
    and other structural issues.
    """

    name = "workflow"
    description = "Detects structural problems in agent workflow design"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (10, 90)
    realtime_capable = False

    # Configuration
    require_error_handling: bool = True
    max_bottleneck_ratio: float = 0.5

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect workflow issues in a trace.

        Builds a workflow graph from trace spans and analyzes topology.
        Also supports direct workflow_definition input via trace metadata.
        """
        # Check for golden-dataset-style input in metadata
        workflow_def = trace.metadata.custom.get("workflow_definition", "")
        execution_result = trace.metadata.custom.get("execution_result", "")

        if workflow_def or execution_result:
            nodes = self._nodes_from_definition(workflow_def, execution_result)
        else:
            nodes = self._nodes_from_trace(trace)

        if not nodes:
            return DetectionResult.no_issue(self.name)

        return self._analyze_workflow(nodes)

    def _nodes_from_trace(self, trace: Trace) -> list[WorkflowNode]:
        """Build workflow nodes from trace spans."""
        spans = sorted(trace.spans, key=lambda s: s.start_time)
        if not spans:
            return []

        nodes: list[WorkflowNode] = []
        for i, span in enumerate(spans):
            incoming = [spans[i - 1].name] if i > 0 else []
            outgoing = [spans[i + 1].name] if i < len(spans) - 1 else []

            nodes.append(
                WorkflowNode(
                    id=span.span_id,
                    name=span.name,
                    node_type=str(span.kind),
                    incoming=incoming,
                    outgoing=outgoing,
                    has_error_handler="error" in str(span.attributes),
                    is_terminal=(i == len(spans) - 1),
                )
            )

        return nodes

    def _nodes_from_definition(
        self,
        workflow_def: Any,
        execution_result: Any,
    ) -> list[WorkflowNode]:
        """Build workflow nodes from golden-dataset-style definition."""
        if isinstance(workflow_def, dict):
            raw_nodes = workflow_def.get("nodes", [])
        elif isinstance(workflow_def, str):
            # Parse simple text-based workflow definitions
            import json

            try:
                parsed = json.loads(workflow_def)
                raw_nodes = parsed.get("nodes", [])
            except (json.JSONDecodeError, AttributeError):
                return []
        else:
            return []

        nodes: list[WorkflowNode] = []
        for i, raw in enumerate(raw_nodes):
            if isinstance(raw, dict):
                nodes.append(
                    WorkflowNode(
                        id=raw.get("id", f"node_{i}"),
                        name=raw.get("name", f"node_{i}"),
                        node_type=raw.get("type", "agent"),
                        incoming=raw.get("incoming", []),
                        outgoing=raw.get("outgoing", []),
                        has_error_handler=raw.get("has_error_handler", False),
                        is_terminal=raw.get("is_terminal", False),
                    )
                )
        return nodes

    def _analyze_workflow(self, nodes: list[WorkflowNode]) -> DetectionResult:
        """Run all workflow analysis checks and produce a detection result."""
        forward, backward = self._build_graph(nodes)
        edge_count = sum(len(edges) for edges in forward.values())

        issues: list[str] = []
        problematic: list[str] = []

        unreachable = self._detect_unreachable(nodes, forward, backward)
        if unreachable:
            issues.append("unreachable_node")
            problematic.extend(unreachable)

        dead_ends = self._detect_dead_ends(nodes, forward)
        if dead_ends:
            issues.append("dead_end")
            problematic.extend(dead_ends)

        loop_risk = self._detect_infinite_loop_risk(nodes, forward)
        if loop_risk:
            issues.append("infinite_loop_risk")
            problematic.extend(loop_risk)

        bottlenecks = self._detect_bottlenecks(nodes, forward, backward)
        if bottlenecks:
            issues.append("bottleneck")
            problematic.extend(bottlenecks)

        orphans = self._detect_orphan_nodes(nodes, forward, backward)
        if orphans:
            issues.append("orphan_node")
            problematic.extend(orphans)

        max_depth = self._detect_excessive_depth(nodes, forward, backward)
        if max_depth > 5:
            issues.append("excessive_depth")

        if self.require_error_handling:
            missing_handlers = self._detect_missing_error_handling(nodes)
            non_trivial = [n for n in nodes if n.node_type not in ("start", "end", "condition")]
            has_any_handler = any(n.has_error_handler for n in nodes)
            if (
                missing_handlers
                and has_any_handler
                and len(missing_handlers) >= len(non_trivial) * 0.5
            ):
                issues.append("missing_error_handling")
                problematic.extend(missing_handlers[:5])

        if self._detect_missing_termination(nodes, forward):
            issues.append("missing_termination")

        if not issues:
            return DetectionResult.no_issue(self.name)

        # Severity
        if "infinite_loop_risk" in issues or "missing_termination" in issues:
            severity = 80
        elif len(issues) >= 3:
            severity = 55
        else:
            severity = 30

        # Confidence
        if severity >= 80:
            confidence = min(0.75 + len(issues) * 0.05, 0.95)
        elif len(issues) >= 3:
            confidence = min(0.6 + len(issues) * 0.05, 0.90)
        elif len(issues) >= 2:
            confidence = 0.55
        else:
            single_issue = issues[0]
            if single_issue in ("dead_end", "unreachable_node"):
                confidence = 0.70
            elif single_issue == "excessive_depth":
                confidence = 0.60
            elif single_issue in ("orphan_node", "bottleneck", "missing_error_handling"):
                confidence = 0.50
            else:
                confidence = 0.55

        unique_problematic = list(set(problematic))[:5]
        summary = (
            f"Workflow has {len(issues)} structural issues: {', '.join(issues)}. "
            f"Affected nodes: {', '.join(unique_problematic)}"
        )

        # Build fix instruction
        fixes: list[str] = []
        if "unreachable_node" in issues:
            fixes.append("Add edges to unreachable nodes or remove them")
        if "dead_end" in issues:
            fixes.append("Add termination or continuation from dead-end nodes")
        if "infinite_loop_risk" in issues:
            fixes.append("Add loop counters or termination conditions")
        if "bottleneck" in issues:
            fixes.append("Parallelize or split bottleneck nodes")
        if "missing_error_handling" in issues:
            fixes.append("Add error handlers to critical nodes")

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=summary,
            fix_type=FixType.SWITCH_STRATEGY,
            fix_instruction="; ".join(fixes) if fixes else "Fix workflow structural issues",
        )
        result.confidence = confidence
        result.metadata = {
            "issues": issues,
            "node_count": len(nodes),
            "edge_count": edge_count,
            "problematic_nodes": list(set(problematic)),
        }
        return result

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(
        self,
        nodes: list[WorkflowNode],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Build forward and backward adjacency maps."""
        forward: dict[str, list[str]] = defaultdict(list)
        backward: dict[str, list[str]] = defaultdict(list)
        for node in nodes:
            for target in node.outgoing:
                forward[node.id].append(target)
                backward[target].append(node.id)
        return dict(forward), dict(backward)

    def _find_reachable(
        self,
        start: str,
        graph: dict[str, list[str]],
    ) -> set[str]:
        """Find all nodes reachable from start via DFS."""
        visited: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)
        return visited

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _detect_unreachable(
        self,
        nodes: list[WorkflowNode],
        forward: dict[str, list[str]],
        backward: dict[str, list[str]],
    ) -> list[str]:
        """Detect nodes unreachable from any entry point."""
        entry_points = [n.id for n in nodes if not backward.get(n.id)]
        if not entry_points:
            entry_points = [nodes[0].id] if nodes else []

        reachable: set[str] = set()
        for entry in entry_points:
            reachable |= self._find_reachable(entry, forward)

        all_nodes = {n.id for n in nodes}
        return list(all_nodes - reachable)

    def _detect_dead_ends(
        self,
        nodes: list[WorkflowNode],
        forward: dict[str, list[str]],
    ) -> list[str]:
        """Detect non-terminal nodes with no outgoing edges."""
        terminal_nodes = {n.id for n in nodes if n.is_terminal}
        dead_ends: list[str] = []
        for node in nodes:
            if not forward.get(node.id) and node.id not in terminal_nodes:
                dead_ends.append(node.id)
        return dead_ends

    def _detect_infinite_loop_risk(
        self,
        nodes: list[WorkflowNode],
        forward: dict[str, list[str]],
    ) -> list[str]:
        """Detect self-loops and cycles in the workflow graph."""
        risky: list[str] = []

        # Self-loops
        for node in nodes:
            if node.id in forward.get(node.id, []):
                risky.append(node.id)

        # General cycles
        def find_cycles(node_id: str, path: set[str]) -> bool:
            if node_id in path:
                return True
            path = path | {node_id}
            for neighbor in forward.get(node_id, []):
                if find_cycles(neighbor, path):
                    return True
            return False

        for node in nodes:
            if node.id not in risky:
                if find_cycles(node.id, set()):
                    risky.append(node.id)

        return list(set(risky))

    def _detect_bottlenecks(
        self,
        nodes: list[WorkflowNode],
        forward: dict[str, list[str]],
        backward: dict[str, list[str]],
    ) -> list[str]:
        """Detect throughput bottlenecks and convergence hubs."""
        bottlenecks: list[str] = []
        total_edges = sum(len(edges) for edges in forward.values())
        num_nodes = len(nodes)

        for node in nodes:
            incoming = len(backward.get(node.id, []))
            outgoing = len(forward.get(node.id, []))
            if total_edges > 0:
                throughput = (incoming + outgoing) / total_edges
                is_throughput_bottleneck = throughput > self.max_bottleneck_ratio and incoming > 2
                is_convergence_hub = num_nodes > 3 and incoming > num_nodes / 2
                if is_throughput_bottleneck or is_convergence_hub:
                    bottlenecks.append(node.id)

        return bottlenecks

    def _detect_missing_error_handling(
        self,
        nodes: list[WorkflowNode],
    ) -> list[str]:
        """Detect nodes that lack error handlers."""
        missing: list[str] = []
        for node in nodes:
            if not node.has_error_handler and node.node_type not in ["start", "end", "condition"]:
                missing.append(node.id)
        return missing

    def _detect_orphan_nodes(
        self,
        nodes: list[WorkflowNode],
        forward: dict[str, list[str]],
        backward: dict[str, list[str]],
    ) -> list[str]:
        """Detect orphan nodes -- disconnected from the main flow."""
        if len(nodes) <= 1:
            return []

        primary_start = nodes[0].id if nodes else None
        orphans: list[str] = []
        for node in nodes:
            if node.id == primary_start:
                continue
            has_incoming = bool(backward.get(node.id))
            has_outgoing = bool(forward.get(node.id))
            if not has_incoming and not has_outgoing:
                orphans.append(node.id)
            elif not has_incoming and node.node_type not in ("start",):
                orphans.append(node.id)
        return orphans

    def _detect_excessive_depth(
        self,
        nodes: list[WorkflowNode],
        forward: dict[str, list[str]],
        backward: dict[str, list[str]],
    ) -> int:
        """Detect excessive sequential chain depth. Returns the longest path length.

        Workflows with depth > 8 are likely over-sequential and should be
        parallelised or decomposed.
        """
        if not nodes:
            return 0

        entry_points = [n.id for n in nodes if not backward.get(n.id)]
        if not entry_points:
            entry_points = [nodes[0].id]

        max_depth = 0
        for start in entry_points:
            visited: set[str] = set()
            stack = [(start, 0)]
            while stack:
                node_id, depth = stack.pop()
                if node_id in visited:
                    continue
                visited.add(node_id)
                max_depth = max(max_depth, depth)
                for neighbor in forward.get(node_id, []):
                    if neighbor not in visited:
                        stack.append((neighbor, depth + 1))

        return max_depth

    def _detect_missing_termination(
        self,
        nodes: list[WorkflowNode],
        forward: dict[str, list[str]],
    ) -> bool:
        """Check if workflow lacks terminal nodes."""
        terminal_nodes = [n for n in nodes if n.is_terminal]
        return not terminal_nodes
