"""Convergence detector for identifying metric convergence issues.

Detects convergence problems in iterative agent systems:
- Plateau: Metric improvement stalls for extended periods
- Regression: Metric worsens past a previously achieved best
- Thrashing: Metric oscillates without a clear trend
- Divergence: Metric consistently trends in the wrong direction

Designed for autonomous research agents (e.g., Karpathy's autoresearch),
iterative code generation, and any agent system that produces numerical
performance signals over multiple iterations.

Version History:
- v1.0: Initial pisama-core port from backend v1.0
"""

from typing import Any, Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.models import Trace


class ConvergenceDetector(BaseDetector):
    """Detects convergence issues in iterative agent metric sequences.

    Analyzes a time series of metric values to detect plateau, regression,
    thrashing, and divergence patterns.
    """

    name = "convergence"
    description = "Detects metric plateau, regression, thrashing, and divergence"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (10, 90)
    realtime_capable = False

    # Configuration
    plateau_threshold: float = 0.02
    plateau_window: int = 10
    regression_tolerance: float = 0.02
    thrashing_min_reversals: int = 3
    min_steps: int = 3

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect convergence issues in a trace.

        Expects metric data either in trace metadata (golden-dataset style)
        or extracted from span attributes.
        """
        # Check for golden-dataset-style input
        metrics = trace.metadata.custom.get("metrics")
        direction = trace.metadata.custom.get("direction", "minimize")
        window_size = trace.metadata.custom.get("window_size")

        if metrics:
            return self._detect_from_metrics(metrics, direction, window_size)

        # Extract metrics from trace spans
        metric_values = self._extract_metrics_from_trace(trace)
        if not metric_values:
            return DetectionResult.no_issue(self.name)

        return self._detect_from_metrics(metric_values, direction, window_size)

    def _extract_metrics_from_trace(self, trace: Trace) -> list[dict[str, Any]]:
        """Extract metric values from trace spans.

        Looks for spans with metric data in attributes or output_data.
        """
        metrics: list[dict[str, Any]] = []
        sorted_spans = sorted(trace.spans, key=lambda s: s.start_time)

        for span in sorted_spans:
            value = None

            # Check attributes for metric values
            for key in ("metric_value", "score", "loss", "accuracy", "f1", "reward"):
                if key in span.attributes:
                    try:
                        value = float(span.attributes[key])
                    except (ValueError, TypeError):
                        continue
                    break

            # Check output_data
            if value is None and span.output_data:
                for key in ("value", "metric", "score", "loss", "accuracy"):
                    if key in span.output_data:
                        try:
                            value = float(span.output_data[key])
                        except (ValueError, TypeError):
                            continue
                        break

            if value is not None:
                metrics.append(
                    {
                        "value": value,
                        "step": len(metrics),
                        "label": span.name,
                    }
                )

        return metrics

    def _detect_from_metrics(
        self,
        metrics: list[dict[str, Any]],
        direction: str = "minimize",
        window_size: Optional[int] = None,
    ) -> DetectionResult:
        """Core detection logic operating on metric data.

        Faithfully ports the backend ConvergenceDetector.detect_convergence_issues() method.
        """
        if not metrics or len(metrics) < self.min_steps:
            return DetectionResult.no_issue(self.name)

        values = [m["value"] for m in metrics]
        window = window_size or self.plateau_window

        issues: list[dict[str, Any]] = []

        # Track best value
        if direction == "minimize":
            best_value = min(values)
            best_idx = values.index(best_value)
        else:
            best_value = max(values)
            best_idx = values.index(best_value)

        current_value = values[-1]
        steps_since_best = len(values) - 1 - best_idx

        # --- Check 1: Regression ---
        regression_issue = self._check_regression(
            values,
            best_value,
            current_value,
            direction,
            steps_since_best,
        )
        if regression_issue:
            issues.append(regression_issue)

        # --- Check 2: Plateau ---
        plateau_issue = self._check_plateau(values, direction, window)
        if plateau_issue:
            issues.append(plateau_issue)

        # --- Check 3: Thrashing ---
        thrashing_issue = self._check_thrashing(values, window)
        if thrashing_issue:
            issues.append(thrashing_issue)

        # --- Check 4: Divergence ---
        divergence_issue = self._check_divergence(values, direction, window)
        if divergence_issue:
            issues.append(divergence_issue)

        # Compute improvement rate
        recent = values[-min(window, len(values)) :]
        if len(recent) >= 2:
            if direction == "minimize":
                improvement_rate = (
                    (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-10) / len(recent)
                )
            else:
                improvement_rate = (
                    (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-10) / len(recent)
                )
        else:
            improvement_rate = 0.0

        # --- Overall trend guard ---
        first_val, last_val = values[0], values[-1]
        scale = max(abs(first_val), abs(last_val), 1e-10)
        if direction == "minimize":
            overall_improvement = (first_val - last_val) / scale
        else:
            overall_improvement = (last_val - first_val) / scale

        # Check smoothness
        reversals = sum(
            1
            for i in range(2, len(values))
            if (values[i] - values[i - 1]) * (values[i - 1] - values[i - 2]) < 0
        )
        max_reversals = max(1, len(values) - 2)
        is_smooth = (reversals / max_reversals) < 0.60

        if overall_improvement > 0.30 and is_smooth:
            issues = [i for i in issues if i["failure_type"] not in ("regression", "thrashing")]

        if overall_improvement > 0.50:
            issues = [
                i
                for i in issues
                if i["failure_type"] != "plateau"
                or i["severity_level"] not in ("minor", "moderate")
            ]

        if not issues:
            result = DetectionResult.no_issue(self.name)
            result.metadata = {
                "best_value": best_value,
                "current_value": current_value,
                "improvement_rate": improvement_rate,
                "steps_since_best": steps_since_best,
            }
            return result

        # Pick primary issue by type priority then severity
        type_priority = {
            "divergence": 4,
            "thrashing": 3,
            "plateau": 2,
            "regression": 1,
        }
        severity_rank = {
            "critical": 4,
            "severe": 3,
            "moderate": 2,
            "minor": 1,
            "none": 0,
        }
        issues.sort(
            key=lambda i: (
                type_priority.get(i["failure_type"], 0),
                severity_rank.get(i["severity_level"], 0),
            ),
            reverse=True,
        )
        primary = issues[0]

        confidence = primary.get("confidence", 0.5)

        # Map severity level to numeric
        severity_map = {"critical": 85, "severe": 65, "moderate": 45, "minor": 25}
        severity = severity_map.get(primary["severity_level"], 45)

        # Determine fix type based on failure type
        fix_map = {
            "divergence": (
                FixType.TERMINATE,
                "Stop the current approach -- metric is diverging. Try a fundamentally different strategy.",
            ),
            "thrashing": (
                FixType.SWITCH_STRATEGY,
                "Metric is oscillating. Reduce learning rate, add momentum, or stabilize the approach.",
            ),
            "plateau": (
                FixType.SWITCH_STRATEGY,
                "Metric has plateaued. Try a different hyperparameter schedule or exploration strategy.",
            ),
            "regression": (
                FixType.ROLLBACK,
                "Metric has regressed from best. Consider rolling back to the best checkpoint.",
            ),
        }
        fix_type, fix_instruction = fix_map.get(
            primary["failure_type"],
            (FixType.SWITCH_STRATEGY, "Address the convergence issue."),
        )

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=primary["description"],
            fix_type=fix_type,
            fix_instruction=fix_instruction,
        )
        result.confidence = confidence
        result.metadata = {
            "primary_failure": primary["failure_type"],
            "issue_count": len(issues),
            "direction": direction,
            "num_steps": len(values),
            "window_size": window,
            "best_value": best_value,
            "current_value": current_value,
            "improvement_rate": improvement_rate,
            "steps_since_best": steps_since_best,
        }
        for issue in issues:
            result.add_evidence(
                description=issue["description"],
                data=issue.get("evidence", {}),
            )
        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_regression(
        self,
        values: list[float],
        best_value: float,
        current_value: float,
        direction: str,
        steps_since_best: int,
    ) -> Optional[dict[str, Any]]:
        """Check if current value has regressed past the best."""
        if steps_since_best == 0:
            return None

        if best_value == 0:
            regression_frac = abs(current_value - best_value)
        else:
            regression_frac = abs(current_value - best_value) / abs(best_value)

        if direction == "minimize":
            regressed = current_value > best_value * (1 + self.regression_tolerance)
        else:
            regressed = current_value < best_value * (1 - self.regression_tolerance)

        if not regressed:
            return None

        confidence = min(1.0, regression_frac / max(self.regression_tolerance, 1e-10))

        if regression_frac > 0.1:
            severity_level = "critical"
        elif regression_frac > 0.05:
            severity_level = "severe"
        elif regression_frac > self.regression_tolerance:
            severity_level = "moderate"
        else:
            severity_level = "minor"

        return {
            "failure_type": "regression",
            "description": (
                f"Metric regressed by {regression_frac:.1%} from best value "
                f"({best_value:.4f}) to current ({current_value:.4f}), "
                f"{steps_since_best} steps after best."
            ),
            "severity_level": severity_level,
            "confidence": confidence,
            "evidence": {
                "regression_frac": regression_frac,
                "steps_since_best": steps_since_best,
            },
        }

    def _check_plateau(
        self,
        values: list[float],
        direction: str,
        window: int,
    ) -> Optional[dict[str, Any]]:
        """Check if metric has plateaued (no meaningful improvement)."""
        recent = values[-min(window, len(values)) :]
        if len(recent) < 2:
            return None

        improvements: list[float] = []
        for i in range(1, len(recent)):
            if direction == "minimize":
                imp = recent[i - 1] - recent[i]
            else:
                imp = recent[i] - recent[i - 1]
            improvements.append(imp)

        scale = max(abs(v) for v in recent) if any(v != 0 for v in recent) else 1.0
        normalized_improvements = [imp / scale for imp in improvements]
        avg_improvement = sum(normalized_improvements) / len(normalized_improvements)

        if avg_improvement >= self.plateau_threshold:
            return None

        stalled_steps = sum(1 for ni in normalized_improvements if ni < self.plateau_threshold)
        stall_ratio = stalled_steps / len(normalized_improvements)

        confidence = min(1.0, stall_ratio)

        if stall_ratio >= 0.9:
            severity_level = "severe"
        elif stall_ratio >= 0.7:
            severity_level = "moderate"
        else:
            severity_level = "minor"

        return {
            "failure_type": "plateau",
            "description": (
                f"Metric plateaued: avg improvement {avg_improvement:.6f} per step "
                f"over last {len(recent)} steps (threshold: {self.plateau_threshold}). "
                f"{stalled_steps}/{len(normalized_improvements)} steps showed no meaningful progress."
            ),
            "severity_level": severity_level,
            "confidence": confidence,
            "evidence": {
                "avg_improvement": avg_improvement,
                "stalled_steps": stalled_steps,
                "stall_ratio": stall_ratio,
            },
        }

    def _check_thrashing(
        self,
        values: list[float],
        window: int,
    ) -> Optional[dict[str, Any]]:
        """Check if metric is oscillating (frequent direction changes)."""
        recent = values[-min(window, len(values)) :]
        if len(recent) < 3:
            return None

        reversals = 0
        for i in range(2, len(recent)):
            prev_dir = recent[i - 1] - recent[i - 2]
            curr_dir = recent[i] - recent[i - 1]
            if prev_dir * curr_dir < 0:
                reversals += 1

        max_possible_reversals = len(recent) - 2
        if max_possible_reversals == 0:
            return None

        if reversals < self.thrashing_min_reversals:
            return None

        reversal_ratio = reversals / max_possible_reversals
        confidence = min(1.0, reversal_ratio)

        if reversal_ratio >= 0.8:
            severity_level = "severe"
        elif reversal_ratio >= 0.6:
            severity_level = "moderate"
        else:
            severity_level = "minor"

        return {
            "failure_type": "thrashing",
            "description": (
                f"Metric thrashing: {reversals} direction reversals in {len(recent)} steps "
                f"(reversal ratio: {reversal_ratio:.0%}). "
                f"No consistent trend detected."
            ),
            "severity_level": severity_level,
            "confidence": confidence,
            "evidence": {
                "reversals": reversals,
                "reversal_ratio": reversal_ratio,
                "max_possible_reversals": max_possible_reversals,
            },
        }

    def _check_divergence(
        self,
        values: list[float],
        direction: str,
        window: int,
    ) -> Optional[dict[str, Any]]:
        """Check if metric is consistently moving in the wrong direction."""
        recent = values[-min(window, len(values)) :]
        if len(recent) < 3:
            return None

        wrong_direction_steps = 0
        for i in range(1, len(recent)):
            if direction == "minimize":
                if recent[i] > recent[i - 1]:
                    wrong_direction_steps += 1
            else:
                if recent[i] < recent[i - 1]:
                    wrong_direction_steps += 1

        total_steps = len(recent) - 1
        wrong_ratio = wrong_direction_steps / total_steps

        if wrong_ratio < 0.7:
            return None

        if direction == "minimize":
            total_change = recent[-1] - recent[0]
        else:
            total_change = recent[0] - recent[-1]

        if total_change <= 0:
            return None

        scale = max(abs(v) for v in recent) if any(v != 0 for v in recent) else 1.0
        normalized_change = total_change / scale

        confidence = min(1.0, wrong_ratio * min(1.0, normalized_change / 0.05))

        if wrong_ratio >= 0.9 and normalized_change > 0.1:
            severity_level = "critical"
        elif wrong_ratio >= 0.8:
            severity_level = "severe"
        elif wrong_ratio >= 0.7:
            severity_level = "moderate"
        else:
            severity_level = "minor"

        return {
            "failure_type": "divergence",
            "description": (
                f"Metric diverging: {wrong_direction_steps}/{total_steps} steps moving "
                f"in wrong direction over last {len(recent)} steps. "
                f"Total change: {total_change:.4f} ({normalized_change:.1%} of scale)."
            ),
            "severity_level": severity_level,
            "confidence": confidence,
            "evidence": {
                "wrong_ratio": wrong_ratio,
                "total_change": total_change,
                "normalized_change": normalized_change,
            },
        }
