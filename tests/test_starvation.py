"""Tests for TaskStarvationDetector."""

import asyncio

from pisama_core import Trace, SpanKind
from pisama_core.detection.detectors.starvation import TaskStarvationDetector


def _run(trace: Trace):
    return asyncio.run(TaskStarvationDetector().detect(trace))


def _plan(trace: Trace, tasks: list[str]) -> None:
    plan_text = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tasks))
    trace.create_span(name="planner", kind=SpanKind.AGENT, output_data={"output": plan_text})


def _execute(trace: Trace, task_name: str) -> None:
    trace.create_span(name=task_name, kind=SpanKind.TASK, output_data={"output": f"Completed: {task_name}"})


class TestTaskStarvation:
    def test_stuck_loop_on_subset_of_plan_is_detected(self):
        # Regression: 5 planned, only 2 executed but repeated 3x each (6 raw
        # execution spans). The negative-signal gate used to compare raw
        # execution-span count (6) against planned count (5) and suppress
        # this as "executed enough" -- it must compare DISTINCT executed
        # tasks (2) instead, since repeating 2 tasks 3 times is the stuck-
        # in-a-loop failure this detector exists to catch, not evidence of
        # completeness.
        t = Trace()
        _plan(t, ["admit", "triage", "vitals", "imaging", "discharge"])
        for _ in range(3):
            _execute(t, "admit")
            _execute(t, "triage")
        r = _run(t)
        assert r.detected

    def test_all_planned_tasks_executed_is_not_detected(self):
        t = Trace()
        _plan(t, ["admit", "triage", "vitals", "imaging", "discharge"])
        for task in ("admit", "triage", "vitals", "imaging", "discharge"):
            _execute(t, task)
        r = _run(t)
        assert not r.detected

    def test_paraphrase_mismatch_is_not_detected(self):
        # More DISTINCT executed tasks than planned, just described
        # differently -- the negative-signal gate's original purpose. Must
        # still hold after the distinct-count fix.
        t = Trace()
        _plan(t, ["send email campaign", "review budget"])
        for task in ("newsletter blast dispatched", "quarterly budget audit", "follow-up scheduled"):
            _execute(t, task)
        r = _run(t)
        assert not r.detected

    def test_too_few_planned_tasks_is_not_detected(self):
        t = Trace()
        _plan(t, ["only one task"])
        r = _run(t)
        assert not r.detected
