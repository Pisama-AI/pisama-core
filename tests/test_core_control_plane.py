"""Control-plane contracts for configuration, registries, and orchestration."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pisama_core.config.loader import load_config, save_config
from pisama_core.config.models import PisamaConfig
from pisama_core.detection.detectors.loop import LoopDetector
from pisama_core.detection.orchestrator import DetectionOrchestrator
from pisama_core.detection.registry import DetectorRegistry
from pisama_core.traces.enums import Platform, SpanKind
from pisama_core.traces.models import Span, Trace, TraceMetadata
from pisama_core.utils.json_utils import safe_json_dumps, safe_json_loads
from pisama_core.utils.time_utils import now_utc


def _looping_trace(count: int = 10) -> Trace:
    trace = Trace(
        trace_id="trace-control-plane",
        metadata=TraceMetadata(platform=Platform.CLAUDE_CODE),
    )
    for _ in range(count):
        trace.create_span(name="Read", kind=SpanKind.TOOL)
    return trace


def test_config_round_trip_and_invalid_root_shape(tmp_path) -> None:
    path = tmp_path / "nested" / "config.json"
    config = PisamaConfig()
    config.detection.severity_threshold = 73
    config.healing.mode = "report"
    config.ignored_patterns = ["known-benign"]

    save_config(config, path)

    assert load_config(path).to_dict() == config.to_dict()
    path.write_text('["valid JSON", "invalid config shape"]')
    fallback = load_config(path)
    assert fallback == PisamaConfig()


def test_default_config_path_respects_user_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = PisamaConfig()
    config.injection.block_threshold = 91

    save_config(config)

    assert (tmp_path / ".pisama" / "config.json").exists()
    assert load_config().injection.block_threshold == 91


def test_json_and_time_utilities_cover_safe_edges() -> None:
    timestamp = datetime(2026, 7, 23, 12, 30, tzinfo=timezone.utc)
    encoded = safe_json_dumps({"at": timestamp}, sort_keys=True)
    assert encoded == '{"at": "2026-07-23T12:30:00+00:00"}'
    assert safe_json_loads(encoded) == {"at": "2026-07-23T12:30:00+00:00"}
    assert safe_json_loads("{invalid") is None
    with pytest.raises(TypeError, match="not JSON serializable"):
        safe_json_dumps({"unsupported": object()})
    assert now_utc().tzinfo is timezone.utc


def test_registry_lifecycle_uses_real_detector() -> None:
    registry = DetectorRegistry()
    detector = LoopDetector()

    registry.register(detector)
    assert detector.name in registry
    assert len(registry) == 1
    assert registry.enabled_count == 1
    assert registry.get_realtime_capable(Platform.CLAUDE_CODE) == [detector]
    assert "count=1" in repr(registry)

    assert registry.disable(detector.name)
    assert registry.get_enabled() == []
    assert not registry.disable("missing")
    assert registry.enable(detector.name)
    assert not registry.enable("missing")
    registry.disable_all()
    assert registry.enabled_count == 0
    registry.enable_all()
    assert registry.enabled_count == 1
    assert registry.unregister(detector.name) is detector
    assert registry.unregister(detector.name) is None


@pytest.mark.asyncio
async def test_explicit_empty_registry_never_falls_back_to_global_registry() -> None:
    registry = DetectorRegistry()
    orchestrator = DetectionOrchestrator(registry=registry)

    assert orchestrator.registry is registry
    result = await orchestrator.analyze(_looping_trace())
    realtime = await orchestrator.analyze_realtime(
        Span(span_id="empty-realtime"),
        {},
    )
    assert result.total_detectors_run == 0
    assert not result.has_issues
    assert result.to_dict()["detection_results"] == []
    assert realtime.to_dict()["severity"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [True, False])
async def test_orchestrator_applies_reporting_threshold_to_real_loop_detector(
    parallel: bool,
) -> None:
    registry = DetectorRegistry()
    registry.register(LoopDetector())
    trace = _looping_trace(count=5)

    reported = await DetectionOrchestrator(
        registry=registry,
        severity_threshold=40,
        parallel=parallel,
    ).analyze(trace)
    suppressed = await DetectionOrchestrator(
        registry=registry,
        severity_threshold=80,
        parallel=parallel,
    ).analyze(trace)

    assert reported.total_detectors_run == 1
    assert reported.issues_detected == 1
    assert reported.max_severity == 50
    assert reported.has_issues
    assert not reported.critical
    assert reported.get_issues()[0].detector_name == "loop"
    assert reported.get_by_severity(50) == reported.detection_results
    assert reported.get_recommendations()[0]["recommendation"]["fix_type"] == "break_loop"
    assert reported.to_dict()["platform"] == "claude_code"

    assert suppressed.total_detectors_run == 1
    assert suppressed.detection_results == []
    assert suppressed.issues_detected == 0
    assert suppressed.max_severity == 0
    assert suppressed.total_execution_time_ms >= 0


@pytest.mark.asyncio
async def test_realtime_orchestration_blocks_real_critical_loop() -> None:
    registry = DetectorRegistry()
    registry.register(LoopDetector())
    orchestrator = DetectionOrchestrator(registry=registry, block_threshold=60)
    current = Span(
        span_id="current",
        trace_id="trace-realtime",
        name="Read",
        kind=SpanKind.TOOL,
        platform=Platform.CLAUDE_CODE,
    )
    recent = [
        Span(
            span_id=f"recent-{index}",
            trace_id="trace-realtime",
            name="Read",
            kind=SpanKind.TOOL,
            platform=Platform.CLAUDE_CODE,
        )
        for index in range(4)
    ]

    result = await orchestrator.analyze_realtime(current, {"recent_spans": recent})

    assert result.should_block
    assert result.severity == 60
    assert result.block_reason == result.issues[0]
    assert result.recommendations[0]["fix_type"] == "break_loop"
    assert result.to_dict()["span_id"] == "current"
    status = orchestrator.get_detector_status()
    assert status == {
        "total": 1,
        "enabled": 1,
        "detectors": [
            {
                "name": "loop",
                "enabled": True,
                "realtime": True,
                "platforms": ["all"],
            }
        ],
    }


@pytest.mark.parametrize(
    ("variable", "expected"),
    [
        ("GITHUB_ACTIONS", "github_actions"),
        ("GITLAB_CI", "gitlab_ci"),
        ("CIRCLECI", "circleci"),
        ("JENKINS_URL", "jenkins"),
        ("TRAVIS", "travis"),
        ("AWS_LAMBDA_FUNCTION_NAME", "aws_lambda"),
        ("VERCEL", "vercel"),
        ("FLY_APP_NAME", "fly"),
        ("MODAL_TASK_ID", "modal"),
        ("KUBERNETES_SERVICE_HOST", "kubernetes"),
    ],
)
def test_telemetry_runtime_environment_classification(
    variable: str,
    expected: str,
    monkeypatch,
) -> None:
    from pisama_core.utils import _telemetry

    for candidate in (
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "CIRCLECI",
        "JENKINS_URL",
        "TRAVIS",
        "AWS_LAMBDA_FUNCTION_NAME",
        "VERCEL",
        "FLY_APP_NAME",
        "MODAL_TASK_ID",
        "KUBERNETES_SERVICE_HOST",
    ):
        monkeypatch.delenv(candidate, raising=False)
    monkeypatch.setenv(variable, "present")

    assert _telemetry._detect_runtime_env() == expected
