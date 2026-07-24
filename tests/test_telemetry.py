"""Tests for the anonymous opt-in install-telemetry module.

The send path is monkey-patched in every test — these never make a real
network request.
"""

from __future__ import annotations

import importlib
import threading

import pytest


@pytest.fixture
def fresh_telemetry(tmp_path, monkeypatch):
    """Reload the module with HOME pointed at a temp dir and an in-memory sink."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PISAMA_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)

    from pisama_core.utils import _telemetry as mod

    importlib.reload(mod)

    sent: list[dict] = []
    done = threading.Event()

    def fake_send(payload):
        sent.append(payload)
        done.set()

    mod._send = fake_send  # type: ignore[assignment]
    yield mod, sent, done


def test_default_no_telemetry_without_opt_in(fresh_telemetry):
    mod, sent, _ = fresh_telemetry
    mod.record_first_run()
    assert sent == []


def test_env_var_zero_keeps_disabled(fresh_telemetry, monkeypatch):
    mod, sent, _ = fresh_telemetry
    monkeypatch.setenv("PISAMA_TELEMETRY", "0")
    mod.record_first_run()
    assert sent == []


def test_do_not_track_overrides_opt_in(fresh_telemetry, monkeypatch):
    mod, sent, _ = fresh_telemetry
    monkeypatch.setenv("PISAMA_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    mod.record_first_run()
    assert sent == []


def test_opt_out_file_overrides_opt_in_env_var(fresh_telemetry, monkeypatch):
    mod, sent, _ = fresh_telemetry
    monkeypatch.setenv("PISAMA_TELEMETRY", "1")
    mod._CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    mod._OPT_OUT_FILE.touch()
    mod.record_first_run()
    assert sent == []


def test_disable_telemetry_persists_opt_out(fresh_telemetry, monkeypatch):
    mod, sent, _ = fresh_telemetry
    monkeypatch.setenv("PISAMA_TELEMETRY", "1")
    mod.disable_telemetry()
    mod.record_first_run()
    assert sent == []
    assert mod._OPT_OUT_FILE.exists()


def test_enable_telemetry_persists_opt_in(fresh_telemetry):
    mod, sent, done = fresh_telemetry
    mod.enable_telemetry()
    mod.record_first_run()
    assert done.wait(timeout=2.0)
    assert len(sent) == 1
    assert mod._OPT_IN_FILE.exists()


def test_env_var_opt_in_sends_payload(fresh_telemetry, monkeypatch):
    mod, sent, done = fresh_telemetry
    monkeypatch.setenv("PISAMA_TELEMETRY", "1")
    mod.record_first_run()
    assert done.wait(timeout=2.0)
    assert len(sent) == 1
    assert sent[0]["event"] == "first_run"
    assert sent[0]["install_id"]
    assert mod._INSTALL_ID_FILE.exists()


def test_second_process_run_is_session_event(fresh_telemetry, monkeypatch):
    mod, sent, done = fresh_telemetry
    monkeypatch.setenv("PISAMA_TELEMETRY", "1")
    mod._CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    mod._INSTALL_ID_FILE.write_text("abc-existing")
    mod.record_first_run()
    assert done.wait(timeout=2.0)
    assert sent[0]["event"] == "session"
    assert sent[0]["install_id"] == "abc-existing"


def test_only_fires_once_per_process(fresh_telemetry, monkeypatch):
    mod, sent, done = fresh_telemetry
    monkeypatch.setenv("PISAMA_TELEMETRY", "1")
    mod.record_first_run()
    assert done.wait(timeout=2.0)
    mod.record_first_run()
    mod.record_first_run()
    assert len(sent) == 1


def test_send_exception_does_not_propagate(fresh_telemetry, monkeypatch):
    mod, _sent, _ = fresh_telemetry
    monkeypatch.setenv("PISAMA_TELEMETRY", "1")

    def boom(_payload):
        raise RuntimeError("network down")

    mod._send = boom  # type: ignore[assignment]
    mod.record_first_run()  # must not raise


def test_payload_has_required_fields(fresh_telemetry, monkeypatch):
    mod, sent, done = fresh_telemetry
    monkeypatch.setenv("PISAMA_TELEMETRY", "1")
    mod.record_first_run()
    assert done.wait(timeout=2.0)
    p = sent[0]
    for key in ("install_id", "sdk_version", "python", "os", "runtime_env", "event"):
        assert key in p, f"missing field: {key}"


def test_runtime_env_detects_github_actions(fresh_telemetry, monkeypatch):
    mod, sent, done = fresh_telemetry
    monkeypatch.setenv("PISAMA_TELEMETRY", "1")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    mod.record_first_run()
    assert done.wait(timeout=2.0)
    assert sent[0]["runtime_env"] == "github_actions"


def test_orchestrator_init_does_not_fire_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PISAMA_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)

    from pisama_core.utils import _telemetry as tel

    importlib.reload(tel)

    sent: list[dict] = []
    tel._send = lambda payload: sent.append(payload)  # type: ignore[assignment]

    from pisama_core.detection import orchestrator as orch_mod

    importlib.reload(orch_mod)
    orch_mod._record_first_run = tel.record_first_run  # type: ignore[attr-defined]

    orch_mod.DetectionOrchestrator()
    threading.Event().wait(0.2)
    assert sent == [], "default-off: orchestrator should not fire telemetry"


def test_orchestrator_init_fires_telemetry_when_opted_in(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PISAMA_TELEMETRY", "1")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)

    from pisama_core.utils import _telemetry as tel

    importlib.reload(tel)

    sent: list[dict] = []
    tel._send = lambda payload: sent.append(payload)  # type: ignore[assignment]

    from pisama_core.detection import orchestrator as orch_mod

    importlib.reload(orch_mod)
    orch_mod._record_first_run = tel.record_first_run  # type: ignore[attr-defined]

    orch_mod.DetectionOrchestrator()
    for _ in range(20):
        if sent:
            break
        threading.Event().wait(0.05)
    assert sent, "with PISAMA_TELEMETRY=1, orchestrator should fire telemetry"
