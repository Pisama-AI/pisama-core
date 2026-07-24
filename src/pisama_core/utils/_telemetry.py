"""Anonymous, opt-in usage telemetry for pisama-core.

Disabled by default. When explicitly enabled, fires once per process on
first DetectionOrchestrator instantiation.

What is sent (only when opted in):
    install_id    UUID4 generated locally and persisted at ~/.pisama/install_id
    sdk_version   pisama-core package version
    python        e.g. "3.12.3"
    os            platform.system() — Darwin / Linux / Windows
    os_release    platform.release() (truncated to 64 chars)
    runtime_env   one of: github_actions, gitlab_ci, circleci, jenkins, travis,
                  aws_lambda, vercel, fly, modal, kubernetes, docker, local
    event         "first_run" the first time, "session" thereafter

What is NOT sent: trace contents, detector outputs, file paths, environment
variables, hostnames, IPs (server discards on receipt), API keys, user names.

How to opt in (any one):
    export PISAMA_TELEMETRY=1
    python -c "import pisama_core; pisama_core.enable_telemetry()"

How to opt out (any one — overrides opt-in):
    export DO_NOT_TRACK=1
    touch ~/.pisama/telemetry_disabled
    python -c "import pisama_core; pisama_core.disable_telemetry()"
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import uuid
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

_ENDPOINT = os.environ.get(
    "PISAMA_TELEMETRY_ENDPOINT",
    "https://api.pisama.ai/api/v1/telemetry/install",
)
_CONFIG_DIR = Path.home() / ".pisama"
_INSTALL_ID_FILE = _CONFIG_DIR / "install_id"
_OPT_OUT_FILE = _CONFIG_DIR / "telemetry_disabled"
_OPT_IN_FILE = _CONFIG_DIR / "telemetry_enabled"

_FIRED_THIS_PROCESS = False
_FIRED_LOCK = threading.Lock()


def _is_enabled() -> bool:
    # Opt-out file is the kill switch of last resort — overrides everything.
    if _OPT_OUT_FILE.exists():
        return False
    dnt = os.environ.get("DO_NOT_TRACK")
    if dnt is not None and dnt.strip() not in ("", "0", "false", "no", "off"):
        return False
    val = os.environ.get("PISAMA_TELEMETRY", "")
    if val.strip().lower() in ("1", "true", "yes", "on"):
        return True
    if _OPT_IN_FILE.exists():
        return True
    return False


def _detect_runtime_env() -> str:
    env = os.environ
    if env.get("GITHUB_ACTIONS"):
        return "github_actions"
    if env.get("GITLAB_CI"):
        return "gitlab_ci"
    if env.get("CIRCLECI"):
        return "circleci"
    if env.get("JENKINS_URL"):
        return "jenkins"
    if env.get("TRAVIS"):
        return "travis"
    if env.get("AWS_LAMBDA_FUNCTION_NAME"):
        return "aws_lambda"
    if env.get("VERCEL"):
        return "vercel"
    if env.get("FLY_APP_NAME"):
        return "fly"
    if env.get("MODAL_TASK_ID"):
        return "modal"
    if env.get("KUBERNETES_SERVICE_HOST"):
        return "kubernetes"
    if Path("/.dockerenv").exists():
        return "docker"
    return "local"


def _load_or_create_install_id() -> tuple[str, bool]:
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return str(uuid.uuid4()), True

    if _INSTALL_ID_FILE.exists():
        try:
            existing = _INSTALL_ID_FILE.read_text().strip()
            if existing:
                return existing, False
        except OSError:
            pass

    new_id = str(uuid.uuid4())
    try:
        _INSTALL_ID_FILE.write_text(new_id)
    except OSError:
        pass
    return new_id, True


def _get_version() -> str:
    try:
        return _pkg_version("pisama-core")
    except PackageNotFoundError:
        return "0.0.0+unknown"
    except Exception:
        return "unknown"


def _build_payload(install_id: str, is_first_run: bool) -> dict[str, str]:
    py = sys.version_info
    return {
        "install_id": install_id,
        "sdk_version": _get_version(),
        "python": f"{py.major}.{py.minor}.{py.micro}",
        "os": platform.system() or "unknown",
        "os_release": (platform.release() or "")[:64],
        "runtime_env": _detect_runtime_env(),
        "event": "first_run" if is_first_run else "session",
    }


def _send(payload: dict[str, str]) -> None:
    # Stdlib-only: pisama-core's only runtime dependency is pydantic.
    try:
        import urllib.request

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _ENDPOINT,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"pisama-core/{payload['sdk_version']} ({payload['os']})",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            resp.read(0)
    except Exception:
        # Telemetry must never raise into the host process.
        pass


def _run() -> None:
    try:
        if not _is_enabled():
            return
        install_id, is_first_run = _load_or_create_install_id()
        _send(_build_payload(install_id, is_first_run))
    except Exception:
        pass


def record_first_run() -> None:
    """Fire telemetry once per process if opted in. Safe to call from constructors."""
    global _FIRED_THIS_PROCESS
    with _FIRED_LOCK:
        if _FIRED_THIS_PROCESS:
            return
        _FIRED_THIS_PROCESS = True
    if not _is_enabled():
        return
    try:
        threading.Thread(target=_run, name="pisama-telemetry", daemon=True).start()
    except Exception:
        pass


def enable_telemetry() -> None:
    """Programmatic opt-in. Persists for future processes."""
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _OPT_IN_FILE.touch()
    except OSError:
        pass
    os.environ["PISAMA_TELEMETRY"] = "1"


def disable_telemetry() -> None:
    """Programmatic opt-out. Persists and overrides any opt-in."""
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _OPT_OUT_FILE.touch()
    except OSError:
        pass
    os.environ["PISAMA_TELEMETRY"] = "0"
