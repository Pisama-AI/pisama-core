"""Canonical destructive/risky verb taxonomy shared across safety detectors.

Lives in pisama-core so both the OSS detectors and the backend can use the
same source of truth without crossing package boundaries. The taxonomy is
organized by *intent*, not by *risk level*. Detectors decide which intents
matter for their context (e.g., cowork_safety only cares about DELETE intents
on cloud-synced paths; approval_bypass cares about DELETE + DEPLOY + EXECUTE).
"""

import re
from typing import Iterable

# ── Canonical verb categories ────────────────────────────────────────────────

# Irreversible deletion of data, files, records, etc.
DELETE_VERBS: frozenset = frozenset(
    {
        "delete",
        "remove",
        "rm",
        "rm -rf",
        "rmdir",
        "unlink",
        "drop",
        "destroy",
        "purge",
        "truncate",
        "wipe",
        "erase",
        "overwrite",
    }
)

# Modification or in-place writes (less destructive than DELETE, still mutates)
WRITE_VERBS: frozenset = frozenset(
    {
        "write",
        "create",
        "save",
        "modify",
        "update",
        "patch",
        "put",
        "insert",
        "append",
        "merge",
    }
)

# External communication / network side effects
SEND_VERBS: frozenset = frozenset(
    {
        "send",
        "email",
        "notify",
        "publish",
        "broadcast",
        "post",
    }
)

# Production deployment
DEPLOY_VERBS: frozenset = frozenset(
    {
        "deploy",
        "push",
        "release",
        "ship",
        "promote",
    }
)

# Arbitrary code/command execution
EXECUTE_VERBS: frozenset = frozenset(
    {
        "execute",
        "run",
        "invoke",
        "trigger",
        "fire",
        "exec",
        "eval",
        "shell",
        "system",
        "subprocess",
        "command",
        "run_code",
    }
)

# Permission/privilege changes
PERMISSION_VERBS: frozenset = frozenset(
    {
        "grant",
        "revoke",
        "chmod",
        "chown",
        "permission",
        "privilege",
        "escalate",
        "elevate",
        "role",
    }
)

# Admin / moderation actions
ADMIN_VERBS: frozenset = frozenset(
    {
        "ban",
        "suspend",
        "block",
        "terminate",
        "kill",
        "shutdown",
        "rollback",
    }
)

# Bulk data operations (export/dump/migrate)
BULK_DATA_VERBS: frozenset = frozenset(
    {
        "bulk",
        "export",
        "dump",
        "migrate",
    }
)

# Financial actions
FINANCIAL_VERBS: frozenset = frozenset(
    {
        "transfer",
        "pay",
        "charge",
        "refund",
    }
)


# ── Per-detector compositions ────────────────────────────────────────────────

COWORK_DESTRUCTIVE_VERBS: frozenset = DELETE_VERBS

APPROVAL_HIGH_RISK_VERBS: frozenset = (
    DELETE_VERBS
    | DEPLOY_VERBS
    | EXECUTE_VERBS
    | ADMIN_VERBS
    | FINANCIAL_VERBS
    | frozenset({"format", "revoke"})
)

EXPLORATION_DANGEROUS_VERBS: frozenset = (
    DELETE_VERBS
    | WRITE_VERBS
    | SEND_VERBS
    | DEPLOY_VERBS
    | EXECUTE_VERBS
    | PERMISSION_VERBS
    | frozenset({"transfer", "move", "migrate", "commit"})
)

OPENCLAW_RISKY_KEYWORDS: dict = {
    "admin_actions": ADMIN_VERBS | frozenset({"delete", "revoke"}),
    "permission_ops": PERMISSION_VERBS,
    "data_operations": BULK_DATA_VERBS | frozenset({"truncate", "drop"}),
    "credential_ops": frozenset(
        {
            "password",
            "reset_password",
            "credential",
            "token",
            "secret",
        }
    ),
    "system_commands": EXECUTE_VERBS - frozenset({"run", "invoke", "trigger", "fire"}),
}


# ── Pattern compilation helper ───────────────────────────────────────────────

# Snake_case-aware boundaries: standard \b doesn't match between letters and
# underscores, so r"\bdelete\b" misses "delete_file". These boundaries treat
# `_` as a word separator.
_WB_START = r"(?:^|[_\W])"
_WB_END = r"(?:[_\W]|$)"


def make_verb_pattern(verbs: Iterable[str]) -> re.Pattern[str]:
    """Compile a single regex matching any verb in *verbs* with snake_case boundaries.

    Sorts verbs longest-first so multi-word verbs like "rm -rf" match before "rm".
    """
    sorted_verbs = sorted({v for v in verbs}, key=len, reverse=True)
    alts = "|".join(re.escape(v) for v in sorted_verbs)
    return re.compile(f"{_WB_START}(?:{alts}){_WB_END}", re.IGNORECASE)
