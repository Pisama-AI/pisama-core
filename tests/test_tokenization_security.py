"""Security and persistence contracts for local PII tokenization.

These tests use real AES-256-GCM, SQLite, filesystem permissions, and parser
implementations. No keychain service or database is simulated.
"""

from __future__ import annotations

import hashlib
import json
import stat

import pytest

from pisama_core.tokenization import (
    FileBackend,
    KeychainError,
    KeychainManager,
    KeychainUnavailableError,
    PIIDetector,
    PIIPattern,
    TokenGenerator,
    Tokenizer,
    TokenParser,
    TokenVault,
    derive_key_from_password,
)

TEST_KEY = bytes(range(32))
SECOND_KEY = bytes(reversed(range(32)))


def _tokenizer_with_real_vault(tmp_path, *, fail_open: bool = False) -> Tokenizer:
    tokenizer = Tokenizer(
        session_id="session-a1b2",
        vault_path=tmp_path / "vault.db",
        fail_open=fail_open,
    )
    tokenizer._vault = TokenVault(tmp_path / "vault.db")
    tokenizer._vault.initialize()
    tokenizer._encryption_key = TEST_KEY
    return tokenizer


def test_pii_detector_configuration_recursion_and_overlap() -> None:
    detector = PIIDetector(exclusions=["test@example.com", "@internal.example"])
    text = "Contact alice@example.org or 202-555-0100. Use 4111 1111 1111 1111 from 192.0.2.25."

    matches = detector.detect(text)
    assert {match.pii_type for match in matches} == {"EMAIL", "PHONE", "CC", "IP"}
    assert "alice@example.org" in repr(matches[0])
    assert detector.contains_pii(text)
    assert not detector.contains_pii("test@example.com")
    assert not detector.contains_pii("alice@internal.example")

    detector.add_exclusion("alice@example.org")
    assert "alice@example.org" not in {match.value for match in detector.detect(text)}
    assert detector.remove_exclusion("alice@example.org")
    assert not detector.remove_exclusion("missing")

    custom = PIIPattern(
        name="INCIDENT",
        pii_type="INCIDENT",
        pattern=r"\bINC-\d{6}\b",
        description="Incident ticket",
    )
    detector.add_pattern(custom)
    assert custom.compiled.pattern == r"\bINC-\d{6}\b"
    assert detector.disable_pattern("INCIDENT")
    assert not detector.disable_pattern("missing")
    assert not detector.detect("INC-123456")
    assert detector.enable_pattern("INCIDENT")
    assert not detector.enable_pattern("missing")
    assert detector.detect("INC-123456")[0].pii_type == "INCIDENT"
    assert detector.remove_pattern("INCIDENT")
    assert not detector.remove_pattern("INCIDENT")

    detector.add_sensitive_field("private_note")
    nested = detector.detect_in_dict(
        {
            "private_note": "opaque value",
            "customer": {"email": "alice@example.org"},
            "events": ["carol@example.org", {"address": "bob@example.org"}],
        }
    )
    assert [path for path, _ in nested] == [
        "private_note",
        "customer.email",
        "events[0]",
        "events[1].address",
    ]
    assert detector.is_sensitive_field("PRIVATE_NOTE")
    assert detector.patterns is not detector.patterns
    assert detector.exclusions == {"test@example.com", "@internal.example"}
    assert detector.sensitive_fields
    assert detector.get_pattern_stats()["EMAIL"]["enabled"]
    detector.disable_pattern("EMAIL")
    assert not detector.contains_pii("alice@example.org")
    detector.enable_pattern("EMAIL")

    overlap = PIIDetector(
        patterns=[
            PIIPattern("SHORT", "SHORT", r"abc"),
            PIIPattern("LONG", "LONG", r"abcdef"),
        ],
        exclusions=[],
        sensitive_fields=[],
    )
    deduplicated = overlap.detect("abcdef")
    assert [(match.pii_type, match.value) for match in deduplicated] == [("LONG", "abcdef")]
    assert overlap.detect("clean") == []
    assert overlap.exclusions == set()
    assert overlap.sensitive_fields == set()
    later_longer = PIIDetector(
        patterns=[
            PIIPattern("EARLY", "EARLY", r"bc"),
            PIIPattern("LATER_LONG", "LATER_LONG", r"cdefgh"),
        ],
        exclusions=[],
        sensitive_fields=[],
    )
    assert later_longer.detect("abcdefgh")[0].pii_type == "LATER_LONG"


def test_token_generator_and_parser_enforce_exact_tokens() -> None:
    generator = TokenGenerator(session_id="A1-B2-session")
    first = generator.generate("EMAIL", "alice@example.org")
    assert first == generator.generate("EMAIL", "alice@example.org")
    forced = generator.generate("EMAIL", "alice@example.org", force_new=True)
    assert first != forced
    assert generator.get_token_count() == 2
    assert generator.get_token_info(first)["original_value"] == "alice@example.org"
    assert generator.get_token_info("[EMAIL:none:00000000]") is None
    assert len(generator.get_all_tokens()) == 2

    parser = TokenParser()
    parts = parser.parse(first)
    assert parts is not None
    assert parts["pii_type"] == "EMAIL"
    assert parts["session_prefix"] == "a1b2"
    assert parser.get_session_prefix(first) == "a1b2"
    assert parser.get_pii_type(first) == "EMAIL"
    assert parser.is_valid_token(first)
    assert not parser.is_valid_token(first + "trailing")
    assert parser.parse(first + "trailing") is None
    assert parser.extract_tokens(f"x {first} y {forced}") == [first, forced]

    short_session = TokenGenerator(session_id="x")
    assert len(short_session._session_prefix) == 4
    generator.clear_cache()
    assert generator.get_token_count() == 0
    exhausted = TokenGenerator(session_id="collision")
    exhausted._collision_retries = 0
    with pytest.raises(RuntimeError, match="Failed to generate unique token"):
        exhausted.generate("EMAIL", "alice@example.org")


def test_file_backend_and_manager_use_owner_only_real_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PISAMA_ALLOW_INSECURE_FILE_KEYCHAIN", "1")
    backend = FileBackend(tmp_path / "keys" / "master.key")
    manager = KeychainManager(allow_file_fallback=False)
    manager._backends = [backend]

    assert manager.backend_name == "file-fallback"
    invalid = manager.store_key(b"too-short")
    assert not invalid.success
    assert invalid.backend == "validation"

    assert manager.store_key(TEST_KEY).success
    assert manager.get_key() == TEST_KEY
    assert manager.get_or_create_key() == TEST_KEY
    assert manager.key_exists()
    assert stat.S_IMODE(backend.key_path.stat().st_mode) == 0o600
    assert manager.get_status() == {
        "available": True,
        "backend": "file-fallback",
        "key_exists": True,
        "is_secure": False,
    }

    old_key, new_key = manager.rotate_key()
    assert old_key == TEST_KEY
    assert new_key != old_key
    assert len(new_key) == 32
    assert manager.get_key() == new_key
    assert manager.delete_key().success
    assert manager.delete_key().success
    assert not manager.key_exists()
    created = manager.get_or_create_key()
    assert len(created) == 32
    manager.delete_key()


def test_file_backend_requires_explicit_insecure_opt_in(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PISAMA_ALLOW_INSECURE_FILE_KEYCHAIN", raising=False)
    backend = FileBackend(tmp_path / "master.key")

    result = backend.store_key(TEST_KEY)

    assert not result.success
    assert "PISAMA_ALLOW_INSECURE_FILE_KEYCHAIN=1" in result.message
    assert not backend.key_path.exists()

    backend.key_path.write_text("not-valid-base64")
    assert backend.get_key() is None


def test_file_backend_reports_real_filesystem_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PISAMA_ALLOW_INSECURE_FILE_KEYCHAIN", "1")
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("file")
    backend = FileBackend(parent_file / "master.key")

    result = backend.store_key(TEST_KEY)

    assert not result.success
    assert "Failed to store key" in result.message


def test_unavailable_keychain_and_rotation_fail_closed(tmp_path) -> None:
    manager = KeychainManager(allow_file_fallback=False)
    manager._backends = []

    with pytest.raises(KeychainUnavailableError):
        manager.get_key()
    assert manager.get_status() == {
        "available": False,
        "backend": None,
        "key_exists": False,
        "is_secure": False,
    }

    manager._backends = [FileBackend(tmp_path / "missing.key")]
    manager._active_backend = manager._backends[0]
    with pytest.raises(KeychainError, match="No existing key"):
        manager.rotate_key()


def test_rotation_preserves_old_key_when_real_backend_rejects_new_key(
    tmp_path,
    monkeypatch,
) -> None:
    backend = FileBackend(tmp_path / "master.key")
    manager = KeychainManager(allow_file_fallback=False)
    manager._backends = [backend]
    monkeypatch.setenv("PISAMA_ALLOW_INSECURE_FILE_KEYCHAIN", "1")
    assert manager.store_key(TEST_KEY).success
    monkeypatch.delenv("PISAMA_ALLOW_INSECURE_FILE_KEYCHAIN")

    with pytest.raises(KeychainError, match="Failed to store rotated key"):
        manager.rotate_key()
    assert manager.get_key() == TEST_KEY


def test_vault_encrypts_queries_erases_and_detects_tampering(tmp_path) -> None:
    vault_path = tmp_path / "private" / "vault.db"
    vault = TokenVault(vault_path)
    value = "alice@example.org"
    value_hash = hashlib.sha256(value.encode()).hexdigest()
    token_a = "[EMAIL:a1b2:00000001]"
    token_b = "[EMAIL:a1b2:00000002]"

    assert vault.store("EMAIL", token_a, value, "session-a", TEST_KEY)
    assert vault.store("EMAIL", token_b, value, "session-b", TEST_KEY)
    assert not vault.store("EMAIL", token_a, value, "session-a", TEST_KEY)
    assert stat.S_IMODE(vault_path.stat().st_mode) == 0o600
    assert value.encode() not in vault_path.read_bytes()

    assert vault.retrieve(token_a, TEST_KEY) == value
    assert vault.retrieve("missing", TEST_KEY) is None
    assert vault.retrieve_batch([token_a, "missing"], TEST_KEY) == {
        token_a: value,
        "missing": None,
    }
    with pytest.raises(ValueError, match="Decryption failed"):
        vault.retrieve(token_a, SECOND_KEY)

    record = vault.get_token_info(token_a)
    assert record is not None
    assert record.pii_type == "EMAIL"
    assert record.session_id == "session-a"
    assert record.value_hash == value_hash
    assert vault.get_token_info("missing") is None
    assert set(vault.find_by_value_hash(value_hash)) == {token_a, token_b}
    assert vault.find_by_value_hash(value_hash, "session-a") == [token_a]
    assert vault.list_session_tokens("session-b") == [token_b]
    assert vault.get_stats()["by_type"] == {"EMAIL": 2}

    connection = vault._get_connection()
    connection.execute(
        "UPDATE tokens SET encrypted_value = zeroblob(length(encrypted_value)) WHERE token = ?",
        (token_a,),
    )
    connection.commit()
    with pytest.raises(ValueError, match="Decryption failed"):
        vault.retrieve(token_a, TEST_KEY)

    assert vault.delete_token(token_a)
    assert not vault.delete_token(token_a)
    assert vault.delete_session("session-b") == 1
    assert vault.delete_by_value_hash(value_hash) == 0
    vault.vacuum()
    vault.close()


def test_vault_context_manager_and_password_derivation(tmp_path) -> None:
    salt = bytes(range(16))
    first_key, returned_salt = derive_key_from_password("correct horse battery staple", salt)
    second_key, _ = derive_key_from_password("correct horse battery staple", salt)
    random_salt_key, random_salt = derive_key_from_password("correct horse battery staple")

    assert first_key == second_key
    assert returned_salt == salt
    assert len(first_key) == 32
    assert len(random_salt) == 16
    assert random_salt_key != first_key

    with TokenVault(tmp_path / "context.db") as vault:
        assert vault.get_stats()["total_tokens"] == 0
    assert vault._conn is None


def test_tokenizer_round_trip_audit_and_recursive_fields(tmp_path) -> None:
    tokenizer = _tokenizer_with_real_vault(tmp_path)
    source = "Email alice@example.org, backup alice@example.org"

    tokenized = tokenizer.tokenize_string(source)

    tokens = TokenParser().extract_tokens(tokenized)
    assert len(tokens) == 2
    assert tokens[0] == tokens[1]
    assert tokenizer.contains_pii(source)
    assert not tokenizer.contains_pii(tokenized)
    assert (
        tokenizer.detokenize_string(
            tokenized,
            reason="Investigate incident",
            principal="security-reviewer",
            ticket="INC-123456",
        )
        == source
    )

    audit_path = tmp_path / "audit_log.jsonl"
    audit = json.loads(audit_path.read_text())
    assert audit["principal"] == "security-reviewer"
    assert audit["reason"] == "Investigate incident"
    assert audit["tokens_count"] == 2
    assert "alice@example.org" not in audit_path.read_text()
    assert stat.S_IMODE(audit_path.stat().st_mode) == 0o600

    nested = {
        "input": "alice@example.org",
        "untouched": "bob@example.org",
        "password": "correct horse battery staple",
        "items": [{"input": "carol@example.org"}, 42],
    }
    protected = tokenizer.tokenize_dict(nested, fields_to_tokenize=["input", "password"])
    assert protected["untouched"] == "bob@example.org"
    assert protected["password"].startswith("[SENSITIVE_FIELD:")
    assert protected["items"][1] == 42
    restored = tokenizer.detokenize_dict(
        protected,
        reason="Validate recovery",
        principal="security-reviewer",
    )
    assert restored == nested
    assert tokenizer.get_stats().total_tokenized >= 5
    assert tokenizer.get_stats().fields_tokenized == 1
    assert tokenizer.get_vault_stats()["total_tokens"] == 3

    tokenizer.close()
    assert tokenizer._vault is None
    assert tokenizer._encryption_key is None


def test_tokenizer_initializes_real_file_keychain_and_vault(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PISAMA_ALLOW_INSECURE_FILE_KEYCHAIN", "1")
    backend = FileBackend(tmp_path / "master.key")
    manager = KeychainManager(allow_file_fallback=False)
    manager._backends = [backend]
    tokenizer = Tokenizer(
        "file-keychain-session",
        vault_path=tmp_path / "vault.db",
        fail_open=False,
    )
    tokenizer._keychain = manager
    tokenizer.add_pattern(
        PIIPattern("EMPLOYEE", "EMPLOYEE", r"\bEMP-\d{6}\b", "Employee identifier")
    )
    tokenizer.add_exclusion("EMP-000000")
    tokenizer.add_sensitive_field("private_note")

    protected = tokenizer.tokenize_dict(
        {
            "employee": "EMP-123456",
            "excluded": "EMP-000000",
            "private_note": "internal-only value",
        }
    )

    assert protected["employee"].startswith("[EMPLOYEE:")
    assert protected["excluded"] == "EMP-000000"
    assert protected["private_note"].startswith("[SENSITIVE_FIELD:")
    assert backend.key_path.exists()
    assert (tmp_path / "vault.db").exists()
    assert tokenizer.get_vault_stats()["total_tokens"] == 2
    tokenizer.close()


def test_tokenizer_disabled_failure_modes_and_detokenization_guards(tmp_path) -> None:
    disabled = Tokenizer("disabled", vault_path=tmp_path / "disabled.db", enabled=False)
    payload = {"email": "alice@example.org"}
    assert disabled.tokenize_string(payload["email"]) == payload["email"]
    assert disabled.tokenize_dict(payload) is payload

    unavailable = KeychainManager(allow_file_fallback=False)
    unavailable._backends = []
    fail_open = Tokenizer("open", vault_path=tmp_path / "open.db", fail_open=True)
    fail_open._keychain = unavailable
    assert fail_open.tokenize_string(payload["email"]) == payload["email"]
    assert fail_open.get_stats().errors == 1
    assert fail_open.get_vault_stats() is None

    fail_closed = Tokenizer("closed", vault_path=tmp_path / "closed.db", fail_open=False)
    fail_closed._keychain = unavailable
    with pytest.raises(KeychainUnavailableError):
        fail_closed.tokenize_string(payload["email"])
    assert fail_closed.get_stats().errors == 1

    guarded = _tokenizer_with_real_vault(tmp_path / "guarded")
    with pytest.raises(ValueError, match="reason"):
        guarded.detokenize_string("plain", reason="", principal="reviewer")
    with pytest.raises(ValueError, match="principal"):
        guarded.detokenize_string("plain", reason="review", principal=" ")
    assert guarded.detokenize_string("plain", reason="review", principal="reviewer") == "plain"
    guarded.close()


def test_detokenization_refuses_to_reveal_when_audit_cannot_be_written(tmp_path) -> None:
    tokenizer = _tokenizer_with_real_vault(tmp_path / "real-vault")
    tokenized = tokenizer.tokenize_string("alice@example.org")
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file")
    tokenizer._vault_path = blocker / "vault.db"

    with pytest.raises(FileExistsError):
        tokenizer.detokenize_string(
            tokenized,
            reason="Security review",
            principal="security-reviewer",
        )
    tokenizer.close()
