# Core 1.11.0 local release candidate

Source baseline: `1da892a2f1a708b66a20e6d294eb3a9f6fe4f4dd` (PR 27).
This minor release exposes additive accounting metadata and intentionally
corrects communication inference. The changelog is the compatibility notice;
fewer legacy findings are not proof of repair, semantic accuracy, or complete
coverage. Wrapper reporting requires published `pisama>=0.7.0`.

## Local gates

- Full core test suite, Ruff, mypy, and the existing 60% coverage floor.
- Build wheel/sdist and validate both with `twine check`.
- Fresh environments install each exact artifact with public `pisama==0.7.0`;
  run dependency checks and installed API/CLI probes for supported, unsupported,
  mixed, duplicate-identity, and malformed inputs without source overrides.
- Record artifact SHA-256 hashes and candidate commit in the handoff.

No source push, merge, tag, release, publication, or provider mutation is part
of this task. Root owns consumer attestation and rollout authorization. Local
Python 3.11 checks do not replace the canonical 3.10–3.13 CI/release matrix.
