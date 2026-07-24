# Changelog

All notable changes to `pisama-core` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Correct standalone repository metadata and ship the `py.typed` marker.
- Add CodeQL, dependency review, Dependabot, and a coverage regression gate.
- Restore PyPI publication attestations and pin release workflow actions.

## [1.8.1] - 2026-07-23

### Fixed

- Parse the standard ISO 8601 `Z` UTC suffix consistently on Python 3.10
  through 3.13.

## [1.8.0] - 2026-07-23

### Added

- A typed `DiagnosisRecord` contract that preserves additive fields during
  serialization and gives SDK consumers structured causal evidence,
  interventions, coverage, and provenance.
- Omnigent event-stream conversion to ATIF trajectories, plus reusable
  conversation trace ingestion and round-trip serialization.
- An escalation-loop detector for repeated, ineffective handoffs between
  agents.

### Changed

- Improved citation, entity-confusion, MCP-protocol, and parallel-consistency
  detection across realistic multi-agent traces.
- Replaced deprecated naive UTC timestamps with timezone-aware UTC values.
- Updated package metadata to point to the canonical monorepo and declare
  Python 3.13 support.

### Fixed

- Hardened fix injection and diagnosis serialization against malformed or
  forward-compatible payloads.

## [1.7.3] — 2026-04-30

### Fixed

- **`cost.py` was missing the `Span` top-level import,** causing
  `from pisama_core.detection import detectors` to raise
  `NameError: name 'Span' is not defined` on Python 3.10–3.13. Hidden
  on Python 3.14 by lazy annotation evaluation (PEP 649). The bug
  shipped in every release since the package was first cut on
  2026-01-02 and silently broke the documented quickstart for every
  user not on Python 3.14. A late `# noqa: E402` re-import at the
  bottom of `cost.py` was the workaround attempt and has been removed.

- **The README quickstart returned zero detectors** because
  `import pisama_core` did not register the built-in detectors. The
  user was implicitly required to add
  `from pisama_core.detection import detectors` themselves. Now
  `pisama_core/__init__.py` imports the detectors subpackage so that
  the documented quickstart fires detectors as advertised.

### Notes

- The audit ran each of the 31 detector modules through `importlib.import_module`
  on Python 3.13. After the fix, all 31 import cleanly. The published
  quickstart now reports 30 detectors running and the loop detector
  firing with severity 100 on the documented 8x Read trace.

## [1.7.2] — 2026-04-30

### Changed

- README: added a "Using Pisama?" feedback CTA after Quick Start and a
  Design Partner Program section. The README ships in the package
  metadata, so a release is required for these to appear on the PyPI
  project page. No code changes.

## [1.7.1] — 2026-04-29

### Changed

- **Telemetry is now opt-in.** 1.7.0 (released earlier today) had
  opt-out telemetry. Reversed before any meaningful uptake to avoid
  the optics of a multi-agent failure-detection library phoning home
  from customer production servers, particularly during enterprise
  security review. By default, no telemetry is sent.

  To opt in (any one):

  ```bash
  export PISAMA_TELEMETRY=1
  ```

  Or programmatically: `pisama_core.enable_telemetry()`.

  The opt-out paths from 1.7.0 (`DO_NOT_TRACK=1`,
  `~/.pisama/telemetry_disabled`, `pisama_core.disable_telemetry()`)
  remain and override any opt-in.

## [1.7.0] — 2026-04-29

### Added

- **Anonymous install telemetry.** The first time you instantiate
  `DetectionOrchestrator` in a process, `pisama-core` sends a single
  install ping containing: a locally-generated UUID install id, the SDK
  version, Python version, OS family + release, and a coarse
  `runtime_env` tag (e.g. `github_actions`, `aws_lambda`, `local`). No
  trace contents, detector outputs, file paths, environment variables,
  hostnames, IPs, or API keys are sent. A one-time consent banner is
  printed to stderr on first use. See
  [Telemetry](https://github.com/tn-pisama/pisama-core#telemetry) for
  the full schema and disable instructions.

  Disable any of the following ways:

  ```bash
  export PISAMA_TELEMETRY=0          # also: false / no / off / empty
  export DO_NOT_TRACK=1              # honored too
  touch ~/.pisama/telemetry_disabled
  ```

  Or programmatically: `pisama_core.disable_telemetry()`.

## [1.6.3] — 2026-04-19

Sprint 10 Phase VV — `citation` detector recall-lift.

### Changed

- **citation** — F1 0.778 → 0.854 (+0.076). Recall 0.700 → 0.820; precision
  held at 0.891. Medium-difficulty F1 0.643 → 0.882 (the headline fix).
  Confusion: TP 35 → 41, FN 15 → 9, FP/TN essentially unchanged. Audit
  (backend/data/citation_fn_audit_sprint10.md) refuted the initial
  "paraphrase + numeric cluster" hypothesis and revealed a deeper bug: the
  seven citation regexes used `[^\.!\n]{10,}` to capture the claim, so a
  period inside a decimal number ("99.2%", "$2.8 million", "6.5 days") or a
  title abbreviation ("Dr.", "Inc.") truncated the claim mid-sentence —
  dropping the fabricated tail entirely. Every medium-difficulty FN landed
  on this bug.

  Changes in `pisama_core.detection.detectors.citation`:
  - Claim capture group `[^\.!\n]{10,}` → `[^!\n]{10,}` across all
    `_CITATION_PATTERNS`. Claims are now trimmed to a single sentence
    post-extraction by a new `_trim_to_first_sentence` helper that skips
    decimal points, currency decimals, and known title abbreviations
    (Dr., Mr., Mrs., Ms., Prof., Inc., Corp., etc.).
  - Small bidirectional synonym map (`_SYNONYM_MAP`, 10 pairs covering
    budget↔allocated, record↔report, serves↔covers, annual↔yearly,
    manual↔human, review↔check) applied inside `_claim_supported_by_source`
    before set overlap. Catches paraphrased grounded citations that the
    previous word-set intersection missed.
  - `min_support_overlap` 0.30 → 0.25. The synonym map + sentence fix move
    most paraphrased TNs back above the threshold; the lower bar only
    admits a handful of additional recalls and does not change precision.
  - `_APPROXIMATION_MARKERS` expanded with `close to`, `just over`,
    `just under`, `nearly`, `almost exactly`, `a bit more than`,
    `approximately equal to`, and `in the [low|mid|high]
    [millions|billions|thousands|hundreds]`.
  - New numeric-fact pattern for single-digit measurements with strong
    units (`\b\d(?:\.\d+)?\s*(?:minutes?|hours?|days?|weeks?|months?|years?|percent|stars?)\b`)
    — catches fabricated small quantities like "8 minutes" when source
    says "18 minutes".
  - Digit-boundary fix in `_numeric_facts_unsupported` via a new
    `_fact_present_in_source` helper: "8minutes" no longer counts as
    supported by "18minutes". Prevents the numeric-fact check from
    silently missing fabrications that are digit-prefix-subsumed by a
    larger source number.
- 18 regression tests added in `tests/test_citation.py` covering the
  decimal-point / title-abbreviation bugs, synonym-aware overlap, expanded
  approximation markers, and precision guards.

### Notes

- Public API unchanged. All changes are internal to
  `pisama_core.detection.detectors.citation`.
- Backend adapter at `backend/app/detection_enterprise/detector_adapters.py`
  is untouched.

## [1.6.2] — 2026-04-18

SDK hygiene release. No detector changes.

### Added

- `DeepAgentsAdapter` and `parse_deep_agents_trace` now re-exported from
  `pisama_core.adapters` (they were defined in 1.6.0 but not surfaced
  at the package level — customers had to import from the submodule).
- Unit test coverage for `DeepAgentsAdapter`: write_todos → TASK goals,
  subagent spawn spans, LangGraph state transitions, platform/version
  reporting, convenience function shape, empty/malformed-trace handling,
  session ID propagation (8 tests).

### Notes

- Public API is unchanged for existing callers (v1.6.0 imports from
  the submodule path continue to work).
- No backend / calibration changes.

## [1.6.1] — 2026-04-18

Sprint 7 patch release. One pisama-core detector fix (approval_bypass).
Backend-side wins for context, coordination, and grounding live in the
backend repo.

### Changed

- **approval_bypass** — F1 0.756 → 0.860 (+0.104). Recall 0.646 → 0.833,
  precision held at 0.889. Prior `_check_approval_before` had an
  unconditional shortcut that treated ANY upstream `USER_INPUT` span as
  "user is in the loop" approval. In practice, users frequently send
  exploratory questions ("What would happen if we dropped the
  temp_calculations table?"), hedges ("We might need to pay an invoice
  soon"), or problem statements ("There might be an issue with MFA on
  some accounts") — none of which grant approval. Replaced the shortcut
  with a tri-state `_user_text_is_approval()` classifier over three
  class-level lists: `APPROVAL_PHRASES` (explicit approvals),
  `EXPLORATORY_MARKERS` (questions/hedges — not approval, keep scanning),
  and `IMPERATIVE_STARTERS` (sentence-initial action verbs — approval).
  Five new regression tests cover all three FN semantic patterns plus
  no-regression cases.

### Notes

- Public API unchanged. All changes are internal to
  `pisama_core.detection.detectors.approval.ApprovalBypassDetector`.
- Backend-side Sprint 7 wins for **context** (F1 0.731 → 0.782 via
  semantic-divergence blend mirroring Sprint 6 Phase BB on specification),
  **grounding** (F1 0.743 → 0.745 via NLI threshold 0.3 → 0.4 — marginal
  lift; the NLI model is the precision ceiling on this dataset), and the
  `semantic_stuck_pair_signal` infrastructure on **coordination** (F1
  0.746 → 0.750 — credits-blocked; full judge escalation will unblock
  the remaining 49 MAST AG2 real-trace FN) live in the backend repo.
- Calibration report at `backend/data/calibration_report.json`.
- Registry: 48 tested, post-Sprint 7 distribution in `capability_registry.json`.

## [1.6.0] — 2026-04-18

Sprint 4 calibration + new platform adapter. Net-new Deep Agents adapter
(minor bump) plus backend-side detector wins for communication and
delegation. Mean F1 0.846 → 0.848 across 57 detectors.

### Added

- **`pisama_core.adapters.deep_agents.DeepAgentsAdapter`** — new adapter
  for LangChain Deep Agents traces. Maps `write_todos` planning spans to
  `AgentState` goals (TASK spans, not TOOL — avoids double counting),
  subagent spawns to child spans, and LangGraph state transitions to
  state deltas. Reports as `Platform.LANGGRAPH` with
  `platform_version="deep-agents-v1"` (no enum expansion). Module-level
  `parse_deep_agents_trace()` convenience function mirrors the LangGraph
  adapter shape. 13 unit tests.
- Integration guide at
  `docs-site/docs/guides/integrations/deep-agents.md`.

### Notes

- Public API additions only; existing adapters unchanged.
- Backend-side Sprint 4 wins for **communication**
  (F1 0.733 → 0.769, +0.036; contradictory-pair anti-exemption promoted
  above the action_confirmed topic-overlap exemption, new older/newer
  ↔ regardless substitution pair, confidence 0.9 on substitution-pair
  mismatches) and **delegation** (F1 0.738 → 0.830, +0.092; overall
  detection threshold tightened 0.45 → 0.28 and adapter schema fallback
  for legacy golden-dataset keys `delegator_request`/`delegate_output`)
  live in the backend repo.
- Phase M specification counter rewrite and Phase P+Q context
  escalation / coordination gray-zone widening were both attempted and
  **rejected**. Specification: counter unit tests passed but threshold
  sweep found no combination clearing F1 ≥ 0.72 — the FN cluster is
  frequency-mismatch / sync-mismatch / access-control, not scope
  expansion. Context + coordination: LLM-judge non-determinism produced
  ±0.08 F1 swings on identical code, so the measured +0.04 gains did not
  clear the noise floor and were not reproducible. Future judge-touching
  sprints should average across `--trials N` before accepting deltas.
- Calibration report at `backend/data/calibration_report.json`
  (mean F1 0.848 across 57 detectors as of 2026-04-18T07:02Z).
- Registry: 48 tested — 34 production, 14 beta, 0 experimental,
  2 untested (up from 33/14/1/2).

## [1.5.1] — 2026-04-18

Sprint 3 calibration: one pisama-core detector fix (propagation) plus
three backend-side wins (dispatch_async, communication, derailment).
Mean F1 0.842 → 0.846 across 57 detectors.

### Changed

- **propagation** — F1 0.730 → 0.899 (+0.169). Precision 0.636 → 1.000;
  recall 0.857 → 0.816. Fixed the dominant FP pattern where every number
  in the fact registry was cross-compared against all priors within the
  same step (e.g. `$9.99` vs `1` registered from "Order #1000: 1 items at
  $9.99 each" flagged as same-magnitude contradiction). Propagation now
  requires a genuine replacement relation before firing the magnitude
  check, and treats unit conversions (`2.75 kg` → `6.06 lb`) and
  abbreviation equivalents (`€850,000` → `€850K`) as non-contradictions.

### Notes

- Public API is unchanged. Backend-side wins for dispatch_async
  (latency 300s → 600s + co-occurrence requirement, `_has_recovery()`
  helper for cross-action retries), communication (intent threshold
  0.45 → 0.35, four new substitution pairs), and derailment (six
  hard-slice golden entries added to rebalance the cap subset) live in
  the backend repo.
- Phase G grounding logistic-confidence fix and Phase I specification
  scope-expansion fix were both tried and **rejected** against the
  acceptance gates. Root-cause notes captured in
  `memory/project_detector_calibration_status.md` for future sprints.
- Calibration report at `backend/data/calibration_report.json`
  (mean F1 0.846 across 57 detectors as of 2026-04-18T06:16Z).
- Registry: 48 tested — 33 production, 14 beta, 1 experimental,
  2 untested (up from 32/16/0/2).

## [1.5.0] — 2026-04-18

Follow-up calibration sprint. One major pisama-core detector fix
(critic_quality) plus infra and backend-side fixes (sampling drift,
grounding LLM judge rewire, loop detector). Mean F1 0.832 → 0.842
across 57 detectors.

### Changed

- **critic_quality** — F1 0.686 → 0.966 (+0.280). Precision held at
  1.000; recall 0.522 → 0.935. Four deterministic pattern groups:
  expanded `_APPROVAL_PATTERNS` (clean/simple, well-organized,
  professional, follows-conventions variants); new `_PRODUCER_RED_FLAGS`
  (bubble sort, divide-by-len() without guard, plaintext password
  compare, Scanner without try-with-resources, CREATE TABLE without
  constraints, fetch().json() without error handling, stub bodies with
  placeholder comments, multi-key dict access without `.get()`);
  cosmetic-only critique detection gated on red flags; question-only
  and substanceless-critique detection.

### Notes

- Public API is unchanged. Backend-side wins for grounding (LLM judge
  gray-zone rewire, 7 new tests), loop (bookkeeping-aware progress
  checks), and OpenClaw token/cost passthrough live in the backend repo.
- Calibration sampling is now deterministic (`cap_per_type` sorts inputs
  before `random.sample`), so sprint-over-sprint F1 deltas are honest.
- Calibration report at `backend/data/calibration_report.json`
  (mean F1 0.842 across 57 detectors as of 2026-04-18).
- Registry: 48 tested — 32 production, 16 beta, 0 experimental,
  2 untested (up from 30/17/1/2).

## [1.4.0] — 2026-04-17

Detector calibration sprint. Five detectors significantly improved, two promoted
toward production tier.

### Changed

- **task_starvation** — F1 0.667 → 0.980. Fixed `_extract_executed_tasks` to
  accept `SpanKind.TASK` spans (prior bug produced TN=0 on task-only traces).
  Added planner-span exclusion, substring task matching, and a small
  Porter-lite stemmer so "summarising" matches "summarize" etc.
- **critic_quality** — recall 0.33 → 0.52. Added reviewer/approver role
  detection and softened the requirement for an explicit approve/reject
  verdict when structured feedback is present.
- **mcp_protocol** — recall 0.26 → 0.98. Broadened the tool-call shape
  matcher to cover both `tool_calls` (OpenAI-style) and `tool_use` (Anthropic)
  payloads, and to treat inline `<tool_use>…</tool_use>` XML the same as
  structured blocks.
- **citation** — F1 0.557 → 0.677 (P 0.800→0.827, R 0.427→0.573). Added a
  numeric-fact unsupported-claim check (percentages, currency, integer+unit)
  gated by approximation markers so paraphrases are not flagged.
- **entity_confusion** — F1 0.607 → 0.823 (P 0.537→1.000). Scoped the
  attribute-proximity heuristic to sentence boundaries via `_attribute_closer_to`
  and added `_entity_referenced` word-boundary matching (surname / given-name,
  role prefix ignored).

### Notes

- Public API is unchanged. All changes are internal to detector implementations
  and corresponding test fixtures.
- Calibration report at `backend/data/calibration_report.json` (mean F1 0.825
  across 57 detectors as of 2026-04-17).

## [1.3.0] — prior release

Baseline prior to the 2026-04-17 calibration sprint.
