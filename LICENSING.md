# Licensing

This document is for anyone doing license diligence on `pisama-core`, an
acquirer, counsel, or a downstream user auditing their dependency tree. It
explains what this package is licensed under, how it relates to the
separate `pisama-detectors` package, and why using `pisama-core` carries no
obligations from that other package's license.

## What license is this package under

`pisama-core` is MIT licensed. See [`LICENSE`](LICENSE) for the exact text.
MIT is permissive: you can use, modify, and redistribute this package,
including in closed-source and commercial products, with no copyleft or
source-disclosure requirement. This has been true since the package's first
release and is not changing.

## What this package is

`pisama-core` ships the detection, scoring, and healing engine for AI agent
systems: loop detection, cost overruns, coordination breakdowns, and a
broader set of heuristic detectors, all running offline with no API key and
no network call. It is meant to be a permanently free, permanently open
baseline tier: simple, auditable, dependency-light heuristics anyone can
read, fork, and run in production without asking anyone's permission.

## How `pisama-detectors` relates to this package

`Pisama-AI/pisama-detectors` is a separate package covering the same
failure-mode taxonomy (loops, hallucinations, injection, state corruption,
coordination failures, and more) with independently implemented,
calibrated, production-tuned detector logic, plus detector families for
specific frameworks (Dify, LangGraph, n8n, OpenClaw). It is licensed under
the Business Source License 1.1 (BUSL), not MIT.

The two packages are commonly confused because they sound related and cover
the same taxonomy. They are not the same code under two licenses. They are
two independent implementations that happen to detect the same failure
modes. `pisama-core` does not depend on `pisama-detectors`, and neither
package's source imports from the other; you can verify this yourself with
`pip show pisama-core` (no `pisama-detectors` dependency listed) or by
grepping either package's source tree for an import of the other.

### Evidence: independent implementations, not shared code

Line counts below are from a fresh clone of both repositories at their
current `main` branch heads, for every detector module that exists under
the same name in both packages:

| Detector | `pisama-core` (MIT) | `pisama-detectors` (BUSL) |
|---|---|---|
| `hallucination.py` | 72 lines | 1,028 lines |
| `loop.py` | 219 lines | 1,006 lines |
| `coordination.py` | 103 lines | 1,445 lines |
| `persona.py` | 689 lines | 601 lines |
| `specification.py` | 1,012 lines | 1,594 lines |
| `withholding.py` | 619 lines | 711 lines |

These are not a subset relationship or a fork with edits. Reading the two
`hallucination.py` files side by side shows entirely different approaches:
`pisama-core`'s version is a short heuristic over tool-call error rates on
an ATIF trace; `pisama-detectors`'s version does embedding-based grounding
scoring against retrieved source documents, with citation-pattern regexes
and confidence calibration. Neither is a bigger or smaller version of the
other; they solve the same detection problem with different code. The same
holds across the other pairs in the table. Note that the size relationship
is not even consistently one-directional (`persona.py` is larger in
`pisama-core`, `withholding.py` is close between the two), which is further
evidence these are independent implementations rather than one package
being a stripped copy of the other.

Across the full source trees, `pisama-core/src` is about 27,000 lines and
`pisama-detectors/src` is about 40,000 lines, covering different code paths
throughout, not a shared core with a thin diff.

## What this means if you use `pisama-core`

Because `pisama-core` contains zero code from `pisama-detectors`, using
`pisama-core` carries no BUSL obligations of any kind. There is no
Additional Use Grant to read, no Change Date to track, no non-compete
clause to worry about, and no need to purchase a commercial license for any
reason connected to this package. The only license that applies to
`pisama-core` is the MIT license in this repository's [`LICENSE`](LICENSE)
file, in full, unconditionally.

If you separately choose to install `pisama-detectors` for its calibrated
or framework-specific detectors, that package's BUSL terms apply to that
package's own code, on their own terms. They do not apply to `pisama-core`,
and they do not apply retroactively or by association just because the two
packages are both published by Pisama LLC and cover related ground.

## Questions

If you are doing diligence and something here does not match what you find
in the source, email [tuomo@pisama.ai](mailto:tuomo@pisama.ai). We would
rather fix a documentation gap than have you guess.
