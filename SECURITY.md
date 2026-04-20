# Security Policy

## Reporting a vulnerability

If you've found a security issue in `pisama-core`, please do **not**
open a public GitHub issue. Instead:

- Email **security@pisama.ai** with a description, reproducer, and
  the affected version.
- We'll acknowledge within 2 business days and aim to ship a fix or
  mitigation within 7 business days for high-severity issues.

## What counts as a security issue

- Code execution via crafted trace input processed by the core engine.
- The optional tokenization module leaking plaintext through an
  adapter path it was meant to redact.
- Dependency vulnerabilities that affect the detection or scoring
  surface.

## Supported versions

Only the latest 1.x release line is supported. Earlier minors receive
security fixes for 90 days after a new minor ships.

## Credit

We'll credit reporters in release notes unless you prefer to stay
anonymous.
