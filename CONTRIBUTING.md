# Contributing to pisama-core

Thanks for your interest in improving `pisama-core`. This package is
the detection, scoring, and healing engine that underpins
[Pisama](https://pisama.ai). Calibrated production weights, golden
datasets, and enterprise detectors live in Pisama Cloud — that split
is deliberate and not up for debate in PRs.

## What we're looking for

- **New framework adapters** under `src/pisama_core/adapters/`. Each
  adapter converts framework-native traces into the core's canonical
  trace shape.
- **Bug reports** with a minimal reproducer — especially scoring edge
  cases or adapter data-loss.
- **Documentation fixes** on the public API reference.

## What we're not looking for

- Tuned scoring thresholds. Defaults are intentionally conservative
  so `pisama-core` works without calibration data.
- Calibration pipelines, golden-dataset generators, or ML model
  artifacts. Those are Pisama Cloud features.
- Changes to the optional tokenization module that weaken its
  redaction guarantees.

## Development setup

```bash
git clone https://github.com/tn-pisama/pisama-core.git
cd pisama-core
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest tests/
```

## PR checklist

- [ ] New adapter (if applicable) has a round-trip test:
      `framework → canonical → framework`.
- [ ] Public API surface documented in the module docstring and in
      the README API section.
- [ ] Clean-venv install succeeds with the declared dependencies.
- [ ] Existing tests pass: `pytest tests/ -q`.

## Licensing and contributor grant

By submitting a PR you agree that your contribution is licensed under
MIT, the same license as this repo.

## Questions

Open a GitHub Discussion or visit [pisama.ai](https://pisama.ai).
