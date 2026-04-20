## What this changes

<!-- 1–3 sentences. What's different after this PR and why. -->

## Type

- [ ] New framework adapter
- [ ] Bug fix (scoring edge case, adapter data-loss, etc.)
- [ ] API change
- [ ] Docs

## Checklist

- [ ] Clean-venv install works: `pip install .` in a fresh env, the
      affected adapter round-trips a sample trace.
- [ ] Public API surface documented in the module docstring and the
      README API section.
- [ ] Existing tests pass: `pytest tests/ -q`.
- [ ] No calibrated thresholds or golden-dataset artifacts added.
- [ ] README adapter matrix updated if a new adapter is added.

## Reproducer or before/after (for bug fixes)

```python
# Before: ...
# After: ...
```
