# Builder Task — Manual Telemetry Fix

Branch: `builder/manual-telemetry-fix`
Base: `beast-manual-baseline`

## Objective
Apply exactly one functional change in `src/jobs/live_trade.py`.

Replace:

```python
loop_status="error" if status == "error" else "completed",
```

with:

```python
loop_status="completed",
```

## Do Not Change
- No other functional code.
- No formatting sweep.
- No encoding conversion.
- No smart punctuation replacement.
- No unrelated cleanup.
- Do not touch Beast Manual runtime behavior beyond this telemetry status line.

## Acceptance Test
The resulting diff must show:
- `src/jobs/live_trade.py`: exactly one functional line changed.
- No mojibake / encoding corruption.
- No other files changed except this task file, which may be deleted before final merge if desired.

Commit the fix and stop. Reviewer will inspect the diff before anything is merged or used as the Auto baseline.
