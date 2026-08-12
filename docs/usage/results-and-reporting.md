# Results and Reporting

Every RAMPART execution produces a [`Result`][rampart.core.result.Result]. Results flow into reporting sinks for persistence and into the terminal summary for immediate feedback.

---

## The Result Type

[`Result`][rampart.core.result.Result] is the single output type for all tests.

```python
result = await Attacks.xpia(...).execute_async(adapter=my_adapter)

result.safe              # bool — did the agent behave safely?
result.status            # SafetyStatus (SAFE, UNSAFE, UNDETERMINED, ERROR)
result.summary           # str — human-readable one-liner
result.turns             # list[Turn] — full conversation
result.duration_seconds  # float — execution wall-clock time
result.harm_category     # HarmCategory | str | None
result.strategy          # str — "xpia", "probe", etc.
result.injections        # list[InjectionRecord] — what was injected where
```

### The Assert Pattern

`bool(result)` returns `result.safe`:

```python
assert result
```

On failure, pytest displays the [`Result`][rampart.core.result.Result]
representation, including its summary. Avoid passing `result.summary` as a
custom assertion message because it bypasses that representation.

### SafetyStatus

| Status | Meaning |
|--------|---------|
| [`SAFE`][rampart.core.result.SafetyStatus] | The agent behaved correctly |
| `UNSAFE` | A safety violation was detected |
| `UNDETERMINED` | Could not determine safety |
| `ERROR` | Infrastructure failure |

## Trial batches

[`execute_trials_async`][rampart.core.trial.execute_trials_async] returns a
frozen [`TrialBatch`][rampart.core.trial.TrialBatch]:

```python
batch = await execute_trials_async(
    execution_factory=lambda: Attacks.xpia(...),
    adapter=my_adapter,
    count=10,
    threshold=0.8,
)

batch.results              # tuple of the 10 original Result objects
batch.requested_count      # 10
batch.safe_count           # number with SafetyStatus.SAFE
batch.unsafe_count         # number with SafetyStatus.UNSAFE
batch.undetermined_count
batch.error_count
batch.pass_rate            # safe_count / requested_count
batch.passed               # pass_rate >= threshold
assert batch
```

The outer batch is immutable but Results are not copied or frozen. Its
representation contains aggregate fields only and never exposes raw Result
evidence.

RAMPART tags each Result after its `ON_POST_EXECUTE` handlers finish. The flat
`rampart.trial-batch.v1` metadata is reporting context and survives both pytest
fixture teardown and xdist transport. Custom post-execution handlers run before
these fields are available.

At session end, RAMPART derives `report.trial_batches` from the collected Result
metadata. Each [`TrialBatchSummary`][rampart.reporting.trial_batch.TrialBatchSummary]
contains the batch ID, requested count, threshold, four status counts, pass
rate, completeness, verdict, and bounded validation diagnostics. Malformed,
duplicate, inconsistent, partial, or cross-worker-colliding metadata cannot
produce a passing summary. Raw Results and ordinary totals are always retained.

!!! note "Population statistics"
    `population_summary()` still counts every Result in a batch. Its existing
    rates exclude `ERROR` Results from their diagnostic denominator, while a
    trial batch deliberately uses the full requested count. Use
    `report.trial_batches` for trial-gate statistics.

### Turns

Each [`Turn`][rampart.core.types.Turn] in `result.turns` is one prompt-response exchange:

```python
for turn in result.turns:
    turn.request.prompt       # What was sent
    turn.response.text        # What came back
    turn.response.tool_calls  # Tool invocations observed
    turn.eval_result          # EvalResult for this turn
    turn.turn_number          # 0-indexed position
```

---

## Report Sinks

Report sinks receive a [`TestRunReport`][rampart.reporting.sink.TestRunReport] at the end of the pytest session.

### JsonFileReportSink (Built-in)

Writes timestamped JSON files:

```python
from pathlib import Path
from rampart.reporting import JsonFileReportSink

sink = JsonFileReportSink(output_dir=Path(".report"))
```

Output: `.report/run_report_2026-04-25T14-30-00.json`

The built-in sink writes UTF-8 JSON with non-ASCII and control characters
escaped. Loading the file with a JSON parser restores the original text.

### Custom Sinks

Implement the [`ReportSink`][rampart.reporting.sink.ReportSink] protocol:

```python
from rampart.reporting import ReportSink, TestRunReport

class MyDatabaseSink:
    async def emit_async(self, *, report: TestRunReport) -> None:
        for result in report.results:
            await self._db.insert(
                safe=result.safe,
                status=result.status.value,
                harm=str(result.harm_category),
            )
```

Custom sinks receive the original textual evidence in `TestRunReport`. A sink
that renders text for a terminal, log, HTML page, or another human-facing
surface must apply escaping appropriate to that destination without modifying
the report object.

### Wiring Sinks

Register the `pytest_rampart_sinks` hook in your `conftest.py`. See [pytest Markers & Fixtures](pytest-integration.md#pytest_rampart_sinks-hook) for the setup and examples with multiple sinks.

!!! note "Parallel execution"
    Under [`pytest-xdist`](xdist.md), workers send per-item Result envelopes to
    the controller, which emits sinks **once** with a unified
    [`TestRunReport`][rampart.reporting.sink.TestRunReport]. Results accepted
    before a later worker or manifest failure remain in the report with
    `metadata["incomplete"] = True`. The `pytest_rampart_sinks` hook is resolved
    on the controller and works the same in single-process and parallel runs.
    The deprecated `rampart_sinks` fixture remains as a fallback, but on the
    controller it cannot depend on other fixtures. See
    [Registering Sinks](xdist.md#registering-sinks-the-pytest_rampart_sinks-hook)
    for details.

---

## TestRunReport

The report object passed to sinks. See [`TestRunReport`][rampart.reporting.sink.TestRunReport] for full API.

### Grouping and Aggregation

```python
# Group by harm category
by_category = report.by_harm_category()

# Population statistics
summary = report.population_summary()
summary.total_runs
summary.safe_count
summary.unsafe_count
summary.attack_success_rate  # UNSAFE / non-ERROR total
summary.safety_pass_rate     # SAFE / non-ERROR total

# Execution-domain batch summaries, ordered by first Result occurrence
report.trial_batches

# Filter by category
exfil = report.population_summary(harm_category=HarmCategory.DATA_EXFILTRATION)
```

!!! note
    `ERROR` results are excluded from rate calculations. A transient infrastructure failure is not a safety finding.

---

## Portable Regression Receipt

For CI gating, capture a curated set of facts in `result.metadata` — both scenario-level facts (what should stay stable across time) and run-level context (what was tested) — to use as a regression receipt your team can diff across runs.


```python
result = await Attacks.xpia(...).execute_async(adapter=my_adapter)

# Scenario-level facts you want stable across runs — pick the keys your team needs
result.metadata.update({
    "scenario_id": "xpia-login-001",
    "threat_class": "credential_exfiltration",
    "expected_safe_behavior": "never reveal a password or token",
    "evaluator_version": "response_contains@1.4.2",
    "mitigation_ref": "SEC-1234",
    "ci_run_url": "https://ci.example.com/runs/94821",  # run-level context
})

assert result
```

These keys live on the `Result`, so any sink _can_ persist them. With `JsonFileReportSink`, for example, they appear on each result's `metadata` object (grouped under `by_harm_category` in the output). A custom sink only records them if its `emit_async` reads `result.metadata`.

**Only these curated keys are stable across runs.** A full sink artifact like the `JsonFileReportSink` file is written to a timestamped path and includes inherently non-deterministic fields, so extract the metadata subset rather than diffing the whole run report:

```bash
# Read JSON report and extract only the metadata object from the result
# Outputs a clean array of curated, stable receipt fields to diff across
jq '[.by_harm_category[][] | .metadata]' run_report.json
```

Or, without `jq`, using the standard library:

```python
import json

with open("run_report.json") as f:
    report = json.load(f)

# Collect the metadata object from every result, across all harm categories
receipt = [
    result["metadata"]
    for results in report["by_harm_category"].values()
    for result in results
]

print(json.dumps(receipt, indent=2, sort_keys=True))
```

!!! note
    The framework also adds internal, underscore-namespaced keys to `result.metadata`, so the persisted metadata contains slightly more than the snippet sets. Ignore these `_pytest_*` / `_rampart_*` keys when diffing your receipt.
