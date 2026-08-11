# CI Integration

RAMPART tests run as standard pytest tests. This guide covers patterns for CI pipelines.

---

## Running in CI

```bash
pytest tests/ -v --tb=short
```

RAMPART tests interact with real or simulated agents and may take longer than unit tests. Set appropriate timeouts:

```bash
pytest tests/ -v --timeout=300
```

### Parallel Execution

For faster CI runs, use [`pytest-xdist`](xdist.md):

```bash
pip install pytest-xdist
pytest tests/ -n auto
```

RAMPART aggregates results across worker processes and emits a single unified
report under **any** `--dist` mode. Execution-domain trials stay sequential on
the worker that owns their pytest item; xdist distributes items, not the
executions inside one `execute_trials_async` call.

---

## Trial Batches for Statistical Confidence

Use `execute_trials_async` when a single execution is not conclusive:

```python
async def test_injection_resistance(adapter):
    batch = await execute_trials_async(
        execution_factory=lambda: Attacks.xpia(...),
        adapter=adapter,
        count=10,
        threshold=0.8,
    )
    assert batch
```

This requests 10 sequential Results and passes when at least 80% are `SAFE`.
`UNSAFE`, `ERROR`, and `UNDETERMINED` remain in the denominator; an `UNSAFE`
Result is tolerated when the explicit threshold still passes.

**CI boundary:**

- one helper call is one pytest item, fixture lifecycle, worker, and JUnit case
- every execution is a distinct object and produces a separate RAMPART Result
- built-in executions request fresh adapter Sessions, but adapter, fixture,
  process, external-agent, and backend state may persist
- use explicit parametrization for worker distribution, per-trial fixture
  setup, retries or selection, and individual JUnit cases
- assert the returned batch to make its threshold the pytest gate

!!! warning "Deprecated clone marker"
    The clone-based `@pytest.mark.trial` marker remains only for the `0.2.x`
    migration window and is removed in `0.3.0`.

---

## Structured Reports

Register the `pytest_rampart_sinks` hook to write JSON reports for downstream processing:

```python
# conftest.py
from pathlib import Path

from rampart.reporting import JsonFileReportSink


def pytest_rampart_sinks(config):
    return [JsonFileReportSink(output_dir=Path(".report"))]
```

The JSON file contains aggregate statistics and per-result data that CI dashboards can consume. The hook is resolved on the controller, so it behaves identically in single-process and [`pytest-xdist`](xdist.md) CI runs. See [Registering Sinks](pytest-integration.md#pytest_rampart_sinks-hook).

!!! warning "Deprecated"
    The older `rampart_sinks` fixture still works but is deprecated and will be removed in `0.3.0`. Prefer the `pytest_rampart_sinks` hook above.

---

## Pytest Options

RAMPART is configured via pytest options and Python (sinks, adapters, payloads).

### `--rampart-xdist-max-bytes`

Maximum size in bytes used by both xdist transport versions. Defaults to
`67108864` (64 MB). The authoritative v1 bulk payload keeps its existing
worker-wide behavior: an over-cap payload is dropped and the run is incomplete.
The v2 validation channel applies the value to each private per-item shadow
envelope, retaining a deterministic fitting prefix and recording oversized
Results as drops. V2 drops also make the run incomplete, but never replace or
supplement v1 report Results. Also configurable via the
`rampart_xdist_max_bytes` ini option.

```bash
pytest -n auto --rampart-xdist-max-bytes=134217728   # 128 MB
```

---

## Environment Variables

Your adapter and test configuration typically read environment variables. Setting them locally for ad-hoc runs:

=== "Linux / macOS"

    ```bash
    export AGENT_API_KEY="..."
    export AGENT_ENDPOINT="https://..."
    pytest tests/
    ```

=== "Windows (PowerShell)"

    ```powershell
    $env:AGENT_API_KEY = "..."
    $env:AGENT_ENDPOINT = "https://..."
    pytest tests/
    ```

Then consume them in your adapter and configuration:

```python
import os
from rampart.core.llm import LLMConfig

@pytest.fixture
def adapter():
    return MyAdapter(
        api_key=os.environ["AGENT_API_KEY"],
        endpoint=os.environ["AGENT_ENDPOINT"],
    )

# For LLM-driven attacks
llm = LLMConfig(
    model="gpt-4o",
    endpoint=os.environ["OPENAI_ENDPOINT"],
    api_key=os.environ.get("OPENAI_API_KEY"),  # None → azure-identity
    deployment=os.environ.get("OPENAI_DEPLOYMENT"),
)
```

---

## Exit Codes

RAMPART preserves existing nonzero pytest exit codes. An asserted
execution-domain `TrialBatch` fails like an ordinary pytest assertion. For the
deprecated clone marker only, RAMPART changes an otherwise-successful status to
`1` when its aggregate fails. Incomplete xdist runs also force status `1`.
Collect-only runs do not evaluate deprecated clone gates.

| Exit Code | Meaning |
|-----------|---------|
| `0` | All tests passed |
| `1` | Tests failed, a deprecated clone gate failed, or the run was incomplete |
| `2` | Test execution interrupted |
| `3` | Internal pytest error |
| `4` | Pytest usage or configuration error, including invalid RAMPART marker values |
| `5` | No tests collected |
