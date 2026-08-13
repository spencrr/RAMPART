# Parallel Execution with pytest-xdist

RAMPART supports parallel test execution via `pytest-xdist`, producing a **single unified report** even when tests run across multiple worker processes.

---

## Quick Start

```bash
pip install pytest-xdist
pytest -n 4
```

With `-n 4`, pytest spawns 4 worker processes that execute tests in parallel. RAMPART intercepts each worker's results, ships them to the controller process, and emits **one consolidated report** at the end of the session.

---

## How It Works

```
Worker 1                    Worker 2                    Controller
─────────                   ─────────                   ──────────
complete item               complete item
    │                           │
teardown TestReport         teardown TestReport
    │                           │
v2 Result envelope          v2 Result envelope
    │                           │
    └───────────┬───────────────┘
                ▼
        pytest_runtest_logreport
        validate + append immediately
                │
                ▼
        pytest_testnodedown (per worker)
        validate clean-finish manifest
        and deprecated clone metadata
                │
                ▼
        pytest_sessionfinish (controller)
        reconcile delivery counts
        aggregate trials → evaluate gates → emit sinks
                │
                ▼
        Single unified TestRunReport
```

- **Workers** collect [`Result`][rampart.core.result.Result] objects normally and
  attach one bounded v2 envelope to each nonempty item's teardown report.
  Workers do **not** emit report sinks.
- **Controller** validates each envelope and immediately appends its Results to
  [`RampartSession`][rampart.pytest_plugin._session.RampartSession]. At session
  end it reconciles clean-finish manifests, evaluates gates, and emits sinks
  once.

The result: **one** `JsonFileReportSink` output file, **one** call to `MyCustomSink.emit_async`, and accurate population statistics over the full result set.

### Authoritative v2 envelopes

Each nonempty item attempt carries one private authoritative JSON envelope:

```json
{
  "schema_version": 2,
  "sequence": 7,
  "produced": 2,
  "results": [
    {
      "index": 0,
      "result": {
        "safe": true,
        "status": "safe",
        "summary": "Agent defended successfully",
        "turns": [],
        "duration_seconds": 0.4,
        "harm_category": "prompt_injection",
        "strategy": "xpia",
        "observability_level": "response_only",
        "injections": [],
        "metadata": {
          "_pytest_test_name": "test_injection",
          "_rampart_result_index": 5
        }
      }
    }
  ],
  "dropped": [
    {
      "index": 1,
      "reason": "size_limit",
      "serialized_bytes": 73400320
    }
  ]
}
```

- `sequence` is monotonic and worker-local. The delivery identity is
  `(worker_id, sequence)`. Each successful or dropped record has a structural,
  zero-based `index`, so `(worker_id, sequence, index)` identifies one local
  slot and the two record lists exactly partition `0..produced-1`.
- The structural slot is independent of Result metadata.
  When present, `_rampart_result_index` retains the cumulative scheduling index
  assigned by the authoritative worker session; v2 does not rewrite, alias, or
  synthesize it when a low-level input omits the key.
- The envelope does not carry a worker ID or node ID. The controller trusts
  the `worker_id` and `nodeid` reconstructed by pytest-xdist.
- Duplicate delivery of identical content is idempotent. Reusing an identity
  for different content is a transport conflict.
- A zero-Result item emits no envelope and consumes no sequence.
- A clean worker sends a separate constant-size manifest with envelope,
  sequence, Result, and drop counts. Missing manifests, gaps, count
  mismatches, malformed data, and worker loss mark the run incomplete.

The controller accepts a valid envelope before waiting for worker shutdown.
Exact duplicate delivery is idempotent; a conflicting duplicate faults without
replacing the first accepted data. A later malformed envelope, missing
manifest, worker crash, sequence/count mismatch, or dropped sibling marks the
run incomplete but cannot remove Results that already arrived. This retained
partial evidence is included in terminal output, JSON reports, custom sinks,
trial summaries, and population statistics.

---

## Execution-domain trials with xdist

`execute_trials_async` runs its N executions sequentially inside the worker that
owns the surrounding pytest item. xdist schedules pytest items; it does not
split one helper call across workers.

| `--dist` mode | Execution-domain batch behavior |
|---------------|---------------------------------|
| `load` (default) | The item runs on one selected worker; all N Results share one batch UUID |
| `loadgroup` / `loadscope` / `loadfile` | Grouping changes which worker owns the item, not its internal trial behavior |
| `each` | Every worker invokes the item and creates its own UUID4 batch with N Results |

Internal trials therefore do not provide worker-level parallelism. Use explicit
pytest parametrization when each trial needs separate worker scheduling,
fixture setup, retries or selection, or a separate JUnit testcase.

One helper invocation shares its adapter and fixture/process context. Built-in
executions request a fresh adapter Session for each Result, but external agent
and backend state may still persist. Custom executions control their own
session behavior.

Every execution Result travels through the authoritative v2 item envelope.
The controller reconstructs `TrialBatchSummary` objects from versioned
primitive Result metadata and orders summaries by first Result occurrence, not
by their random UUIDs.

### Deprecated clone marker with xdist

The clone-based `@pytest.mark.trial` marker remains during `0.2.x` and is removed
in `0.3.0`. Its old behavior is unchanged:

| `--dist` mode | Deprecated clone behavior |
|---------------|---------------------------|
| `each` | Every worker executes every trial clone |
| `loadgroup` | All clones for one test are pinned to one worker |
| `load` (default) | Clones may be distributed across workers |
| `loadscope` / `loadfile` | Clones are grouped by class/module/file |

Controller aggregation remains correct in every mode. The informational
`--dist=loadgroup` warning remains for clone locality, and `--dist=each`
continues to retain every worker attempt while gating the logical clone count.
Until removal, workers send only clone node IDs, base node IDs, and thresholds
through a temporary `rampart.xdist.trial-specs.v1` clean-finish payload.
Missing or malformed clone metadata marks the run incomplete without
discarding Results that arrived through v2. Do not use the marker for new
tests.

---

## Registering Sinks: the `pytest_rampart_sinks` hook

The **recommended** way to register report sinks is the `pytest_rampart_sinks`
hook. It is resolved on the controller — which never executes fixtures — so it
behaves identically in single-process and xdist runs, and (unlike the fixture
path) supports sinks that need configuration.

Implement it in your `conftest.py`:

```python
# conftest.py
from pathlib import Path

from rampart.reporting import JsonFileReportSink


def pytest_rampart_sinks(config):
    return [JsonFileReportSink(output_dir=Path(".report"))]
```

- Multiple implementations are supported; RAMPART emits to the **union** of every
  returned sink.
- An implementation may return an empty list to contribute none.
- Non-`ReportSink` items (or a non-list return) are dropped with a warning, so one
  malformed implementation cannot break emission.

### Precedence vs the `rampart_sinks` fixture

!!! warning "Deprecated"
    The `rampart_sinks` fixture is deprecated and will be removed in `0.3.0`.
    Prefer the `pytest_rampart_sinks` hook above. Resolving the fixture emits a
    `DeprecationWarning` in both single-process and controller discovery.

The legacy `rampart_sinks` fixture is still supported as a **single-process
fallback**. The rule is:

- If **any** `pytest_rampart_sinks` hook implementation exists, the hook is
  authoritative and the fixture path is skipped entirely (so a project that
  defines both does **not** double-register).
- If **no** hook implementation exists, RAMPART falls back to the fixture. On the
  xdist controller this fallback scans registered conftest modules for a
  `rampart_sinks` attribute.

### Fixture fallback constraints (no hook present)

When you rely on the fixture fallback under xdist, pytest's fixture machinery
does not run on the controller. RAMPART therefore unwraps a **parameterless**
`rampart_sinks` fixture and calls its underlying function directly, so these
shapes resolve:

```python
# Parameterless session fixture — resolves single-process AND on the
# xdist controller.
@pytest.fixture(scope="session")
def rampart_sinks():
    return [JsonFileReportSink(output_dir=Path(".report"))]

# Plain list assigned at module level — resolved on the xdist controller
# only. Single-process discovery looks up a *fixture* named rampart_sinks,
# so a bare module-level list is silently ignored there; use the fixture
# form above (or the hook) for single-process runs.
rampart_sinks = [JsonFileReportSink(output_dir=Path(".report"))]
```

A **fixture with dependencies** cannot be resolved on the controller and is
skipped with a warning:

```python
# Not resolvable on the controller — use the hook instead
@pytest.fixture(scope="session")
def rampart_sinks(my_sink_config, db_connection):
    return [DatabaseSink(connection=db_connection)]
```

If your sinks need dependencies, **use the `pytest_rampart_sinks` hook** — it
receives the `pytest.Config` and runs on the controller, so you can build sinks
from `config` values or environment variables there.

---

## Transport Validation and Evidence Fidelity

Worker Results cross the process boundary on pytest-xdist teardown reports.
RAMPART projects Result data to JSON-safe primitives and applies the following
transport validation and normalization:

- **Schema version** — envelopes and manifests with missing or unknown schema
  versions are rejected.
- **Structural identity** — worker sequence and local slots are validated
  independently from Result scheduling metadata.
- **Value shape** — exact enum values, container depth, JSON-compatible
  metadata, finite numeric values, and full nested Result fields are validated.
- **Route metadata** — the controller supplies node and source-worker identity;
  envelopes cannot forge it.
- **Deprecated clone metadata** — the separate temporary trial-spec payload is
  exact-schema and cannot contain Results.
- **Worker-local artifacts** — artifact paths are retained as opaque metadata
  strings; the controller does not access worker files.

Textual evidence is preserved through deserialization, so serial and xdist
runs expose the same text to `Result`, `TestRunReport`, and report sinks.
RAMPART escapes terminal controls only when it renders its own terminal summary
or dynamic log fields. The built-in JSON sink represents those characters with
JSON escapes, while parsing the file restores the original text. Custom sinks
receive raw evidence and own the safety of any human-facing rendering.

### Size cap

The default 64 MB cap can be overridden via the pytest CLI option or an ini setting:

```bash
pytest -n 4 --rampart-xdist-max-bytes=134217728
```

Or in `pytest.ini` / `pyproject.toml`:

```ini
[pytest]
rampart_xdist_max_bytes = 134217728
```

The value caps each authoritative per-item envelope. RAMPART measures the
actual compact JSON encoding in UTF-8 bytes, including structural wrappers. An
individually oversized Result becomes a bounded drop record while fitting
siblings remain. When combined Results exceed the cap, RAMPART scans them in
item order, retains each Result that fits, and records each nonfitting sibling
as a drop. The final attribute never exceeds the cap and is never split across
multiple report properties.
If even an all-drop envelope cannot fit, no attribute is emitted and the clean
manifest reports the omitted count. Every drop marks the run incomplete and
forces an otherwise-successful run to exit nonzero.

The temporary deprecated clone trial-spec payload is not governed by this
option and never carries Result data.

---

## Incomplete Runs

If a worker crashes, runs out of time, or hits the size cap, the controller marks the run as incomplete:

```python
report.metadata["incomplete"]            # True if any worker failed
report.metadata["incomplete_reasons"]    # list[str] — one per failure
```

Reports are still emitted with whatever data was collected, and RAMPART changes
an otherwise-successful pytest exit status to `TESTS_FAILED`. Sinks and
post-processing can also inspect the `incomplete` flag for diagnostics.

---

## Run-Mode Metadata

Reports produced under xdist include:

```python
report.metadata["xdist_active"]   # True
report.metadata["worker_count"]   # int
report.metadata["dist_mode"]      # "load", "loadgroup", etc.
```

---

## Durability boundaries

Once a completed item's teardown envelope reaches the controller, its valid
Results are retained even if that worker later crashes or fails clean-finish
reconciliation. Reports remain explicitly incomplete and exit nonzero, but
preserve the evidence that did arrive.

Results can still be absent when a worker dies before its teardown report is
sent, when an individual Result exceeds the configured envelope cap, or when an
envelope is malformed. These cases are reported through bounded incomplete
reasons; they never trigger a fallback to clean-finish bulk Result data.

The deprecated clone marker still depends on its small clean-finish trial-spec
payload. Losing that payload fails the run closed, while independently
delivered Results remain available.

---

## Limitations

- Sinks discovered through the **fixture fallback** on the controller cannot depend
  on other pytest fixtures — use the `pytest_rampart_sinks` hook instead (see
  [Registering Sinks](#registering-sinks-the-pytest_rampart_sinks-hook)).
- Results from an item whose teardown report never reaches the controller
  cannot be recovered (see [Durability boundaries](#durability-boundaries)).
- Mixed RAMPART versions across controller and workers are unsupported; install the
  same version everywhere.
- `pytest-xdist` itself does not support interactive debugging (`--pdb`, `--trace`);
  use single-process mode for debugging.
