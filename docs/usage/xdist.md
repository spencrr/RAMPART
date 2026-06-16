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
collect results             collect results
    │                           │
pytest_sessionfinish        pytest_sessionfinish
    │                           │
serialize → workeroutput    serialize → workeroutput
    │                           │
    └───────────┬───────────────┘
                ▼
        pytest_testnodedown (per worker)
        deserialize + merge into
        controller's RampartSession
                │
                ▼
        pytest_sessionfinish (controller)
        aggregate trials → evaluate gates → emit sinks
                │
                ▼
        Single unified TestRunReport
```

- **Workers** collect [`Result`][rampart.core.result.Result] objects normally and serialize them into `config.workeroutput`. Workers do **not** emit reports.
- **Controller** receives each worker's payload via the `pytest_testnodedown` hook, merges results into its own [`RampartSession`][rampart.pytest_plugin._session.RampartSession], and emits sinks once at session end.

The result: **one** `JsonFileReportSink` output file, **one** call to `MyCustomSink.emit_async`, and accurate population statistics over the full result set.

---

## Trial Tests with xdist

`@pytest.mark.trial(n=, threshold=)` clones a test into N independent runs. Under xdist, clones may be distributed across workers depending on the `--dist` mode.

| `--dist` mode | Trial behavior |
|---------------|----------------|
| `loadgroup` | All trial clones for one test run on the same worker (recommended for locality) |
| `load` (default) | Trial clones distributed round-robin across workers |
| `loadscope` / `loadfile` | Grouped by class/module/file |

**Correctness is preserved regardless of mode** — the controller aggregates trial groups from the merged result set. You'll see a warning if you use `@trial` markers without `--dist=loadgroup`:

```text
RAMPART @trial markers present with --dist=load. Trial clones may be
split across workers. Aggregation remains correct (controller merges
all results), but using --dist=loadgroup keeps trial clones co-located
on one worker for better locality.
```

To silence the warning and improve locality:

```bash
pytest -n 4 --dist=loadgroup
```

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

The legacy `rampart_sinks` fixture is still supported as a **single-process
fallback**. The rule is:

- If **any** `pytest_rampart_sinks` hook implementation exists, the hook is
  authoritative and the fixture path is skipped entirely (so a project that
  defines both does **not** double-register).
- If **no** hook implementation exists, RAMPART falls back to the fixture. On the
  xdist controller this fallback scans registered conftest modules for a
  `rampart_sinks` attribute.

### Fixture fallback constraints (no hook present)

When you rely on the fixture fallback under xdist, the controller cannot execute
fixtures, so only these shapes resolve:

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

## Trust Boundary & Security

Worker payloads cross a process boundary via `execnet` and may contain attacker-controlled content (agent responses, payload text, evaluator rationale). RAMPART's serialization defends against:

- **Arbitrary code execution** — strict JSON-safe primitives only; no `pickle`, `marshal`, or custom `__reduce__`.
- **Schema drift** — payloads with missing or unknown schema versions are rejected fail-closed.
- **Memory exhaustion** — worker payloads are capped at 64 MB by default.
- **Terminal/log injection** — ANSI escape sequences are stripped from free-form text at the deserialization boundary.
- **Path traversal** — worker-local artifact paths are stored as opaque strings in metadata; the controller never accesses worker files.

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

Workers that exceed the cap log a warning and emit a truncation marker. The controller records the affected worker as incomplete in `TestRunReport.metadata`.

---

## Incomplete Runs

If a worker crashes, runs out of time, or hits the size cap, the controller marks the run as incomplete:

```python
report.metadata["incomplete"]            # True if any worker failed
report.metadata["incomplete_reasons"]    # list[str] — one per failure
```

Reports are still emitted with whatever data was collected. For safety-critical CI, sinks or post-processing should check the `incomplete` flag and fail the build accordingly.

---

## Run-Mode Metadata

Reports produced under xdist include:

```python
report.metadata["xdist_active"]   # True
report.metadata["worker_count"]   # int
report.metadata["dist_mode"]      # "load", "loadgroup", etc.
```

---

## Durability limitations

The current worker→controller transport flushes a worker's results only at its
clean `pytest_sessionfinish`. This has two consequences you should be aware of:

- **A worker killed mid-run loses its already-finished results.** Because results
  are shipped in a single batch at session end, a worker that crashes, is killed
  (e.g. OOM, timeout, `-x` shutdown), or otherwise never reaches
  `pytest_sessionfinish` contributes **nothing** — even tests it had already
  completed. The run is marked incomplete (see [Incomplete Runs](#incomplete-runs)).
- **The size cap drops the whole worker payload, not just the oversized record.**
  When a worker's aggregate serialized payload exceeds
  `--rampart-xdist-max-bytes`, the **entire** worker payload is dropped (and the
  worker marked incomplete), rather than only the single oversized transcript.

Both behaviors are deliberate fail-closed choices for this release. A durable
per-worker transport (incremental JSONL shards that survive a killed worker, with
the size cap applied per-record) is in progress as a follow-up change; until it
lands, prefer `--dist=loadgroup` for trial locality and size your cap to your
largest expected worker payload.

---

## Limitations

- Sinks discovered through the **fixture fallback** on the controller cannot depend
  on other pytest fixtures — use the `pytest_rampart_sinks` hook instead (see
  [Registering Sinks](#registering-sinks-the-pytest_rampart_sinks-hook)).
- Results from a worker that dies before `pytest_sessionfinish` are lost, and an
  over-cap worker payload is dropped wholesale (see
  [Durability limitations](#durability-limitations)).
- Mixed RAMPART versions across controller and workers are unsupported; install the
  same version everywhere.
- `pytest-xdist` itself does not support interactive debugging (`--pdb`, `--trace`);
  use single-process mode for debugging.
