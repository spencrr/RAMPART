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

## Constraints on `rampart_sinks`

When running under xdist, the controller process does not execute test fixtures. To discover your sinks, RAMPART scans registered conftest modules for a `rampart_sinks` attribute and calls it directly.

**Supported shapes:**

```python
# Parameterless fixture — works on both single-process and xdist
@pytest.fixture(scope="session")
def rampart_sinks():
    return [JsonFileReportSink(output_dir=Path(".report"))]

# Plain list assigned at module level — works on both
rampart_sinks = [JsonFileReportSink(output_dir=Path(".report"))]
```

**Not supported under xdist** (the warning is logged and the sink is skipped):

```python
# Fixture with dependencies — cannot be resolved on the controller
@pytest.fixture(scope="session")
def rampart_sinks(my_sink_config, db_connection):
    return [DatabaseSink(connection=db_connection)]
```

If your sinks need dependencies, consider:

- Constructing them at module level with explicit configuration
- Reading configuration from environment variables inside a parameterless function
- Running without xdist (`pytest` instead of `pytest -n 4`) until a hook-based registration API is added

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

## Limitations

- Sinks discovered on the controller cannot depend on other pytest fixtures (see Constraints above).
- Mixed RAMPART versions across controller and workers are unsupported; install the same version everywhere.
- `pytest-xdist` itself does not support interactive debugging (`--pdb`, `--trace`); use single-process mode for debugging.

A hook-based sink registration API for complex sink configurations is a planned follow-up.
