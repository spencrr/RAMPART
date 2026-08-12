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

### V2 shadow validation

RAMPART also sends a **shadow-only** v2 copy of each completed item's Results on
its teardown `TestReport`. This incremental channel validates a future
transport design; it does not supply Results to terminal output, JSON reports,
sinks, trial gates, or population statistics. The v1 bulk worker payload above
remains authoritative for every user-visible report.

Each nonempty item attempt carries one private JSON envelope:

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

On a clean no-drop run, the controller compares semantic v1 and v2 Result
multisets. Only RAMPART's node, result-index, and source-worker bookkeeping is
excluded; Result content and multiplicity, including textual evidence, must
match. Any divergence or v2 drop forces an otherwise-successful run to exit
nonzero through the same incomplete-run enforcement used by v1 failures. The
v1 Result population is never replaced or supplemented by shadow data.

---

## Trial Tests with xdist

`@pytest.mark.trial(n=, threshold=1.0)` clones a test into N independent runs. Under xdist, clones may be distributed across workers depending on the `--dist` mode.

| `--dist` mode | Trial behavior |
|---------------|----------------|
| `each` | Every worker executes every trial clone |
| `loadgroup` | All trial clones for one test pinned to the same worker |
| `load` (default) | Trial clones distributed across all workers |
| `loadscope` / `loadfile` | Grouped by class/module/file |

**Correctness is preserved regardless of mode** — the controller retains the
merged result population before evaluating logical trial-clone groups. You'll
see a warning if you use `@trial` markers without `--dist=loadgroup`:

```text
RAMPART @trial markers present with --dist=load. Trial clones may be
split across workers. Aggregation remains correct (controller merges
all results), but using --dist=loadgroup keeps trial clones co-located
on one worker for better locality.
```

This warning is **informational, not a correctness signal** — see below for when it's safe to ignore.

`--dist=each` deliberately creates multiple genuine executions with the same
node ID. Every execution is retained in `TestRunReport.results` and
`total_runs`. The temporary clone-based trial gate still uses the number of
logical clones as its denominator rather than multiplying it by worker count;
multiple Results for one clone are evaluated fail-closed (`UNSAFE`, then
`ERROR`, then `SAFE`). This distinction will be replaced by the forthcoming
execution-domain trial API.

### Choosing `loadgroup` vs `load`

**Both modes produce an identical, correct report.** The controller merges per-worker
partials into one population and evaluates each trial's threshold against the full
group either way. The choice is about *execution*, not correctness:

- **`load` (default)** spreads a test's trial clones across **all** workers, so a
  20-clone trial keeps every worker busy. It is usually the **fastest** option and is
  the right default when trial clones are **independent** (no shared per-group state).
- **`loadgroup`** pins all clones of one trial group to a **single** worker. Prefer it
  only when a trial group needs **cohesion** — e.g. clones share a session-scoped
  fixture, a per-group cache/connection, or other worker-local state that must not be
  split across processes. The trade-off is less parallelism, so it can run slower.

**Rule of thumb:** independent trials → plain `pytest -n 4` (faster); trials that
share per-group worker state → `pytest -n 4 --dist=loadgroup`.

As an illustration, one 22-item suite containing a 20-clone trial measured:

| Mode | Command | Wall time | Reports | `total_runs` |
|------|---------|-----------|---------|--------------|
| Serial | `pytest -n 0` | 203.4s | 1 | 22 |
| Parallel, loadgroup | `pytest -n 4 --dist=loadgroup` | 165.5s | 1 | 22 |
| Parallel, default load | `pytest -n 4` | **113.8s** | 1 | 22 |

All three emit the same single report and the same trial verdict; `load` is fastest
here because the 20 clones fan out across the 4 workers instead of being pinned to one.

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

Worker results cross the process boundary through pytest-xdist's
`config.workeroutput`. RAMPART projects result data to JSON-safe primitives and
applies the following transport validation and normalization:

- **Schema version** — payloads with missing or unknown schema versions are rejected.
- **Trial metadata** — the current `rampart.xdist.v1` schema requires
  `trial_specs`; an empty list means the worker collected no trials. Missing or
  malformed trial metadata marks the run incomplete without discarding Results
  that already passed strict Result validation.
- **Size** — worker payloads are capped at 64 MB by default.
- **Value shape** — enum values, container depth, JSON-compatible metadata, and
  finite numeric values are validated or normalized.
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

Workers that exceed the cap log a warning and emit a truncation marker. The controller records the affected worker as incomplete in `TestRunReport.metadata`.

The same configured value also caps each v2 shadow report attribute without
changing the v1 rule above. V2 measures the actual compact JSON envelope in
UTF-8 bytes. An individually oversized Result becomes a bounded drop record
while fitting siblings remain; when combined Results exceed the cap, v2 keeps
a deterministic prefix and records the remainder as drops. The final
attribute never exceeds the cap and is never split across multiple report
properties. Every drop marks the shadow divergent/incomplete.

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

## Durability limitations

The current worker→controller transport flushes a worker's results only at its
clean `pytest_sessionfinish`. Because v1 remains authoritative, this has two
user-visible consequences:

- **A worker killed mid-run loses its already-finished results.** Because results
  are shipped in a single batch at session end, a worker that crashes, is killed
  (e.g. OOM, timeout, `-x` shutdown), or otherwise never reaches
  `pytest_sessionfinish` contributes **nothing** — even tests it had already
  completed. The run is marked incomplete (see [Incomplete Runs](#incomplete-runs)).
- **The size cap drops the whole worker payload, not just the oversized record.**
  When a worker's aggregate serialized payload exceeds
  `--rampart-xdist-max-bytes`, the **entire** worker payload is dropped (and the
  worker marked incomplete), rather than only the single oversized transcript.

The v2 shadow channel can retain Results whose teardown reports reached the
controller before a worker died, and its per-item cap can retain siblings around
an oversized Result. Those shadow Results remain internal on this release:
worker loss or a drop still marks the run incomplete, while emitted reports
continue to contain only the v1 population. This deliberate observation period
must reconcile cleanly before any later production cutover.

Until then, use `--dist=loadgroup` only when your trial groups need worker
cohesion (see [Choosing `loadgroup` vs `load`](#choosing-loadgroup-vs-load)) and
size the cap for both your largest expected worker payload and largest expected
single-item Result envelope.

---

## Limitations

- Sinks discovered through the **fixture fallback** on the controller cannot depend
  on other pytest fixtures — use the `pytest_rampart_sinks` hook instead (see
  [Registering Sinks](#registering-sinks-the-pytest_rampart_sinks-hook)).
- User-visible Results from a worker that dies before `pytest_sessionfinish`
  are lost, and an over-cap v1 worker payload is dropped wholesale. Incremental
  v2 copies may remain in internal shadow state but never backfill the report
  (see [Durability limitations](#durability-limitations)).
- Mixed RAMPART versions across controller and workers are unsupported; install the
  same version everywhere.
- `pytest-xdist` itself does not support interactive debugging (`--pdb`, `--trace`);
  use single-process mode for debugging.
