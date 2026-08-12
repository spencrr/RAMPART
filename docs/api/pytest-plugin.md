# API Reference — pytest Plugin

RAMPART's pytest integration. Activates automatically when installed.

::: rampart.pytest_plugin._collection
    options:
      members:
        - record_result
        - ResultCollectionHandler
        - ResultCollector

::: rampart.pytest_plugin._session
    options:
      members:
        - RampartSession
        - TrialGroupResult

## Parallel Execution Hooks

When `pytest-xdist` is installed, a private per-config runtime consumes
authoritative v2 teardown-report envelopes, appends validated Results
incrementally, and reconciles clean-finish manifests before report emission.
`config.workeroutput` carries only temporary deprecated clone metadata, never a
bulk Result fallback. See [Parallel Execution](../usage/xdist.md) for the data
flow, validation, durability, and evidence-fidelity behavior.

::: rampart.pytest_plugin._xdist
    options:
      members:
        - TRIAL_SPECS_SCHEMA_VERSION
        - TRIAL_SPECS_WORKEROUTPUT_KEY
        - SIZE_LIMIT_OPTION
        - DEFAULT_SIZE_LIMIT_BYTES
        - WorkerOutputError
        - SchemaVersionError
        - is_xdist_worker
        - is_xdist_controller
        - get_dist_mode
        - get_worker_count
        - deserialize_trial_specs
        - finalize_trial_specs_worker
        - discover_sinks_from_conftest

::: rampart.pytest_plugin._xdist_transport
    options:
      members:
        - SCHEMA_VERSION
        - REPORT_ATTRIBUTE
        - MANIFEST_KEY
