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

When `pytest-xdist` is installed, the plugin registers `pytest_testnodedown` (as an optional hook) to merge authoritative v1 worker results into the controller session. A private per-config runtime separately consumes v2 teardown-report envelopes and validates their semantic multiset against v1; it is an internal cutover gate, not a public Result or reporting API. See [Parallel Execution](../usage/xdist.md) for the data flow, validation, and evidence-fidelity behavior.

::: rampart.pytest_plugin._xdist
    options:
      members:
        - SCHEMA_VERSION
        - WORKEROUTPUT_KEY
        - SIZE_LIMIT_OPTION
        - DEFAULT_SIZE_LIMIT_BYTES
        - WorkerOutputError
        - SchemaVersionError
        - SizeLimitError
        - is_xdist_worker
        - is_xdist_controller
        - get_dist_mode
        - get_worker_count
        - serialize_worker_data
        - deserialize_worker_data
        - finalize_worker
        - handle_testnodedown
        - discover_sinks_from_conftest
