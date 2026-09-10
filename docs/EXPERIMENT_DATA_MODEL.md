# Cumulative experiment data model

SQLite at `data/experiments.sqlite3` is append-only during normal operation.
The file is runtime data and is ignored by Git. Tests use temporary databases.

`batches` records batch/experiment IDs, UTC creation time, mode, base seed,
requested run count, complete batch configuration and schema/physics/scenario
versions. `runs` records an individual run ID, pair index/variant, resolved
scenario and pipeline, algorithm versions, effective and subsystem seeds,
Git commit when available, duration, sampled inputs, resolved configuration, metrics, evidence-based failure
cause, completion status and optional telemetry.

Individual runs remain queryable. API history filtering currently supports
scenario and estimator; the store also supports scenario preset, controller,
predictor, failure, variant, batch and version filters. More UI filters can be
added without changing storage.

Default historical queries require current `simulation_schema_version` and
`physics_model_version`. Cross-version records remain stored and are returned
only with `include_incompatible=true`. Scenario version is stored for explicit
filtering; changing physical meaning should also increment the physics version.

Normal batches retain per-run summaries and important counts. Selected or failed
runs may use `summary_only=false` for bounded raw telemetry. This avoids storing
every image or an unbounded trace. Images and model weights are not stored in the
experiment database.
