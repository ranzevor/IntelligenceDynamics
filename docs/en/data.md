# Measurement and data contract

[中文](../data.md) · [Home](../../README.en.md)

## Measurement streams

Each line is a UTF-8 JSON object containing `record_type` and `protocol_version: "0.1"`. Records are limited to 1 MiB. Duplicate keys, unknown fields, empty lines, non-finite numbers, invalid UTF-8, and boolean counters are errors.

|Type|Purpose|Main fields|
|---|---|---|
|`RunMetadata`|Identify a run|run_id, track, system_name, environment_name, seed, source_revision, config_digest|
|`StepRecord`|A measurement point|run_id, step, lifetime, capability, computation, memory, routing|
|`AdaptationEvent`|An environment or objective change|event_id, step, event_type, hidden_from_agent, target_recovery_fraction|
|`InterventionRecord`|Mechanism intervention and deltas|run_id, mechanism, intervention_type, start_step, end_step, delta_capability|

`track` is A, B, C, AB, or cross. A stream starts with unique `RunMetadata` for each run. Steps increase strictly and lifetime never decreases within that run. Events have their own event_id; they do not implicitly belong to the preceding run record.

`read_records` checks individual records. CLI `validate` also checks run identities and temporal ordering. `to_object` and `from_object` convert between typed records and JSON objects.

## Units and missing values

- `step` is a nonnegative integer measurement index. The run protocol defines lifetime units; converters use environment interactions.
- Tasks define capability, reward, and loss units and directions. Values from different tasks are not directly additive.
- `ComputationCounters` separates unit updates, edge events, matrix operations, spikes, recurrence, routing, and learning updates.
- `MemoryAccounting` separates persistent state, parameters, replay, external memory, and read/write bytes.
- `RoutingSummary` records active/available modules, units, and edges, reuse, allocation, and routing entropy.
- Unobserved values are `null`, not zero. Activation ratios are `None` when potential counts are zero.
- `plasticity_probe_score` is the observed result of a specified probe. Weight changes, rewards, and update counts do not automatically become plasticity scores.

## Metrics

`metrics.recovery(steps, values, baseline=..., floor=..., fraction=...)` computes a recovery threshold and normalized area for higher-is-better capability.

```text
target = floor + fraction × (baseline − floor)
normalized = (capability − floor) / (baseline − floor)
```

The baseline must exceed the floor. Area uses trapezoidal integration with actual step spacing, divided by the observed span. A single point returns its normalized value. Observations above baseline are not clipped. `first_recovery_step` is the absolute first step meeting the threshold, or `None` when none does.

`metrics.plasticity` divides probe scores by a selected nonzero reference. `metrics.observations` reports observed and missing counts, returning a mean only for complete observations. `metrics.pareto_front` takes explicit maximizing and minimizing objectives, rejects incomplete points, and returns original indices not strictly dominated by another point.

## File output

`write_records` and `write_json` serialize and fsync a temporary file in the destination directory before publication. Existing outputs are refused by default. Explicit `overwrite=True` atomically replaces them. A failed generator does not publish a partial file.

`append_records` validates the current stream and atomically replaces it with the extended stream. It is a single-writer interface with memory use proportional to the complete stream. Symbolic-link paths and non-regular targets are rejected.

## Python example

```python
from intelligence_dynamics.records import ComputationCounters, StepRecord
from intelligence_dynamics.storage import write_records

record = StepRecord(
    run_id="example",
    step=1,
    lifetime=1.0,
    capability=0.75,
    computation=ComputationCounters(learning_updates=2),
)
write_records("one-step.jsonl", [record])
```

This is an individual-record file. Prepend its `RunMetadata` when producing a complete validated run stream.

## External model state

`state.state_snapshot` accepts scalars, lists, tuples, Mapping objects, dataclasses, and NumPy-convertible numeric arrays. Digests retain types, mapping keys, array dtype, shape, and C-order byte identity. Observation is read-only; it neither deserializes checkpoints nor updates a model.

`tensor_elements` and `tensor_bytes` sum logical elements and storage sizes across tensor leaves. Views sharing underlying storage are counted separately, so these values are not deduplicated physical memory. Execution-management interfaces measure actual process RSS.

`state_differences` returns field paths. `verify_state_intervention` confines changes to declared path prefixes, allowing an external model's carry reset to be checked while parameters and optimizer state remain unchanged. Model adapters provide data loading and framework-specific restoration.

`configs/measurement-record.schema.json` provides a structural JSON Schema. Runtime contracts additionally check relationships between counts, stream ordering, and finite numbers.
