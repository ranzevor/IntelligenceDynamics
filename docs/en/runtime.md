# Execution management

[中文](../runtime.md) · [Home](../../README.en.md)

## Local commands

```bash
intelligence-dynamics run configs/local-command.json local-run
```

The output directory must not exist. The runner stores `request.json`, `stdout.log`, `stderr.log`, and `run.json`. Commands use argument arrays rather than shell interpolation. By default, request metadata stores only the command digest and argument count. Set `record_command=true` to retain explicitly reviewed arguments. Results include exit code, elapsed time, declared artifact sizes and SHA256 hashes, and sampled RSS maxima for the direct process and its process tree when available. Memory maxima come from RSS observations collected during sampling; unobserved memory is `null`. Measurement errors are recorded separately from program exit status.

Artifact formats include binary, JSON, measurement JSONL, SQLite, NPZ, and LZMA. `Artifact` can declare required JSON keys, SQLite tables, NPZ arrays, axis lengths, and sequence-change counts. Expanded containers are bounded by `max_bytes`, which defaults to 256 MiB. LZMA inspection validates the container without deserializing its contents. SQLite artifacts require a completed checkpoint and no nonempty WAL or rollback journal before read-only `quick_check`. NPZ rejects object arrays and non-finite values. Artifact paths must be canonical relative paths within the output directory, without symbolic links or Windows reparse points. Format validation, the reported size, and SHA-256 use the same bounded snapshot; changes to the original during inspection are rejected.

Timeouts terminate the POSIX process group; the group is also cleaned up after command exit. Windows termination covers the direct child. Use a host Job Object or container for Windows process-tree isolation.

The `runtime` field in `request.json` records the package version, Python version, operating system, and architecture. Task status follows command exit, declared artifact inspection, and process and log cleanup; other working files are outside artifact acceptance. If `run.json` cannot be saved, `run_command` raises `RunPersistenceError`, whose `result` attribute retains the completed outcome.

The `environment` argument of `run_command` supplies non-sensitive experiment overrides. Requests retain only override keys and a canonical-content digest, not plaintext values. Inherited system environment is excluded. Reproduction supplies the same command configuration and explicit settings, and separately fixes the system environment used by the model. Runner diagnostics hide known absolute command paths, the output path, and the user home directory. Program stdout, stderr, and business artifacts are preserved verbatim, without content redaction.

## Job queue

`runtime.Queue` stores requests, states, and transition history in SQLite. Repeating the same request_id and specification returns the same job. Reusing an ID with different content is an error. Leases carry random tokens; expired or superseded tokens cannot update state.

`enqueue(..., requires=[...])` declares existing jobs as prerequisites. Submission waits until every prerequisite succeeds; failed prerequisites do not release dependent work.

Each controller tick handles one eligible job using a persisted polling order. Claimed jobs move to the tail, and new jobs join the tail. The first cancellation request and an explicit cancellation retry each promote the job once; repeated cancellation requests do not keep moving it ahead. Blocked cancellation, unknown submission results, and cleanup waits remain in polling without blocking independent jobs. Opening an older database transactionally adds polling order from the original enqueue sequence while preserving identities, states, history, and leases.

```text
QUEUED → SUBMITTING → SUBMITTED / RUNNING
RUNNING → CLEANUP_WAIT → ARTIFACT_WAIT → SUCCEEDED / FAILED
Cancellation: CANCEL_SUBMITTING → CANCEL_REQUESTED → CLEANUP_WAIT → CANCELLED
Not accepted: CANCEL_SUBMITTING → CANCEL_RETRY / CANCEL_BLOCKED
```

Submission intent is persisted first. After an unknown response, the next tick queries the stable provider_id and checks its specification-digest label. When the job is not observable and cancellation has not been requested, creation may be retried with the same job and idempotency identities. The submission result remains unknown until a provider job state is retrieved. Cancellation stops further creation attempts while queries continue to reconcile late submission results.

A cancellation call first persists `CANCEL_SUBMITTING` and its attempt count. Provider acceptance becomes `CANCEL_REQUESTED`; subsequent ticks only query status and cleanup. Timeouts, connection failures, and recoverable HTTP errors enter `CANCEL_RETRY`, with a fresh remote query before every retry and a default budget of three attempts. Rejection or exhausted retries enters `CANCEL_BLOCKED`, which continues monitoring without declaring cancellation successful. After addressing the cause, `queue.retry_cancellation(request_id, now=...)` permits another bounded set of attempts. Providers must support idempotent cancellation of the stable job identity.

Local `CANCELLED` requires either cancellation before submission or an explicit provider `CANCELLED` state plus an empty resource check. A temporarily missing job after accepted cancellation remains under polling. A remote completion that wins the race retains its actual success or failure outcome. Provider success additionally requires resource cleanup and artifact validation. Artifact checks default to three attempts, including exceptions; exhaustion marks the job failed. `execution_timeout` starts at the first submission intent and triggers cancellation on expiry, with a maximum of seven days.

`Controller` explicitly receives provider, resources_present, and artifacts_valid. resources_present queries whether job resources remain, and artifacts_valid checks whether declared artifacts satisfy their content contracts. Both callbacks return booleans. Lease duration must cover the provider call and artifact checks. Expired workers cannot commit results.

## Google Batch

`configs/batch-job.json` is a structure template. Copy it to machine-private configuration and replace `image` with an immutable reference obtained from the image provider at runtime before generating a request:

```bash
intelligence-dynamics batch-spec local-config/batch-job.json batch-job.json
```

This command only constructs a request body. Replace the sample project, bucket, and image with actual resources. Images require SHA256 digest references. Task retries are fixed to zero, and deadlines cannot exceed seven days.

`job_spec` supports multiple tasks, parallelism, and per-task environment variables. `estimate_timeout` scales an observed duration by workload, margin, and an explicit cap.

`BatchClient` provides get, create, and cancel. Construction neither obtains credentials nor opens a connection. Each request obtains a short-lived token from the caller's token_provider and uses the fixed HTTPS Batch API. Creation uses a stable jobId and UUID requestId. HTTP redirects are refused.

`GoogleInspector` paginates compute-instance queries using the exact job label and checks JSON objects against required keys and canonical-content digests. Pagination cycles, query failures, and unresolved warnings raise errors and stop the current inspection.

An integration supplies the execution identity, image access, output bucket, resource-cleanup query, and artifact validator. `resources_present` queries compute resources belonging to the job; `artifacts_valid` checks the expected objects against their content contracts.

API references: [Batch v1 Job](https://docs.cloud.google.com/batch/docs/reference/rest/v1/projects.locations.jobs), [Create job](https://docs.cloud.google.com/batch/docs/reference/rest/v1/projects.locations.jobs/create), [Cloud Storage bucket names](https://docs.cloud.google.com/storage/docs/buckets).

## Control service

`runtime.Application(controller, bearer_token=...)` is a WSGI application. Mount it in an organizational WSGI container; importing it does not start a service.

|Endpoint|Method|Input|
|---|---|---|
|`/health`|GET|None|
|`/jobs`|POST|request_id, specification|
|`/describe`|POST|request_id|
|`/cancel`|POST|request_id|
|`/tick`|POST|None|

Every endpoint requires a bearer token. Request bodies are limited to 64 KiB. Responses expose job identity and state, not tokens, internal exceptions, or full specifications. Serve over TLS and use a service manager for process lifetime, persistent queue storage, and resource quotas.
