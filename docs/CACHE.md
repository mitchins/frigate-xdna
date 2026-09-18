# Cache, registry and model manifest

## 1. On-disk layout

```
/data/
  registry.sqlite3
  manager.lock
  device.lock
  sources/<source_sha256>/model.onnx
  sources/<source_sha256>/bundle.json        # only for supported external-data bundles
  metadata/<metadata_sha256>.json
  artifacts/<compile_key>/
    model.rai
    artifact.json
    compile.log                            # bounded/redacted
    reference-probes/                      # private, bounded
  validations/<validation_key>.json
  work/<job_uuid>/
  failures/<job_uuid>/                      # bounded diagnostic evidence
  quarantine/<compile_key>.json
  operations/current.json
  operations/history.jsonl                 # rotate/bound
```

Refs, wire aliases, pins, jobs and active bindings are registry records, not user-controlled symlink paths. A Plus ID or wire basename must never become an unsanitized filesystem path.

`work` and committed artifacts must be on the same local filesystem for atomic publication. SQLite WAL and locks must not live on CIFS/NFS. Back up with SQLite's backup API or while quiesced, plus the immutable objects; do not simply copy a live DB and omit its WAL.

## 2. Identity layers

### Source identity

Single ONNX: SHA-256 of exact bytes received, before Quark conversion. Do not substitute Frigate's separate MD5 display hash. [S4]

Multi-file ONNX import, only if explicitly implemented: digest a canonical sorted manifest of relative paths, lengths and SHA-256 values, including ONNX and all referenced external-data members. No traversal, symlink escape, HTTP external weights or absolute external paths.

### Compile identity

Canonical JSON, UTF-8, sorted keys and stable number/string encoding:

```json
{
  "schema": 1,
  "source_sha256": "<full digest>",
  "compiler_payload_sha256": "<audited full payload manifest digest>",
  "recipe_id": "bf16-vaiml-v1",
  "recipe_config_sha256": "<canonical flags/config digest>",
  "target_profile": "<validated Strix-Halo/AIE2P profile ID>",
  "artifact_compatibility_id": "<verified RAI format ABI>",
  "compile_input": {"shape": [1, 3, 320, 320], "dtype": "float32"}
}
```

Hash this object for `compile_key`. The examples intentionally do not invent missing toolchain hashes or numeric ABI IDs.

### Imported RAI identity

An imported precompiled RAI may have no available source ONNX. Its key is a separate namespace: SHA-256 of `kind=imported-rai`, artifact digest and declared target/format profile. Do not invent an ONNX source digest. It cannot satisfy an ONNX-transfer cache lookup unless an explicit descriptor supplies a verified source-to-artifact provenance link. An imported RAI still needs a serving contract and activation validation.

### Serving identity

Hash compile key + resolved input semantics + output tensor/decoder profile + decoder implementation version/options + labelMap/attributes metadata digest. This prevents serving the same graph under the wrong labels or normalizing twice without forcing recompilation for every display-name change.

### Validation identity

Hash artifact digest + serving identity + runtime compatibility profile + validation suite version. Record exact XRT/FlexML/driver/firmware observations. A changed runtime requires revalidation according to the compatibility policy; it does not automatically imply recompilation.

Do not use model ID, basename, path, timestamp, host BDF, boot ID or current image version as substitutes for content/compatibility keys.

## 3. Registry entities

* `model_refs`: canonical ref, kind, source digest, metadata digest, configured/manual pin, acquisition status; Plus IDs never cross-account-share artifacts publicly.
* `sources`: digest, size, internal path, last integrity check, origin type.
* `artifacts`: compile key, artifact digest/path, recipe/toolchain/target, compile duration/resource stats, creation time.
* `serving_contracts`: digest, compile key, resolved contract JSON.
* `validations`: validation key, pass/fail, reason, fixture IDs/digests, runtime profile, timestamp.
* `jobs`: UUID, source/ref/compile key, stage, attempt, timestamps, interruption/boot token, progress stage, error code.
* `pins`: active, configured, manual, rollback, running-job, quarantine-evidence.
* `service_state`: active artifact/generation and safety breaker state; never secrets.

Use schema migrations with an explicit version; back up before nontrivial changes. A reader must refuse a newer unsupported schema, not rebuild/delete it.

## 4. State machines

Acquisition/compile:

```
NEW -> FETCHING -> DOWNLOADED -> INSPECTED
    -> QUEUED -> [WAITING_FOR_DEVICE] -> COMPILING
    -> PREPARED
```

Validation/serving:

```
PREPARED -> ACTIVATING -> VERIFIED + ACTIVE
ACTIVE -> DRAINING -> INACTIVE_VERIFIED
```

Terminal/exception states:

```
AUTH_FAILED, DOWNLOAD_FAILED, UNSUPPORTED_CONTRACT,
COMPILE_FAILED, RESOURCE_EXCEEDED, VALIDATION_FAILED,
SOURCE_CHANGED, CACHE_CORRUPT, QUARANTINED, INTERRUPTED,
DEVICE_BUSY, DEVICE_FAULT, SAFETY_INHIBITED
```

A compiled artifact can exist while activation is not verified. Do not flatten these into `done`.

Retries: bounded HTTP retries for transient network failures; no automatic compiler retry for deterministic compiler failure/NaNs; no automatic retry after suspected host reset/device fault. Repeated same-key requests join one job. Different refs sharing the same compile key do not start duplicate jobs.

## 5. Transactional publication

1. Acquire per-key lock; recheck committed artifact.
2. Reserve scratch space and create a unique staging directory.
3. Persist input/recipe manifest and operation-start record.
4. Run the compiler, collect exit/resources and validate output format/hash.
5. Write final artifact manifest; fsync files and directory.
6. Rename the staging result into `artifacts/<compile_key>` on the same filesystem.
7. Commit SQLite state pointing to the immutable artifact.
8. Remove obsolete scratch; retain bounded failure logs and original source.

On crash, an output in `work` is not a successful cache entry. An immutable artifact published just before a DB crash may be adopted only after its manifest/hash is fully revalidated. Never overwrite a committed artifact in place.

Readers verify hashes on first use after daemon start and record provenance; corruption quarantines the entry. Never continue with a same-name alternative.

## 6. Cache retention

The default cache holds multiple models indefinitely subject to space; no fixed one-model cache. Active/configured/manual/rollback/evidence pins are retained.

`cache prune` reports potential deletions before `--apply`. Prune takes locks, excludes active/read leases, and removes registry references before deleting unreferenced content. Two aliases sharing bytes must not double-count reclaimable space.

Disk preflight uses measured recipe scratch requirements and a safety reserve. Initial reserve is the larger of 2 GiB and 10% of the data filesystem; expose the exact check in status. No emergency deletion of production models to rescue a new compile. If the filesystem cannot support the job, fail/defer it with an estimate of bytes required.

## 7. Private model contract

Never publish source/RAI from Plus in general cache registries, public Docker layers, CI artifacts or bug reports. The cache is private local user data. Export/report commands redact URLs/tokens and omit weights/probe images unless explicitly requested.

Model access authorization and model hashing are distinct. Knowing a hash is not permission to fetch another account's model.
